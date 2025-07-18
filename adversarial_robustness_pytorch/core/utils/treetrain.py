from core import animal_classes, vehicle_classes
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.metrics import accuracy, binary_accuracy, subclass_accuracy
from .model import create_model

from core.utils.mart import mart_loss, mart_tree_loss
from core.utils.rst import CosineLR
from core.utils.trades import trades_loss, trades_tree_loss

from core.models.treeresnet import lighttreeresnet

from .context import ctx_noparamgrad_and_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def focal_loss_with_pt(logits, targets, gamma=2.0, reduction='mean'):
    """
    Multi-class focal loss with pt recording.
    logits: [N, C]
    targets: [N]
    """
    log_probs = F.log_softmax(logits, dim=1)  # [N, C]
    probs = torch.exp(log_probs)              # [N, C]
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))  # [N, C]

    pt = (probs * targets_one_hot).sum(dim=1)      # [N]
    log_pt = (log_probs * targets_one_hot).sum(dim=1)  # [N]

    focal_term = (1 - pt) ** gamma
    loss = -focal_term * log_pt                    # [N]

    if reduction == 'mean':
        return loss.mean(), pt
    elif reduction == 'sum':
        return loss.sum(), pt
    else:
        return loss, pt  # no reduction
    

class TreeEnsemble(object):
    def __init__(self, 
        info, args,
        alpha1: float = 1,
        alpha2: float = 0.6,
        alpha3: float = 0.4,
        max_epochs: int = 100,  # Total number of training epochs
        alpha_update_strategy: dict = None,
        gamma: float = 2.0,  # Focal loss gamma parameter
    ):
        
        self.model = create_model(args.model, args.normalize, info, device)
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.gamma = gamma  # Focal loss gamma parameter

        self.max_epochs = args.num_adv_epochs
        self.alpha_update_strategy = alpha_update_strategy or {
            "balance_ratio": 1 / 1,  # alpha2:alpha3 ratio, e.g., 6:4 for animal vs vehicle
            # "balance_ratio": 1 / 1,
        }

        self.params = args
        #self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = focal_loss_with_pt
        # for kendall loss weight, set as trainable parameter 
        # self.s_r = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        # self.s_a = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        # self.s_v = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)

        self.init_optimizer(self.params.num_adv_epochs)
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))

        self.attack, self.eval_attack = self.init_attack(self.model, self.wrap_loss_fn, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)
        

    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
        
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler with different learning rates for root and subroot models.
        """
        self.optimizer = torch.optim.SGD([
            {"params": self.model.root_model.parameters(), "lr": self.params.lr },  # root: 学得稳定，稍小
            {"params": self.model.subroot_animal.parameters(), "lr": self.params.lr},    # subroot-animal: 正常
            {"params": self.model.subroot_vehicle.parameters(), "lr": self.params.lr },  # vehicle 学得慢一点，可以提速
            #{"params": [self.s_r, self.s_a, self.s_v], "lr": self.params.lr }  # kendall loss weight
        ],
            weight_decay=self.params.weight_decay, 
            momentum=0.9, 
            nesterov=self.params.nesterov
        )

        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler for different parameter groups.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples / self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[self.params.lr , self.params.lr , self.params.lr  ],
                pct_start=0.25,
                steps_per_epoch=update_steps,
                epochs=int(num_epochs),
            )
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, gamma=0.1, milestones=[100, 105]
            )
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[self.params.lr , self.params.lr , self.params.lr ],
                pct_start=0.025,
                total_steps=int(num_epochs),
            )
        else:
            self.scheduler = None

    def update_alphas(self, current_epoch: int, decay_factor: float=0.98, strategy: str = 'linear'):
        """
        linear update: Dynamically update alpha1, alpha2, and alpha3 based on the current epoch.
        """
        if strategy == 'linear':
            # linear update
            progress = current_epoch / self.max_epochs  # Calculate training progress (0 to 1)
            self.alpha1 = 0.9 - 0.8 * progress  # Decrease alpha1 from 0.9 to 0
            alpha23_total = 0.1 + 0.8 * progress  # Increase alpha2 + alpha3 from 0.1 to 0.9
        
            # Use a custom strategy if provided
            if callable(self.alpha_update_strategy):
                self.alpha2, self.alpha3 = self.alpha_update_strategy(alpha23_total, progress)
            else:
                # Split alpha23_total between alpha2 and alpha3 based on the balance ratio
                balance_ratio = self.alpha_update_strategy["balance_ratio"]
                self.alpha2 = alpha23_total * balance_ratio / (1 + balance_ratio)
                self.alpha3 = alpha23_total / (1 + balance_ratio)

                return self.alpha1, self.alpha2, self.alpha3
        if strategy == 'exponential':
            self.alpha1 *= decay_factor
            return self.alpha1, self.alpha2, self.alpha3
        

    def forward(self, x):
        root_logits, subroot_animal_logits, subroot_vehicle_logits = self.model(x)

        root_pred = torch.argmax(root_logits, dim=1)

        subroot_logits = torch.zeros_like(root_logits)
        animal_classes_index = torch.tensor(animal_classes, device=root_pred.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=root_pred.device)

        is_animal = torch.isin(root_pred, animal_classes_index)
        is_vehicle = torch.isin(root_pred, vehicle_classes_index)

        # Fix for animal subroot logits
        if is_animal.any():
            subroot_logits[is_animal] = subroot_animal_logits[is_animal]
        if is_vehicle.any():
            subroot_logits[is_vehicle] = subroot_vehicle_logits[is_vehicle]

        return subroot_logits

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Train each trainer on a given (sub)set of data.
        """
        
        metrics = pd.DataFrame()  # Initialize metrics
        self.model.train()

        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            # each batch
            x, y = data
            x, y = x.to(device), y.to(device)

            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y) 
            loss.backward()

            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])], ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        return dict(metrics.mean())
    
    
    def loss_fn(self, logits_set, y):
        """
        Loss function for the model with pt recording.
        """
        root_logits, logits_animal, logits_vehicle = logits_set
        root_loss, root_pt = focal_loss_with_pt(root_logits, y, gamma=self.gamma)
        subroot_loss_animal, subroot_pt_animal = torch.tensor(0.0, device=y.device), torch.tensor([], device=y.device)
        subroot_loss_vehicle, subroot_pt_vehicle = torch.tensor(0.0, device=y.device), torch.tensor([], device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch.isin(y, animal_classes_index)
        is_vehicle = torch.isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal, subroot_pt_animal = focal_loss_with_pt(logits_animal[is_animal], y[is_animal], gamma=self.gamma)
        if is_vehicle.any():
            subroot_loss_vehicle, subroot_pt_vehicle = focal_loss_with_pt(logits_vehicle[is_vehicle], y[is_vehicle], gamma=self.gamma)

        loss = self.alpha1*root_loss + self.alpha2*subroot_loss_animal + self.alpha3*subroot_loss_vehicle
        return loss, root_loss, subroot_loss_animal, subroot_loss_vehicle, root_pt, subroot_pt_animal, subroot_pt_vehicle
    
    def wrap_loss_fn(self, logits_set, y):
        """
        Wrapper for loss function to return only the loss value.
        """
        loss, _, _, _, _, _, _ = self.loss_fn(logits_set, y)
        return loss
    
    def KL_loss_fn(self, adv_logits_set, logits_set, y):
        """
        KL Loss function for tree model. In trades loss calculation.
        nn.KLDivLoss(reduction='sum') used not divided by batch size.
        """
        criterion = nn.KLDivLoss(reduction='sum')

        adv_root_logits, adv_logits_animal, adv_logits_vehicle = adv_logits_set #10dim, 10dim, 10dim
        root_logits, logits_animal, logits_vehicle = logits_set #10dim, 10dim, 10dim 
      
        root_loss = criterion(F.log_softmax(adv_root_logits, dim=1), F.softmax(root_logits, dim=1)) / len(y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch.isin(y, animal_classes_index)
        is_vehicle = torch.isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = criterion(F.log_softmax(adv_logits_animal[is_animal],dim=1), F.softmax(logits_animal[is_animal], dim=1)) / len(y[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle = criterion(F.log_softmax(adv_logits_vehicle[is_vehicle],dim=1), F.softmax(logits_vehicle[is_vehicle], dim=1)) / len(y[is_vehicle])

        loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        return loss

    def mart_loss_fn(self, adv_logits_set, logits_set, y):
        """
        Robust Loss function for tree model. In mart loss calculation.
        """
        kl = nn.KLDivLoss(reduction='none')

        adv_root_logits, adv_logits_animal, adv_logits_vehicle = adv_logits_set
        root_logits, logits_animal, logits_vehicle = logits_set

        adv_probs_root = F.softmax(adv_root_logits, dim=1)
        tmp1_root = torch.argsort(adv_probs_root, dim=1)[:, -2:]
        new_y_root = torch.where(tmp1_root[:, -1] == y, tmp1_root[:, -2], tmp1_root[:, -1])
        root_loss_adv = F.cross_entropy(adv_root_logits, y) + F.nll_loss(torch.log(1.0001 - adv_probs_root + 1e-12), new_y_root)

        nat_probs_root = F.softmax(root_logits, dim=1)
        true_probs_root = torch.gather(nat_probs_root, 1, (y.unsqueeze(1)).long()).squeeze()
        root_loss_robust = (1.0 / len(y)) * torch.sum(
            torch.sum(kl(torch.log(adv_probs_root + 1e-12), nat_probs_root), dim=1) * (1.0000001 - true_probs_root)
        )

        vehicle_loss_adv = torch.tensor(0.0, device=y.device)
        animal_loss_adv = torch.tensor(0.0, device=y.device)
        vehicle_loss_robust = torch.tensor(0.0, device=y.device)
        animal_loss_robust = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch.isin(y, animal_classes_index)
        is_vehicle = torch.isin(y, vehicle_classes_index)

        if is_animal.any():
            adv_probs_animal = F.softmax(adv_logits_animal[is_animal], dim=1)
            tmp1_animal = torch.argsort(adv_probs_animal, dim=1)[:, -2:]
            new_y_animal = torch.where(tmp1_animal[:, -1] == y[is_animal], tmp1_animal[:, -2], tmp1_animal[:, -1])
            animal_loss_adv = F.cross_entropy(adv_logits_animal[is_animal], y[is_animal]) + F.nll_loss(
                torch.log(1.0001 - adv_probs_animal + 1e-12), new_y_animal
            )
            
            nat_probs_animal = F.softmax(logits_animal[is_animal], dim=1)
            true_probs_animal = torch.gather(nat_probs_animal, 1, (y[is_animal].unsqueeze(1)).long()).squeeze()
            animal_loss_robust = (1.0 / len(y[is_animal])) * torch.sum(
                torch.sum(kl(torch.log(adv_probs_animal + 1e-12), nat_probs_animal), dim=1) * (1.0000001 - true_probs_animal)
            )
            
        if is_vehicle.any():
            adv_probs_vehicle = F.softmax(adv_logits_vehicle[is_vehicle], dim=1)
            tmp1_vehicle = torch.argsort(adv_probs_vehicle, dim=1)[:, -2:]
            new_y_vehicle = torch.where(tmp1_vehicle[:, -1] == y[is_vehicle], tmp1_vehicle[:, -2], tmp1_vehicle[:, -1])
            vehicle_loss_adv = F.cross_entropy(adv_logits_vehicle[is_vehicle], y[is_vehicle]) + F.nll_loss(
                torch.log(1.0001 - adv_probs_vehicle + 1e-12), new_y_vehicle
            )
            
            nat_probs_vehicle = F.softmax(logits_vehicle[is_vehicle], dim=1)
            true_probs_vehicle = torch.gather(nat_probs_vehicle, 1, (y[is_vehicle].unsqueeze(1)).long()).squeeze()
            vehicle_loss_robust = (1.0 / len(y[is_vehicle])) * torch.sum(
                torch.sum(kl(torch.log(adv_probs_vehicle + 1e-12), nat_probs_vehicle), dim=1) * (1.0000001 - true_probs_vehicle)
            )
        
        loss_adv = self.alpha1 * root_loss_adv + self.alpha2 * animal_loss_adv + self.alpha3 * vehicle_loss_adv
        loss_robust = self.alpha1 * root_loss_robust + self.alpha2 * animal_loss_robust + self.alpha3 * vehicle_loss_robust
        return loss_adv, loss_robust


    def standard_loss(self, x, y):
        """
        Standard training with pt recording.
        """
        self.optimizer.zero_grad()
        out = self.forward(x)
        root_logits, subroot_animal, subroot_vehicle = self.model(x)
        loss, root_loss, subroot_loss_animal, subroot_loss_vehicle, root_pt, subroot_pt_animal, subroot_pt_vehicle = self.loss_fn(
            [root_logits, subroot_animal, subroot_vehicle], y
        )

        preds = out.detach()
        batch_metrics = {
            'loss': loss.item(),
            'root_loss': root_loss.item(),
            'subroot_loss_animal': subroot_loss_animal.item(),
            'subroot_loss_vehicle': subroot_loss_vehicle.item(),
            'clean_acc': accuracy(y, preds),
            'root_pt': root_pt.mean().item(),
            'subroot_pt_animal': subroot_pt_animal.mean().item() if len(subroot_pt_animal) > 0 else None,
            'subroot_pt_vehicle': subroot_pt_vehicle.mean().item() if len(subroot_pt_vehicle) > 0 else None,
        }
        return loss, batch_metrics

    def adversarial_loss(self, x, y):
        """
        Adversarial training with pt recording.
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)

        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
        out = self.forward(x_adv)
        root_logits, subroot_animal, subroot_vehicle = self.model(x_adv)
        loss, root_loss, subroot_loss_animal, subroot_loss_vehicle, root_pt, subroot_pt_animal, subroot_pt_vehicle = self.loss_fn(
            [root_logits, subroot_animal, subroot_vehicle], y_adv
        )

        preds = out.detach()
        batch_metrics = {
            'loss': loss.item(),
            'root_loss': root_loss.item(),
            'subroot_loss_animal': subroot_loss_animal.item(),
            'subroot_loss_vehicle': subroot_loss_vehicle.item(),
            'root_pt': root_pt.mean().item(),
            'subroot_pt_animal': subroot_pt_animal.mean().item() if len(subroot_pt_animal) > 0 else None,
            'subroot_pt_vehicle': subroot_pt_vehicle.mean().item() if len(subroot_pt_vehicle) > 0 else None,
        }
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy(y, preds)})
        return loss, batch_metrics
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training. TO DO ... 
        """
        # loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
        #                                   epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
        #                                   beta=beta, attack=self.params.attack)
        loss, batch_metrics = trades_tree_loss(self.model, self.forward, self.KL_loss_fn, self.wrap_loss_fn, x, y, self.optimizer,
                                                  step_size=self.params.attack_step, 
                                                  epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                                  beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training. TO DO ...
        """
        loss, batch_metrics = mart_tree_loss(self.model, self.forward, self.mart_loss_fn, self.wrap_loss_fn, x, y, self.optimizer,
                                                    step_size=self.params.attack_step, 
                                                    epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                                    beta=beta, attack=self.params.attack)

        return loss, batch_metrics  

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc, acc_animal, acc_vehicle = 0.0, 0.0, 0.0
        root_acc, root_acc_animal, root_acc_vehicle = 0.0, 0.0, 0.0
        root_acc_bi = 0.0

        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                root_out, subroot_animal, subroot_veicle = self.model(x_adv)
                out = self.forward(x_adv)
            else:
                root_out, subroot_animal, subroot_veicle = self.model(x)
                out = self.forward(x)
                
            acc += accuracy(y, out)
            temp_acc_animal, temp_acc_vehicle = subclass_accuracy(y, out)
            acc_animal += temp_acc_animal
            acc_vehicle += temp_acc_vehicle

            # 10 classes
            root_acc += accuracy(y, root_out)
            temp_root_acc_animal, temp_root_acc_vehicle = subclass_accuracy(y, root_out)
            root_acc_animal += temp_root_acc_animal
            root_acc_vehicle += temp_root_acc_vehicle
            # 2 classes based on animal_classes(1) or vehicle_classes(0)
            root_acc_bi += binary_accuracy(y, root_out)

        acc /= len(dataloader)
        root_acc /= len(dataloader)
        root_acc_bi /= len(dataloader)
        acc_animal /= len(dataloader)
        acc_vehicle /= len(dataloader)
        
        return dict(
            acc=acc,
            acc_animal=acc_animal,
            acc_vehicle=acc_vehicle,
            root_acc=root_acc,
            root_acc_animal=root_acc_animal,
            root_acc_vehicle=root_acc_vehicle,
            root_acc_bi=root_acc_bi
        )
    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
