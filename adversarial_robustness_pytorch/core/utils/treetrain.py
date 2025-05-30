from core import animal_classes, vehicle_classes
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import torch
import torch.nn as nn


from core.attacks import create_attack
from core.metrics import accuracy, binary_accuracy, subclass_accuracy
from core.models import create_model

from core.models import Normalization
from .mart import mart_loss
from .rst import CosineLR
from .trades import trades_loss

from core.models.treeresnet import lighttreeresnet

from .context import ctx_noparamgrad_and_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TreeEnsemble(object):
    def __init__(self, 
        info, args,
        alpha1: float = 0.9,
        alpha2: float = 0.1,
        alpha3: float = 0.1,
        max_epochs: int = 100,  # Total number of training epochs
        alpha_update_strategy: dict = None,
    ):
        
        self.model = self.create_tree_model(info, args, device)
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.max_epochs = max_epochs
        self.alpha_update_strategy = alpha_update_strategy or {
            "balance_ratio": 2 / 3,  # alpha2:alpha3 ratio, e.g., 6:4 for animal vs vehicle
        }

        self.params = args
        self.criterion = nn.CrossEntropyLoss()

        self.init_optimizer(self.params.num_adv_epochs)
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))

        self.attack, self.eval_attack = self.init_attack(self.model, self.loss_fn, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)
        

    def create_tree_model(self, info, args, device):
        model = lighttreeresnet(args.model, num_classes=info['num_classes'], device=device)
    
        if args.normalize:
            normalization_layer = Normalization(info['mean'], info['std']).to(device)
            model.root_model = torch.nn.Sequential(normalization_layer, model.root_model)
            model.subroot_animal = torch.nn.Sequential(normalization_layer, model.subroot_animal)
            model.subroot_vehicle = torch.nn.Sequential(normalization_layer, model.subroot_vehicle)
        else:
            model.root_model = torch.nn.Sequential(model.root_model)
            model.subroot_animal = torch.nn.Sequential(model.subroot_animal)
            model.subroot_vehicle = torch.nn.Sequential(model.subroot_vehicle)
        model = model.to(device)
        return model
    
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
    
    # @staticmethod
    # def init_attack(model, tree_criterion, attack_type, attack_eps, attack_iter, attack_step):
        
    #     class wrapper(object):
    #         def __init__(self, model):
    #             self.model = model

    #         def forward(self, x):
    #             root_logits, subroot_animal_logits, subroot_vehicle_logits = self.model(x)
    #             root_pred = torch.argmax(root_logits, dim=1)

    #             subroot_logits = torch.zeros_like(root_logits)
    #             animal_classes_index = torch.tensor(animal_classes, device=root_pred.device)
    #             vehicle_classes_index = torch.tensor(vehicle_classes, device=root_pred.device)

    #             is_animal = root_pred.unsqueeze(1) == animal_classes_index
    #             is_animal = is_animal.any(dim=1)

    #             is_vehicle = root_pred.unsqueeze(1) == vehicle_classes_index
    #             is_vehicle = is_vehicle.any(dim=1)

    #             # Fix for animal subroot logits
    #             if is_animal.any():
    #                 subroot_logits[is_animal] = subroot_animal_logits[is_animal]

    #             if is_vehicle.any():
    #                 subroot_logits[is_vehicle] = subroot_vehicle_logits[is_vehicle]

    #             return subroot_logits
            
    #         def __call__(self, x):
    #             return self.forward(x)
        
    #     wrapper_model = wrapper(model)
    #     criterion = nn.CrossEntropyLoss()

    #     # Initialize attack
    #     if attack_type in ['linf-df', 'l2-df']:
    #         attack = create_attack(wrapper_model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
    #     else:
    #         attack = create_attack(model, tree_criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        
    #     # Initialize evaluation attack
        

    #     if attack_type in ['linf-pgd', 'l2-pgd']:
    #         eval_attack = create_attack(wrapper_model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
    #     elif attack_type in ['fgsm', 'linf-df']:
    #         eval_attack = create_attack(wrapper_model, criterion, 'linf-pgd', 8/255, 20, 2/255)
    #     elif attack_type in ['fgm', 'l2-df']:
    #         eval_attack = create_attack(wrapper_model, criterion, 'l2-pgd', 128/255, 20, 15/255)
    #     return attack, eval_attack
        
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler with different learning rates for root and subroot models.
        """
        self.optimizer = torch.optim.SGD([
            {"params": self.model.root_model.parameters(), "lr": self.params.lr * 0.5},  # root: 学得稳定，稍小
            {"params": self.model.subroot_animal.parameters(), "lr": self.params.lr},    # subroot-animal: 正常
            {"params": self.model.subroot_vehicle.parameters(), "lr": self.params.lr * 1.5},  # vehicle 学得慢一点，可以提速
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
                max_lr=[self.params.lr * 0.5, self.params.lr * 1.0, self.params.lr * 1.5],
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
                max_lr=[self.params.lr * 0.5, self.params.lr * 1.0, self.params.lr * 1.5],
                pct_start=0.025,
                total_steps=int(num_epochs),
            )
        else:
            self.scheduler = None

    def update_alphas(self, current_epoch: int, root_acc: float):
        """
        Dynamically update alpha1, alpha2, and alpha3 based on the current epoch.
        """
        progress = current_epoch / self.max_epochs  # Calculate training progress (0 to 1)
        self.alpha1 = max(0.0, 0.9 * (1 - progress))  # Decrease alpha1 from 0.9 to 0
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

    def forward(self, x):
        root_logits, subroot_animal_logits, subroot_vehicle_logits = self.model(x)

        root_pred = torch.argmax(root_logits, dim=1)

        subroot_logits = torch.zeros_like(root_logits)
        animal_classes_index = torch.tensor(animal_classes, device=root_pred.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=root_pred.device)

        is_animal = root_pred.unsqueeze(1) == animal_classes_index
        is_animal = is_animal.any(dim=1)

        is_vehicle = root_pred.unsqueeze(1) == vehicle_classes_index
        is_vehicle = is_vehicle.any(dim=1)


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
        Loss function for the model.
        """
        root_logits, logits_animal, logits_vehicle = logits_set #10dim, 10dim, 10dim 
      
        root_loss = self.criterion(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch.isin(y, animal_classes_index)
        is_vehicle = torch.isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = self.criterion(logits_animal[is_animal], y[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle = self.criterion(logits_vehicle[is_vehicle], y[is_vehicle])

        loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        return loss

    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        out = self.forward(x)
        root_logits, subroot_animal, subroot_veicle = self.model(x)
        loss = self.loss_fn([root_logits, subroot_animal, subroot_veicle], y)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)

        #self.model.train()
        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
        out = self.forward(x_adv)
        root_logits, subroot_animal, subroot_veicle = self.model(x_adv)
        loss = self.loss_fn([root_logits, subroot_animal, subroot_veicle], y_adv)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item()}
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
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training. TO DO ...
        """
        loss, batch_metrics = mart_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
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
