from .train import Trainer
from core import animal_classes, vehicle_classes
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import torch
import torch.nn as nn

from .rst import CosineLR

from core.metrics import accuracy
from core.models import create_model

from .context import ctx_noparamgrad_and_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
    """
    Replacement for torch.isin for PyTorch < 1.10
    """
    test_elements = test_elements.to(elements.device)

    elements_shape = elements.shape
    elements_flat = elements.flatten()
    test_elements_flat = test_elements.flatten()
    result = (elements_flat[..., None] == test_elements_flat).any(-1)
    return result.reshape(elements_shape)

class TreeEnsemble(object):
    def __init__(self, 
        info, args,
        alpha1: float = 0.9,
        alpha2: float = 0.1,
        alpha3: float = 0.1,
        max_epochs: int = 100,  # Total number of training epochs
        alpha_update_strategy: dict = None,
    ):
        
        self.model = create_model(args.model, args.normalize, info, device)
        self.root_trainer = Trainer( info, args, self.model.root_model)
        self.animal_trainer = Trainer( info, args, self.model.subroot_animal)
        self.vehicle_trainer = Trainer( info, args, self.model.subroot_vehicle)
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.max_epochs = max_epochs
        self.alpha_update_strategy = alpha_update_strategy or {
            "balance_ratio": 2 / 3,  # alpha2:alpha3 ratio, e.g., 6:4 for animal vs vehicle
        }

        self.params = args
        self.init_optimizer(self.params.num_adv_epochs)
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
        
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100, 105])    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None

    def update_alphas(self, current_epoch: int):
        """
        Dynamically update alpha1, alpha2, and alpha3 based on the current epoch.
        """
        progress = current_epoch / self.max_epochs  # Calculate training progress (0 to 1)
        self.alpha1 = max(0.0, 0.9 * (1 - progress))  # Decrease alpha1 from 0.9 to 0
        alpha23_total = 0.1 + 0.8 * progress  # Increase alpha2 + alpha3 from 0.1 to 0.9

        # Split alpha23_total between alpha2 and alpha3 based on the balance ratio
        balance_ratio = self.alpha_update_strategy["balance_ratio"]
        self.alpha2 = alpha23_total * balance_ratio / (1 + balance_ratio)
        self.alpha3 = alpha23_total / (1 + balance_ratio)

    def forward(self, x):
        root_logits, subroot_logits = self.model(x)
        return root_logits, subroot_logits

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Train each trainer on a given (sub)set of data.
        """
        # update alpha every epoch 
        self.update_alphas(epoch)
        
        metrics = pd.DataFrame()  # Initialize metrics
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

    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss, _ = self.root_trainer.standard_loss(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal, _ = self.animal_trainer.standard_loss(subroot_logits[is_animal], y[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle, _ = self.vehicle_trainer.standard_loss(subroot_logits[is_vehicle], y[is_vehicle])

        loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)

        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y

        root_logits, subroot_logits = self(x_adv)
        preds = torch.argmax(subroot_logits, 1)

        root_loss, _ = self.root_trainer.adversarial_loss(root_logits, y_adv)

        subroot_loss_animal = torch.tensor(0.0, device=y_adv.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y_adv.device)

        animal_classes_index = torch.tensor(animal_classes, device=y_adv.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y_adv.device)

        is_animal = torch_isin(y_adv, animal_classes_index)
        is_vehicle = torch_isin(y_adv, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal, _ = self.animal_trainer.adversarial_loss(subroot_logits[is_animal], y_adv[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle, _ = self.vehicle_trainer.adversarial_loss(subroot_logits[is_vehicle], y_adv[is_vehicle])

        loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        batch_metrics = {'loss': loss.item()}
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        self.optimizer.zero_grad()
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss, _ = self.root_trainer.trades_loss(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal, _ = self.animal_trainer.trades_loss(subroot_logits[is_animal], y[is_animal], beta)
        if is_vehicle.any():
            subroot_loss_vehicle, _ = self.vehicle_trainer.trades_loss(subroot_logits[is_vehicle], y[is_vehicle], beta)

        loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def mart_loss(self, x, y, beta):
        """
        MART training.
        """
        self.optimizer.zero_grad()
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss, _ = self.root_trainer.mart_loss(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal, _ = self.animal_trainer.mart_loss(subroot_logits[is_animal], y[is_animal], beta)
        if is_vehicle.any():
            subroot_loss_vehicle, _ = self.vehicle_trainer.mart_loss(subroot_logits[is_vehicle], y[is_vehicle], beta)

        loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                _, out = self.model(x_adv)
            else:
                _, out = self.model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc
    
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
