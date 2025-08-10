import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from core.attacks import create_attack
from core.metrics import accuracy, binary_accuracy, subclass_accuracy, accuracy_exclude_other
from .model import create_model

from core.utils.context import ctx_noparamgrad_and_eval
from core.utils.utils import seed

from core.utils.mart import mart_loss
from core.utils.rst import CosineLR
from core.utils.trades import trades_loss

from core.utils.logger import Logger
from core.data import load_data, get_data_info
import wandb
from gowal21uncovering.utils import WATrainer
import shutil
import time
from core.utils import format_time
from .train import focal_loss_with_pt

from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']
 
# pretrain share + coarse 
class Trainer(object):
    def __init__(self, info, args, other_label, weighted_loss):
        super(Trainer, self).__init__()
        seed(args.seed)
        self.model = create_model(
            name=args.model,
            normalize=args.normalize,
            info=info,
            device=device
        )
        self.model = self.model.to(device)
        self.info = info
        if weighted_loss:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weighted_loss).to(device) if weighted_loss else None)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)

        self.init_optimizer(self.params.num_adv_epochs)
        
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
        
        self.attack, self.eval_attack = self.init_attack(self.model, self.criterion, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)

        self.other_label = other_label

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
    
    def pretrain_share_coarse(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Pretrain the share and coarse models.
        """
        metrics = pd.DataFrame()
        self.model.train()

        for data in tqdm(dataloader, desc=f'Epoch {epoch}: ', disable=not verbose):
            
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

    def pretrain_subroot(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Pretrain the animal and vehicle subroot models.
        """
        metrics = pd.DataFrame()
        self.model.share.eval()  # Ensure the share model is in eval mode
        self.model.subroot.train()  # Train the coarse subroot model
        for data in tqdm(dataloader, desc=f'Epoch {epoch}: ', disable=not verbose):
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
                nn.utils.clip_grad_norm_(self.model.subroot.parameters(), self.params.clip_grad)
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
        out = self.model(x)
        # Ensure the criterion uses the weighted loss
        loss = self.criterion(out, y)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy_exclude_other(y, preds, self.other_label)}
        return loss, batch_metrics
    
    
    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        #print(y.unique())
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
            
        out = self.model(x_adv)
        
        # Ensure the criterion uses the weighted loss
        loss = self.criterion(out, y_adv)
        #print(f"Loss: {loss.item()}")
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item()}
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy_exclude_other(y, preds_clean, self.other_label), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy_exclude_other(y, preds, self.other_label)})    
        return loss, batch_metrics
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training.
        """
        loss, batch_metrics = mart_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                        epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                        beta=beta, attack=self.params.attack)
        return loss, batch_metrics  
    
    def eval_test(self, x, adversarial=False):
        """
        Evaluate performance of the model.
        """
        x = x.to(device)
        if adversarial:
            with ctx_noparamgrad_and_eval(self.model):
                x_adv, _ = self.eval_attack.perturb(x, y)            
            out = self.model(x_adv)
        else:
            out = self.model(x)
        return out 
    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        acc_animal, acc_vehicle, acc_bi = 0.0, 0.0, 0.0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            #print(f"Evaluating batch: {x.shape}, labels: {y.shape}")
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.model(x_adv)
            else:
                out = self.model(x)
            acc += accuracy_exclude_other(y, out, self.other_label)
            temp_acc_animal, temp_acc_vehicle = subclass_accuracy(y, out)
            acc_animal += temp_acc_animal
            acc_vehicle += temp_acc_vehicle
            acc_bi += binary_accuracy(y, out)

        acc /= len(dataloader)
        acc_animal /= len(dataloader)
        acc_vehicle /= len(dataloader)
        acc_bi /= len(dataloader)

        return dict(
            acc=acc,
            acc_animal=acc_animal,
            acc_vehicle=acc_vehicle,
            acc_bi=acc_bi
        )
    def save_share(self, path):
        """
        Save the share model to a file.
        """
        torch.save(self.model.share.state_dict(), path)

    def save_subroot(self, path):
        """
        Save the subroot model to a file.
        """
        torch.save(self.model.subroot.state_dict(), path)

    def load_model(self, share_path, subroot_path):
        """
        Load the model from a file.
        """
        if share_path is not None:
            self.model.share.load_state_dict(torch.load(share_path, map_location=device))
        if subroot_path is not None:
            self.model.subroot.load_state_dict(torch.load(subroot_path, map_location=device))
        self.model.share.eval()
        self.model.subroot.eval()
    
def setup_data(data_dir, batch_size, batch_size_validation, args, filter_classes=None, pseudo_label_classes=None):
    """
    Setup data for training and testing, with optional filtering and pseudo-labeling.
    """
    seed(args.seed)
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
        data_dir, batch_size, batch_size_validation, use_augmentation=args.augment, shuffle_train=True, 
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, 
        filter_classes=filter_classes, pseudo_label_classes=pseudo_label_classes
    )
    del train_dataset, test_dataset
    return train_dataloader, test_dataloader

def pretrain_coarse(train_dataloader, test_dataloader, args, logger, device, info):
    """Pretrain the shared feature extractor and coarse subroot classifier."""
    trainer = Trainer(info, args, other_label=None, weighted_loss=None)
    logger.info("Pretraining share + coarse...")

    # Initialize wandb for logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.desc}-share-coarse",
        reinit=True,
    )

    for epoch in range(args.num_adv_epochs):
        train_metrics = trainer.pretrain_share_coarse(train_dataloader, epoch=epoch, adversarial=True)
        test_metrics = trainer.eval(test_dataloader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics['loss'],
            "train_clean_acc": train_metrics['clean_acc'],
            "test_clean_acc": test_metrics['acc'],
        })

        logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Clean Acc: {train_metrics['clean_acc']:.4f}, "
                    f"Test Clean Acc: {test_metrics['acc']:.4f}")

    trainer.save_share(os.path.join(args.log_dir, 'share_layer', 'weights-share.pt'))
    trainer.save_subroot_coarse(os.path.join(args.log_dir, 'coarse_layer', 'weights-coarse.pt'))

    wandb.finish()


def pretrain_submodels(animal_train_dataloader, animal_test_dataloader, vehicle_train_dataloader, vehicle_test_dataloader, args, logger, device, info):
    """Pretrain the animal and vehicle subroot classifiers."""
    trainer = Trainer(info, args, other_label=None, weighted_loss=None)

    # Pretrain animal subroot
    logger.info("Pretraining animal subroot...")
    trainer.load_model(os.path.join(args.log_dir, 'share_layer', 'weights-share.pt'), None)

    # Initialize wandb for animal subroot
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.desc}-animal-subroot",
        reinit=True,
    )

    for epoch in range(args.num_adv_epochs):
        train_metrics = trainer.pretrain_subroot(animal_train_dataloader, epoch=epoch, adversarial=True)
        test_metrics = trainer.eval(animal_test_dataloader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics['loss'],
            "train_clean_acc": train_metrics['clean_acc'],
            "test_clean_acc_animal": test_metrics['acc_animal'],
        })

        logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Clean Acc: {train_metrics['clean_acc']:.4f}, "
                    f"Test Clean Acc (Animal): {test_metrics['acc_animal']:.4f}")

    trainer.save_subroot(os.path.join(args.log_dir, 'animal_layer', 'weights-animal.pt'))
    wandb.finish()

    # Pretrain vehicle subroot
    logger.info("Pretraining vehicle subroot...")
    trainer.load_model(os.path.join(args.log_dir, 'share_layer', 'weights-share.pt'), None)

    # Initialize wandb for vehicle subroot
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.desc}-vehicle-subroot",
        reinit=True,
    )

    for epoch in range(args.num_adv_epochs):
        train_metrics = trainer.pretrain_subroot(vehicle_train_dataloader, epoch=epoch, adversarial=True)
        test_metrics = trainer.eval(vehicle_test_dataloader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss_vehicle": train_metrics['loss'],
            "train_clean_acc": train_metrics['clean_acc'],
            "test_clean_acc_vehicle": test_metrics['acc_vehicle'],
        })

        logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Clean Acc: {train_metrics['clean_acc']:.4f}, "
                    f"Test Clean Acc (Vehicle): {test_metrics['acc_vehicle']:.4f}")

    trainer.save_subroot(os.path.join(args.log_dir, 'vehicle_layer', 'weights-vehicle.pt'))
    wandb.finish()
