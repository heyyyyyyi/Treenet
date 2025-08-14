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
    def __init__(self, info, args, model, other_label, weighted_loss):
        super(Trainer, self).__init__()
        seed(args.seed)
        self.model = model
        self.params = args
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

        # Determine the correct subroot attribute
        if hasattr(self.model, 'subroot_animal'):
            subroot = self.model.subroot_animal
        elif hasattr(self.model, 'subroot_vehicle'):
            subroot = self.model.subroot_vehicle
        else:
            raise AttributeError("The model does not have a valid subroot attribute to pretrain.")

        subroot.train()  # Train the specific subroot model

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
                nn.utils.clip_grad_norm_(subroot.parameters(), self.params.clip_grad)
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
        if hasattr(self.model, 'subroot_coarse'):  # For CoarseWrapper
            torch.save(self.model.subroot_coarse.state_dict(), path)
        elif hasattr(self.model, 'subroot_animal'):  # For AnimalWrapper
            torch.save(self.model.subroot_animal.state_dict(), path)
        elif hasattr(self.model, 'subroot_vehicle'):  # For VehicleWrapper
            torch.save(self.model.subroot_vehicle.state_dict(), path)
        else:
            raise AttributeError("The model does not have a valid subroot attribute to save.")

    def load_model(self, share_path, subroot_path):
        """
        Load the model from a file.
        """
        if share_path is not None:
            self.model.share.load_state_dict(torch.load(share_path, map_location=device))
        
        if subroot_path is not None:
            if hasattr(self.model, 'subroot_coarse'):  # For CoarseWrapper
                self.model.subroot_coarse.load_state_dict(torch.load(subroot_path, map_location=device))
            elif hasattr(self.model, 'subroot_animal'):  # For AnimalWrapper
                self.model.subroot_animal.load_state_dict(torch.load(subroot_path, map_location=device))
            elif hasattr(self.model, 'subroot_vehicle'):  # For VehicleWrapper
                self.model.subroot_vehicle.load_state_dict(torch.load(subroot_path, map_location=device))
            else:
                raise AttributeError("The model does not have a valid subroot attribute to load.")
        
        self.model.share.eval()
        if hasattr(self.model, 'subroot_coarse'):
            self.model.subroot_coarse.eval()
        elif hasattr(self.model, 'subroot_animal'):
            self.model.subroot_animal.eval()
        elif hasattr(self.model, 'subroot_vehicle'):
            self.model.subroot_vehicle.eval()
    
    def load_share(self, path):
        """
        Load the share model from a file.
        """
        self.model.share.load_state_dict(torch.load(path, map_location=device))
    
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

def pretrain_coarse(model, train_dataloader, test_dataloader, temp_args, logger, device, info):
    """Pretrain the shared feature extractor and coarse subroot classifier."""
    args = argparse.Namespace(**vars(temp_args))
    args.num_classes = 10

    trainer = Trainer(info, args, model, other_label=None, weighted_loss=None)
    logger.log("Pretraining share + coarse...")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.desc}-share-coarse",
        reinit=True,
    )

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, f"{args.desc}")
    WEIGHTS_SHARE = os.path.join(LOG_DIR, 'weights-share.pt')
    WEIGHTS_COARSE = os.path.join(LOG_DIR, 'weights-coarse.pt')
    os.makedirs(LOG_DIR, exist_ok=True)

    metrics = pd.DataFrame()
    old_score = [0.0, 0.0]

    for epoch in range(1, args.num_adv_epochs + 1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))

        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]

        train_metrics = trainer.pretrain_share_coarse(train_dataloader, epoch=epoch, adversarial=True)
        test_metrics = trainer.eval(test_dataloader)

        epoch_metrics = {'train_'+k: v for k, v in train_metrics.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr})
        epoch_metrics.update({
            'test_clean_acc': test_metrics['acc'],
            'test_clean_acc_animal': test_metrics['acc_animal'],
            'test_clean_acc_vehicle': test_metrics['acc_vehicle'],
            'test_clean_acc_bi': test_metrics['acc_bi'],
            'test_adversarial_acc': None,
            'test_adversarial_acc_animal': None,
            'test_adversarial_acc_vehicle': None,
            'test_adversarial_acc_bi': None,
        })

        if epoch % args.adv_eval_freq == 0 or epoch > (args.num_adv_epochs - 5):
            test_adv_metrics = trainer.eval(test_dataloader, adversarial=True)
            epoch_metrics.update({
                'test_adversarial_acc': test_adv_metrics['acc'],
                'test_adversarial_acc_animal': test_adv_metrics['acc_animal'],
                'test_adversarial_acc_vehicle': test_adv_metrics['acc_vehicle'],
                'test_adversarial_acc_bi': test_adv_metrics['acc_bi'],
            })

        if test_metrics['acc'] >= old_score[0]:
            old_score[0] = test_metrics['acc']
            trainer.save_share(WEIGHTS_SHARE)
            trainer.save_subroot(WEIGHTS_COARSE)

        logger.log('Time taken: {}'.format(format_time(time.time() - start)))
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats_coarse.csv'), index=False)
        wandb.log(epoch_metrics)

    logger.log('\nPretraining completed.')
    logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(old_score[0] * 100))
    wandb.finish()

def pretrain_submodels(animal_model, vehicle_model, animal_train_dataloader, animal_test_dataloader, vehicle_train_dataloader, vehicle_test_dataloader, temp_args, logger, device, info):
    """Pretrain the animal and vehicle subroot classifiers."""
    args = argparse.Namespace(**vars(temp_args))
    # Pretrain animal subroot
    args.num_classes = 6
    trainer = Trainer(info, args, animal_model, other_label=None, weighted_loss=None)
    logger.log("Pretraining animal subroot...")

    LOG_DIR = os.path.join(args.log_dir, f"{args.desc}")
    trainer.load_model(os.path.join(LOG_DIR, 'weights-share.pt'), None)  # Fix: Load shared weights from LOG_DIR
    WEIGHTS_ANIMAL = os.path.join(LOG_DIR, 'weights-animal.pt')  # Ensure save path matches

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.desc}-animal-subroot",
        reinit=True,
    )

    metrics = pd.DataFrame()
    old_score = 0.0

    for epoch in range(1, args.num_adv_epochs + 1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))

        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]

        train_metrics = trainer.pretrain_subroot(animal_train_dataloader, epoch=epoch, adversarial=True)
        test_metrics = trainer.eval(animal_test_dataloader)

        epoch_metrics = {'train_'+k: v for k, v in train_metrics.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr})
        epoch_metrics.update({
            'test_clean_acc_animal': test_metrics['acc'],
            'test_adversarial_acc_animal': None,
        })

        if epoch % args.adv_eval_freq == 0 or epoch > (args.num_adv_epochs - 5):
            test_adv_metrics = trainer.eval(animal_test_dataloader, adversarial=True)
            epoch_metrics['test_adversarial_acc_animal'] = test_adv_metrics['acc']

        if test_metrics['acc'] >= old_score:
            old_score = test_metrics['acc']
            trainer.save_subroot(WEIGHTS_ANIMAL)

        logger.log('Time taken: {}'.format(format_time(time.time() - start)))
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats_animal.csv'), index=False)
        wandb.log(epoch_metrics)

    logger.log('\nPretraining animal subroot completed.')
    logger.log('Standard Accuracy-\tTest (Animal): {:.2f}%.'.format(old_score * 100))
    wandb.finish()

    # Pretrain vehicle subroot
    args.num_classes = 4
    trainer = Trainer(info, args, vehicle_model, other_label=None, weighted_loss=None)
    logger.log("Pretraining vehicle subroot...")
    trainer.load_model(os.path.join(LOG_DIR, 'weights-share.pt'), None)  # Fix: Load shared weights from LOG_DIR

    WEIGHTS_VEHICLE = os.path.join(LOG_DIR, 'weights-vehicle.pt')  # Ensure save path matches

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.desc}-vehicle-subroot",
        reinit=True,
    )

    metrics = pd.DataFrame()
    old_score = 0.0

    for epoch in range(1, args.num_adv_epochs + 1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))

        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]

        train_metrics = trainer.pretrain_subroot(vehicle_train_dataloader, epoch=epoch, adversarial=True)
        test_metrics = trainer.eval(vehicle_test_dataloader)

        epoch_metrics = {'train_'+k: v for k, v in train_metrics.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr})
        epoch_metrics.update({
            'test_clean_acc_vehicle': test_metrics['acc'],
            'test_adversarial_acc_vehicle': None,
        })

        if epoch % args.adv_eval_freq == 0 or epoch > (args.num_adv_epochs - 5):
            test_adv_metrics = trainer.eval(vehicle_test_dataloader, adversarial=True)
            epoch_metrics['test_adversarial_acc_vehicle'] = test_adv_metrics['acc']

        if test_metrics['acc'] >= old_score:
            old_score = test_metrics['acc']
            trainer.save_subroot(WEIGHTS_VEHICLE)

        logger.log('Time taken: {}'.format(format_time(time.time() - start)))
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats_vehicle.csv'), index=False)
        wandb.log(epoch_metrics)

    logger.log('\nPretraining vehicle subroot completed.')
    logger.log('Standard Accuracy-\tTest (Vehicle): {:.2f}%.'.format(old_score * 100))
    wandb.finish()
