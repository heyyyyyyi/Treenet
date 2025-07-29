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

# Add one more wrapper for tree model 
class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args, other_label, weighted_loss):
        super(Trainer, self).__init__()
        seed(args.seed)
        self.model = create_model(args.model, args.normalize, info, device, num_classes=args.num_classes)
            
        self.params = args
        # Pass weighted_loss to CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weighted_loss).to(device) if weighted_loss else None)

        #self.criterion = partial(focal_loss_with_pt, weights=torch.tensor(weighted_loss).to(device) if weighted_loss else None)


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
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training with improved logging.
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

# run_individual_train.py
def run_training(info, temp_args, logger, train_dataloader, test_dataloader, desc, num_classes, eval_metrics=None, other_label=None, weighted_loss = None):
    # setup new runs in wandb
    # copy args 
    args = argparse.Namespace(**vars(temp_args))
    args.model = 'lightresnet20'
    args.num_classes = num_classes
    args.desc = desc

    # print logger runs name 
    logger.log('Running training with description: {}'.format(desc))
    
    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    info = get_data_info(DATA_DIR)
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_VALIDATION = args.batch_size_validation
    NUM_ADV_EPOCHS = args.num_adv_epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log('Using device: {}'.format(device))
    if args.debug:
        NUM_ADV_EPOCHS = 1

    seed(args.seed)
    if args.tau:
        logger.log('Using WA.')
        trainer = WATrainer(info, args)
    else:
        trainer = Trainer(info, args, other_label=other_label, weighted_loss=weighted_loss)
    
    logger.log("Model Summary:")
    try:
        from torchsummary import summary
        summary(trainer.model, input_size=(3, 32, 32), device=str(device))
    except ImportError:
        logger.log("torchsummary not installed. Skipping detailed summary.")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log("Total Trainable Parameters: {}".format(count_parameters(trainer.model)))

    last_lr = args.lr
    metrics = pd.DataFrame()
    old_score = [0.0, 0.0]
    test_adv_acc = 0.0

    if NUM_ADV_EPOCHS > 0:
        logger.log('\n\n')
        logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)['acc']*100))
        logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
        trainer.init_optimizer(args.num_adv_epochs)

    for epoch in range(1, NUM_ADV_EPOCHS+1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))
        
        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]
        
        res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
        test_res = trainer.eval(test_dataloader)
        epoch_metrics = {'train_'+k: v for k, v in res.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr})

        # Update metrics based on configuration
        if "bi" in eval_metrics:
            epoch_metrics.update({
                'test_clean_acc': test_res['acc'],
                'test_clean_acc_animal': test_res['acc_animal'],
                'test_clean_acc_vehicle': test_res['acc_vehicle'],
                'test_clean_acc_bi': test_res['acc_bi'],
                'test_adversarial_acc': None,
                'test_adversarial_acc_animal': None,
                'test_adversarial_acc_vehicle': None,
                'test_adversarial_acc_bi': None,
            })
        elif "animal" in eval_metrics:
            epoch_metrics.update({
                'test_clean_acc_animal': test_res['acc'],
                'test_adversarial_acc_animal': None,
            })
        else:
            epoch_metrics.update({
                'test_clean_acc_vehicle': test_res['acc'],
                'test_adversarial_acc_vehicle': None,
            })

        # Evaluate adversarial metrics
        if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5) or (epoch >= (NUM_ADV_EPOCHS-10) and NUM_ADV_EPOCHS > 90):
            test_adv_res = trainer.eval(test_dataloader, adversarial=True)
            test_adv_acc = test_adv_res['acc']
            if "bi" in eval_metrics:
                epoch_metrics.update({
                    'test_adversarial_acc': test_adv_res['acc'],
                    'test_adversarial_acc_animal': test_adv_res['acc_animal'],
                    'test_adversarial_acc_vehicle': test_adv_res['acc_vehicle'],
                    'test_adversarial_acc_bi': test_adv_res['acc_bi'],
                })
            elif "animal" in eval_metrics:
                epoch_metrics['test_adversarial_acc_animal'] = test_adv_res['acc']
            else:
                epoch_metrics['test_adversarial_acc_vehicle'] = test_adv_res['acc']

        else:
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))

        if test_adv_acc >= old_score[1]:
            old_score[0], old_score[1] = test_res['acc'], test_adv_acc
            trainer.save_model(WEIGHTS)
        trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

        logger.log('Time taken: {}'.format(format_time(time.time()-start)))
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)
        wandb.log(epoch_metrics)
    
    train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)['acc']
    logger.log('\nTraining completed.')
    logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
    if NUM_ADV_EPOCHS > 0:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

    logger.log('Script Completed.')






