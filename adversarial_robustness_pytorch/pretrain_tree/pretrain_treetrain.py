from .model import animal_classes, vehicle_classes
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import create_attack
from core.metrics import accuracy, binary_accuracy, subclass_accuracy
from core.models import create_model

from core.models import Normalization
from .mart import mart_loss, mart_tree_loss
from .rst import CosineLR
from .trades import trades_loss, trades_tree_loss

from .models import lighttreeresnet

from .context import ctx_noparamgrad_and_eval
from .core.utils.treetrain import focal_loss_with_pt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TreeTrain:
    """
    Helper class for training a tree-based model.
    Arguments:
        args (dict): input arguments.
        info (dict): dataset information.
    """
    def __init__(self, args, info):
        super(TreeTrain, self).__init__()
        seed(args.seed)
        self.model = create_model(args.model, args.normalize, info, device)
            
        self.params = args
        self.criterion = nn.CrossEntropyLoss()
        self.init_optimizer(self.params.num_adv_epochs)
        
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
        
        self.model.subroot_coarse, self.model.subroot_animal, self.model.subroot_vehicle = self.wrapper_model()
        # dictionary 
        self.attack, self.eval_attack = self.init_attack(self.model, self.criterion, self.params.attack, self.params.attack_eps, 
                                                            self.params.attack_iter, self.params.attack_step)
    def wrapper_model(self):
        class WrapperModel(nn.Module):
            def __init__(self, share, model):
                super(WrapperModel, self).__init__()
                self.share = share
                self.model = model
            def forward(self, x):
                return self.model(self.share(x))
        return  [WrapperModel(self.model.root_model, self.model.subroot_coarse),
                WrapperModel(self.model.root_model, self.model.subroot_animal),
                WrapperModel(self.model.root_model, self.model.subroot_vehicle)]

    @staticmethod
    def init_attack_helper(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        else:
            raise ValueError('Unknown attack type: {}'.format(attack_type))
        
        return attack, eval_attack
    
    def init_attack(self, model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        coarse_attack, coarse_eval_attack = self.init_attack_helper(model.subroot_coarse, criterion, attack_type, attack_eps, attack_iter, attack_step)
        animal_attack, animal_eval_attack = self.init_attack_helper(model.subroot_animal, criterion, attack_type, attack_eps, attack_iter, attack_step)
        vehicle_attack, vehicle_eval_attack = self.init_attack_helper(model.subroot_vehicle, criterion, attack_type, attack_eps, attack_iter, attack_step)
        return {
            'coarse': coarse_attack,
            'animal': animal_attack,
            'vehicle': vehicle_attack
        }, {
            'coarse': coarse_eval_attack,
            'animal': animal_eval_attack,
            'vehicle': vehicle_eval_attack
        }
    
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
        pass 
    
    def load_model(self, path):