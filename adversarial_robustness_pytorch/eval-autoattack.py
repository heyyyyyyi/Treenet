"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed

from core import animal_classes, vehicle_classes

import wandb

# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = os.path.join(args.log_dir, args.desc)
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

DATA_DIR = os.path.join(args.data_dir, args.data)
WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + '/log-aa.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))

# wandb.init(
#     project="ablation_test", entity="yhe106-johns-hopkins-university",
#     name=f"{args.desc}-autoattack",  # Use a descriptive name for this evaluation
# )


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model
print('Creating model: {}'.format(args.model))
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint

# Wrap model for TreeEnsemble forward logic
class TreeModelWrapper(nn.Module):
    def __init__(self, model, softroute=False):
        super(TreeModelWrapper, self).__init__()
        self.model = model
        self.softroute = softroute

    def forward(self, x):
        if not self.softroute:
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
        
        else:
            # weighted logits for soft routing
            root_logits, subroot_animal_logits, subroot_vehicle_logits = self.model(x)

            root_probs = torch.softmax(root_logits, dim=1)

            # animal mask: shape [B, 1]，每个样本是否属于 animal coarse 的置信度
            animal_score = root_probs[:, animal_classes].sum(dim=1, keepdim=True)
            vehicle_score = root_probs[:, vehicle_classes].sum(dim=1, keepdim=True)

            # Normalize
            total = animal_score + vehicle_score + 1e-8
            w_animal = animal_score / total
            w_vehicle = vehicle_score / total
            w_root = 1.0 - w_animal - w_vehicle

            # shape broadcast
            final_logits = w_root * root_logits + w_animal * subroot_animal_logits + w_vehicle * subroot_vehicle_logits
            return final_logits

if "tree" in args.model:
    model = TreeModelWrapper(model, softroute=False)

# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print('Script Completed.')