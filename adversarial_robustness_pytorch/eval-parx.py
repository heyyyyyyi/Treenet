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
from par_x.model import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
import torch.nn.functional as F

from core import animal_classes, vehicle_classes

import wandb
from par_x.train import ParEnsemble

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
class ModelWrapper(nn.Module):
    def __init__(self, model, softroute=False,):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        subroot_animal_logits, subroot_vehicle_logits = self.model(x)
        # Ensure indices are on the same device as the logits
        device = x.device
        animal_classes_index = torch.tensor(animal_classes, device=device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=device)
        
        conf_animal = 1 - F.softmax(subroot_animal_logits, dim=1)[:, -1]  # scalar confidence
        conf_vehicle = 1 - F.softmax(subroot_vehicle_logits, dim=1)[:, -1]

        animal_logits = torch.full((subroot_animal_logits.shape[0], 10), fill_value=-5.0, device=device)
        vehicle_logits = torch.full((subroot_vehicle_logits.shape[0], 10), fill_value=-5.0, device=device)

        animal_logits[:, animal_classes_index] = subroot_animal_logits[:, :-1]
        vehicle_logits[:, vehicle_classes_index] = subroot_vehicle_logits[:, :-1]

        # Normalize confidence
        total_conf = conf_animal + conf_vehicle + 1e-8
        logits_final = (conf_animal.unsqueeze(1) * animal_logits + 
                        conf_vehicle.unsqueeze(1) * vehicle_logits) / total_conf.unsqueeze(1)

        return logits_final 
    
if "tree" in args.model:
    model = ModelWrapper(model)


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