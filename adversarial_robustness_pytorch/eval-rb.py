"""
Evaluation with Robustbench.
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

from robustbench import benchmark

from core.data import get_data_info
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

args.data = 'cifar10' if args.data in ['cifar10s', 'cifar10g'] else args.data

DATA_DIR = os.path.join(args.data_dir, args.data)
WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + f'/log-corr-{args.threat}.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert args.data in ['cifar10'], 'Evaluation on Robustbench is only supported for cifar10!'

threat_model = args.threat
dataset = args.data
model_name = args.desc

wandb.init(
    project="ablation_test", entity="yhe106-johns-hopkins-university",
    name=f"{args.desc}-robustbench",  # Use a descriptive name for this evaluation
)

# Model
print('Creating model: {}'.format(args.model))
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint

# Wrap model for TreeEnsemble forward logic
class TreeModelWrapper(nn.Module):
    def __init__(self, model):
        super(TreeModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        root_logits, subroot_animal_logits, subroot_vehicle_logits = self.model(x)
        root_pred = torch.argmax(root_logits, dim=1)

        subroot_logits = torch.zeros_like(root_logits)
        animal_classes_index = torch.tensor(animal_classes, device=root_pred.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=root_pred.device)

        is_animal = torch.isin(root_pred, animal_classes_index)
        is_vehicle = torch.isin(root_pred, vehicle_classes_index)

        if is_animal.any():
            subroot_logits[is_animal] = subroot_animal_logits[is_animal]
        if is_vehicle.any():
            subroot_logits[is_vehicle] = subroot_vehicle_logits[is_vehicle]

        return subroot_logits

if "tree" in args.model:
    model = TreeModelWrapper(model)

# Common corruptions

seed(args.seed)
clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=args.num_samples, dataset=dataset,
                                  threat_model=threat_model, eps=args.attack_eps, device=device, to_disk=False, 
                                  data_dir=DATA_DIR+'c')


logger.log('Model: {}'.format(args.desc))
logger.log('Evaluating robustness on {} with threat model={}.'.format(args.data, args.threat))
logger.log('Clean Accuracy: \t{:.2f}%.\nRobust Accuracy: \t{:.2f}%.'.format(clean_acc*100, robust_acc*100))
