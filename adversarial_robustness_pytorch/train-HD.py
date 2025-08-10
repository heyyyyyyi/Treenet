"""
Adversarial Training. (Tree)
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

from core.data import get_data_info
from core.data import load_data

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train

from HD_tree.train import HDEnsemble
from HD_tree.individual_train import pretrain_coarse, pretrain_submodels, setup_data

from core.utils import seed
import wandb
import subprocess  # 用于调用评估脚本

from core import animal_classes, vehicle_classes

_WANDB_USERNAME = "yhe106-johns-hopkins-university"
_WANDB_PROJECT = "ablation_test"

# Setup

parse = parser_train()

# Add missing arguments
# parse.add_argument('--decay_factor', type=float, default=0.98, help='Decay factor for alpha values.')
# parse.add_argument('--strategy', type=str, default='constant', choices=['exponential', 'linear', 'constant'], help='Strategy for alpha decay.')
# parse.add_argument('--softroute', type=bool, default=True, help='Use soft routing for the tree ensemble.')
# parse.add_argument('--unknown_classes', type=bool, default=True, help='Use unknown classes for the tree ensemble.')
# parse.add_argument('--pretrained', type=bool, default=True, help='Load pre-trained sub-models.')
parse.add_argument('--train_submodels', type=bool, default=False, help='Train sub-models independently before ensemble training.')
parse.add_argument('--train_coarse', type=bool, default=False, help='Train shared and coarse models before ensemble training.')

args = parse.parse_args()

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

torch.backends.cudnn.benchmark = True

# Load data
seed(args.seed)
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, 
)
del train_dataset, test_dataset
print (f"Train Dataloader Size: {len(train_dataloader.dataset)}")
print (f"Test Dataloader Size: {len(test_dataloader.dataset)}")

# Pretrain shared and coarse models if required
if args.train_coarse:
    logger.info("Pretraining share + coarse...")
    pretrain_coarse(train_dataloader, test_dataloader, args, logger, device, info)

# Pretrain sub-models if required
if args.train_submodels:
    logger.info("Pretraining subroot models...")
    animal_train_dataloader, animal_test_dataloader = setup_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, args, filter_classes=vehicle_classes, pseudo_label_classes=animal_classes
    )
    vehicle_train_dataloader, vehicle_test_dataloader = setup_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, args, filter_classes=animal_classes, pseudo_label_classes=vehicle_classes
    )
    # 输出数据大小 检查
    print(f"Animal Train Dataloader Size: {len(animal_train_dataloader.dataset)}")
    print(f"Animal Test Dataloader Size: {len(animal_test_dataloader.dataset)}")
    print(f"Vehicle Train Dataloader Size: {len(vehicle_train_dataloader.dataset)}")
    print(f"Vehicle Test Dataloader Size: {len(vehicle_test_dataloader.dataset)}")

    pretrain_submodels(
        animal_train_dataloader, animal_test_dataloader, vehicle_train_dataloader, vehicle_test_dataloader,
        args, logger, device, info
    )

# Initialize wandb
wandb.init(
    project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
    name=args.desc,
    reinit=True,
)

# Initialize HDEnsemble
trainer = HDEnsemble(info, args)

# Load pre-trained components if specified
if args.pretrained:
    logger.info("Loading pre-trained components...")
    trainer.load_share(os.path.join(args.log_dir, 'share_layer', 'weights-share.pt'))
    trainer.load_subroot_coarse(os.path.join(args.log_dir, 'coarse_layer', 'weights-coarse.pt'))
    trainer.load_subroot_animal(os.path.join(args.log_dir, 'animal_layer', 'weights-animal.pt'))
    trainer.load_subroot_vehicle(os.path.join(args.log_dir, 'vehicle_layer', 'weights-vehicle.pt'))

logger.info("Model Summary:")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info("Total Trainable Parameters: {}".format(count_parameters(trainer.model)))

# Save initial model weights
if not os.path.exists(WEIGHTS):
    logger.info('Saving initial model weights to {}'.format(WEIGHTS))
    trainer.save_model(WEIGHTS)

# Adversarial Training
if NUM_ADV_EPOCHS > 0:
    logger.info('\n\n')
    metrics = pd.DataFrame()
    logger.info('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)['acc']*100))
    logger.info('Adversarial Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader, adversarial=True)['acc']*100))
    
    old_score = [0.0, 0.0]
    logger.info('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    

for epoch in range(1, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.info('======= Epoch {} ======='.format(epoch))
    
    # Extract learning rates
    root_lr = trainer.optimizer.param_groups[0]['lr']
    subroot_animal_lr = trainer.optimizer.param_groups[1]['lr']
    subroot_vehicle_lr = trainer.optimizer.param_groups[2]['lr']

    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.fine_tune(train_dataloader, epoch=epoch, adversarial=True)
    test_res = trainer.eval(test_dataloader)
    test_acc = test_res['acc']
    acc_animal = test_res['acc_animal']
    acc_vehicle = test_res['acc_vehicle']

    alpha1, alpha2, alpha3 = trainer.update_alphas(epoch, args.decay_factor)

    logger.info('Loss: {:.4f}.\tRoot LR: {:.6f}.\tSubroot Animal LR: {:.6f}.\tSubroot Vehicle LR: {:.6f}'.format(
        res['loss'], root_lr, subroot_animal_lr, subroot_vehicle_lr
    ))
    
    logger.info('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({
        'epoch': epoch,
        'lr_root': root_lr,
        'lr_subroot_animal': subroot_animal_lr,
        'lr_subroot_vehicle': subroot_vehicle_lr,
        'test_clean_acc': test_acc,
        'test_clean_acc_animal': acc_animal,
        'test_clean_acc_vehicle': acc_vehicle,
        'test_adversarial_acc': None,
        'test_adversarial_acc_animal': None,
        'test_adversarial_acc_vehicle': None,
    })
    
    if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5):
        test_adv_res = trainer.eval(test_dataloader, adversarial=True)
        test_adv_acc = test_adv_res['acc']
        test_adv_acc_animal = test_adv_res['acc_animal']
        test_adv_acc_vehicle = test_adv_res['acc_vehicle']

        logger.info('Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        epoch_metrics.update({'test_adversarial_acc_animal': test_adv_acc_animal})
        epoch_metrics.update({'test_adversarial_acc_vehicle': test_adv_acc_vehicle})

    logger.info('Alpha1: {:.4f}.\tAlpha2: {:.4f}.\tAlpha3: {:.4f}'.format(alpha1, alpha2, alpha3))
    epoch_metrics.update({'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3})

    if test_adv_acc >= old_score[1]:
        old_score[0], old_score[1] = test_acc, test_adv_acc
        trainer.save_model(WEIGHTS)
    trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

    logger.info('Time taken: {}'.format(format_time(time.time()-start)))
    
    metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)
    wandb.log(epoch_metrics)

# Record metrics
train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)['acc']
logger.info('\nTraining completed.')
logger.info('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0:
    logger.info('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

logger.info('Script Completed.')

wandb.summary["final_train_acc"] = train_acc
wandb.summary["final_test_clean_acc"] = old_score[0]
wandb.summary["final_test_adv_acc"] = old_score[1]

# Ensure AutoAttack evaluation subprocess runs successfully
try:
    logger.info('Starting AutoAttack evaluation...')
    aa_result = subprocess.run(
        ['python', 'eval-aa.py', '--desc', args.desc, '--log-dir', args.log_dir, '--data-dir', args.data_dir, '--softroute', str(args.softroute).lower(), '--unknown_classes', str(args.unknown_classes).lower()],
        capture_output=True, text=True
    )
    logger.info(aa_result.stdout)

    # Parse and log AutoAttack results to wandb
    for line in aa_result.stdout.splitlines():
        if "autoattack_clean_accuracy" in line:
            clean_acc = float(line.split(":")[1].strip().replace("%", "")) / 100
            wandb.summary["autoattack_clean_acc"] = clean_acc
        if "autoattack_robust_accuracy" in line:
            robust_acc = float(line.split(":")[1].strip().replace("%", "")) / 100
            wandb.summary["autoattack_robust_acc"] = robust_acc
except Exception as e:
    logger.info(f"AutoAttack evaluation failed: {e}")

wandb.finish()

