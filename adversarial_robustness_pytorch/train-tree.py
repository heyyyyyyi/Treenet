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

#from core.utils import Trainer
from core.utils import TreeEnsemble
from core.utils import seed
# use wandb for logging

import wandb

_WANDB_USERNAME = "yhe106-johns-hopkins-university"
_WANDB_PROJECT = "ablation_test"


# Setup

parse = parser_train()
# add args decay_foctor
parse.add_argument('--decay_factor', type=float, default=0.98, help='Decay factor for alpha values.')
args = parse.parse_args()

wandb.init(
    project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
    name=args.desc,  # 以你的描述作为 run 的名字
    #config=args,     # 自动记录所有超参数
)

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_ADV_EPOCHS = args.num_adv_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1

# To speed up training
torch.backends.cudnn.benchmark = True



# Load data

seed(args.seed)
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction
)
del train_dataset, test_dataset



# Adversarial Training (AT, TRADES and MART)

seed(args.seed)
trainer = TreeEnsemble(info, args)

logger.log("Model Summary:")
# try:
#     from torchsummary import summary
#     summary(trainer, input_size=(3, 32, 32), device=str(device))
# except ImportError:
#     logger.log("torchsummary not installed. Skipping detailed summary.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.log("Total Trainable Parameters: {}".format(count_parameters(trainer.model)))

last_lr = args.lr


if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)['acc']*100))
    
    old_score = [0.0, 0.0]
    logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    
    test_adversarial_acc_animal = 0.0
    test_adversarial_acc_vehicle = 0.0
    

for epoch in range(1, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    # 提取学习率
    root_lr = trainer.optimizer.param_groups[0]['lr']
    subroot_animal_lr = trainer.optimizer.param_groups[1]['lr']
    subroot_vehicle_lr = trainer.optimizer.param_groups[2]['lr']

    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
    test_res = trainer.eval(test_dataloader)
    test_acc = test_res['acc']
    root_acc = test_res['root_acc']
    root_acc_bi = test_res['root_acc_bi']
    acc_animal = test_res['acc_animal']
    acc_vehicle = test_res['acc_vehicle']
    # root_acc_animal = test_res['root_acc_animal']
    # root_acc_vehicle = test_res['root_acc_vehicle']

    alpha1, alpha2, alpha3 = trainer.update_alphas(epoch, args.decay_factor)

    logger.log('Loss: {:.4f}.\tRoot LR: {:.6f}.\tSubroot Animal LR: {:.6f}.\tSubroot Vehicle LR: {:.6f}'.format(
        res['loss'], root_lr, subroot_animal_lr, subroot_vehicle_lr
    ))
    
    if 'clean_acc' in res:
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    else:
        logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    
    
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({
        'epoch': epoch,
        'lr_root': root_lr,
        'lr_subroot_animal': subroot_animal_lr,
        'lr_subroot_vehicle': subroot_vehicle_lr,
        'test_clean_acc': test_acc,
        'test_clean_acc_bi': root_acc_bi,
        'test_clean_root_acc': root_acc,
        'test_clean_acc_animal': acc_animal,
        'test_clean_acc_vehicle': acc_vehicle,
        'test_adversarial_acc': None,
        'test_adversarial_acc_animal': None,
        'test_adversarial_acc_vehicle': None,
        'test_adversarial_root_acc': None,
        'test_adversarial_acc_bi': None,
    })
    
    if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5) or (epoch >= (NUM_ADV_EPOCHS-10) and NUM_ADV_EPOCHS > 90):
        test_adv_res = trainer.eval(test_dataloader, adversarial=True)
        test_adv_acc = test_adv_res['acc']
        test_adv_root_acc = test_adv_res['root_acc']
        test_adv_root_acc_bi = test_adv_res['root_acc_bi']
        test_adv_acc_animal = test_adv_res['acc_animal']
        test_adv_acc_vehicle = test_adv_res['acc_vehicle']
        # test_adv_root_acc_animal = test_adv_res['root_acc_animal']
        # test_adv_root_acc_vehicle = test_adv_res['root_acc_vehicle']

        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, 
                                                                                   test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        epoch_metrics.update({'test_adversarial_root_acc': test_adv_root_acc})
        epoch_metrics.update({'test_adversarial_acc_bi': test_adv_root_acc_bi})
        epoch_metrics.update({'test_adversarial_acc_animal': test_adv_acc_animal})
        epoch_metrics.update({'test_adversarial_acc_vehicle': test_adv_acc_vehicle})
        # epoch_metrics.update({'test_adversarial_root_acc_animal': test_adv_root_acc_animal})
        # epoch_metrics.update({'test_adversarial_root_acc_vehicle': test_adv_root_acc_vehicle})

    else:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))
    
    # log alpha1, alpha2, alpha3
    # logger.log('Alpha1: {:.4f}.\tAlpha2: {:.4f}.\tAlpha3: {:.4f}'.format(alpha1, alpha2, alpha3))
    # epoch_metrics.update({'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3})

    if test_adv_acc >= old_score[1]:
        old_score[0], old_score[1] = test_acc, test_adv_acc
        trainer.save_model(WEIGHTS)
    trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    #metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)

    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)
    wandb.log(epoch_metrics)

    
    
# Record metrics

train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)['acc']
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0:
    logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

logger.log('Script Completed.')

wandb.summary["final_train_acc"] = train_acc
wandb.summary["final_test_clean_acc"] = old_score[0]
wandb.summary["final_test_adv_acc"] = old_score[1]

wandb.finish()
