"""
Baseline Training Script.
Runs training for three configurations:
1. 10-class classification (info.num_classes = 10).
2. 6-class classification (filter_classes = animal_classes).
3. 4-class classification (filter_classes = vehicle_classes).
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
from core.utils import Trainer
from core.utils import seed
from core import animal_classes, vehicle_classes

import wandb

_WANDB_USERNAME = "yhe106-johns-hopkins-university"
_WANDB_PROJECT = "baseline-adv-resnet20"

def run_training(desc, num_classes, filter_classes=None, eval_metrics=None):
    # Setup
    parse = parser_train()
    args = parse.parse_args()
    args.desc = desc
    args.num_classes = num_classes

    wandb.init(
        project=_WANDB_PROJECT, entity=_WANDB_USERNAME,
        name=args.desc,
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
    info.num_classes = num_classes  # Update number of classes
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
        filter_classes=filter_classes
    )
    del train_dataset, test_dataset

    # Training
    seed(args.seed)
    trainer = Trainer(info, args)
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

    for epoch in range(1, NUM_ADV_EPOCHS+1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))
        
        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]
        
        res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
        test_res = trainer.eval(test_dataloader)
        
        epoch_metrics = {'train_'+k: v for k, v in res.items()}
        epoch_metrics.update({
            'epoch': epoch, 
            'lr': last_lr, 
        })
        
        for metric in eval_metrics:
            epoch_metrics[f'test_clean_{metric}'] = test_res[f'acc_{metric}']
            epoch_metrics[f'test_adversarial_{metric}'] = None
        
        if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5):
            test_adv_res = trainer.eval(test_dataloader, adversarial=True)
            for metric in eval_metrics:
                epoch_metrics[f'test_adversarial_{metric}'] = test_adv_res[f'acc_{metric}']
        
        if test_res['acc'] >= old_score[1]:
            old_score[0], old_score[1] = test_res['acc'], test_adv_res['acc']
            trainer.save_model(WEIGHTS)
        trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

        logger.log('Time taken: {}'.format(format_time(time.time()-start)))
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)
        wandb.log(epoch_metrics)

    wandb.summary["final_train_acc"] = res['clean_acc']
    wandb.summary["final_test_clean_acc"] = old_score[0]
    wandb.summary["final_test_adv_acc"] = old_score[1]
    wandb.finish()

# Run training for three configurations
run_training(desc="400_10-class", num_classes=10, eval_metrics=["animal", "vehicle", "bi"])
run_training(desc="400_6-class-animal", num_classes=7, filter_classes=animal_classes, eval_metrics=["animal"])
run_training(desc="400_4-class-vehicle", num_classes=5, filter_classes=vehicle_classes, eval_metrics=["vehicle"])
