import os
import numpy as np

import torch
import pytorch_lightning as pl
import torchvision.datasets as dset
import torchvision.transforms as T

from absl import app, flags
from ml_collections.config_flags import config_flags
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from lightning.data import _DATASETS
from lightning.data import get_cc_dataset
from lightning.data import get_dataset

from models import AugModel, TransformLayer
from models import get_model, _MODELS

from lightning.systems import AugClassifier
from lightning.systems import Classifier
from lightning.systems import TreeClassifier
from lightning.systems import TreeAugClassifier

import utils
from utils.rand_filter import RandomFilter
from utils.color_jitter import RandomSmoothColor
from utils.diffeomorphism import Diffeo

from utils.augmix import AugMixDataset
from utils.prime import GeneralizedPRIMEModule
from utils.prime import PRIMEAugModule

from setup import setup_all, _setup

_WANDB_USERNAME = "yhe106-johns-hopkins-university"
_WANDB_PROJECT = "PRIME-light20"

def validate_config(cfg):
    if cfg.dataset not in _DATASETS:
        raise ValueError(f'Dataset {cfg.dataset} not supported!')
    elif cfg.model.name not in _MODELS[cfg.dataset]:
        raise ValueError(f'Model {cfg.model.name} not supported!')
    assert not (cfg.use_augmix and cfg.use_prime), 'Use only one augmentation!'
    if cfg.use_deepaugment:
        assert 'imagenet' in cfg.dataset, 'DeepAugment only supported on ImageNet!'
    
    if 'TMPDIR' in os.environ and 'imagenet' in cfg.dataset:
        setup_all(cfg.data_dir, cfg.cc_dir)
        if cfg.use_deepaugment:
            _setup(cfg.data_dir, 'EDSR')
            _setup(cfg.data_dir, 'CAE')
        cfg.data_dir = os.path.join(os.environ['TMPDIR'], cfg.dataset)
        cfg.cc_dir = os.path.join(os.environ['TMPDIR'], f'{cfg.dataset}c')

def test_model(config, checkpoint_path):
    # Validate config
    validate_config(config)
    # Initialize Wandb logger
    wandb_logger = WandbLogger(project=_WANDB_PROJECT, entity=_WANDB_USERNAME, log_model=False)

    # Load dataset
    dataset = get_dataset(config.dataset)(
        config.data_dir,
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        num_workers=config.test_num_workers,
    )

    transforms = [] if config.use_augmix else [T.ToTensor()]
    if 'imagenet' in config.dataset:
        transforms += [
            T.RandomResizedCrop(224), T.RandomHorizontalFlip()
        ]
        if not (config.use_prime or config.use_augmix):
            transforms.append(T.Normalize(dataset.mean, dataset.std))
        
        dataset.train_transform = T.Compose(transforms)
        dataset.test_transform = T.Compose([
            T.ToTensor(), T.Resize(256), T.CenterCrop(224),
            T.Normalize(dataset.mean, dataset.std)
        ])
    elif 'cifar' in config.dataset:
        transforms += [
            T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()
        ]
        if not (config.use_prime or config.use_augmix):
            transforms.append(T.Normalize(dataset.mean, dataset.std))
        
        dataset.train_transform = T.Compose(transforms)
        dataset.test_transform = T.Compose([
            T.ToTensor(), T.Normalize(dataset.mean, dataset.std)
        ])
        
    dataset.prepare_data()
    dataset.setup()

    if config.use_prime:
        augmentations = []

        if config.enable_aug.diffeo:
            diffeo = Diffeo(
                sT=config.diffeo.sT, rT=config.diffeo.rT,
                scut=config.diffeo.scut, rcut=config.diffeo.rcut,
                cutmin=config.diffeo.cutmin, cutmax=config.diffeo.cutmax,
                alpha=config.diffeo.alpha, stochastic=True
            )
            augmentations.append(diffeo)

        if config.enable_aug.color_jit:
            color = RandomSmoothColor(
                cut=config.color_jit.cut, T=config.color_jit.T,
                freq_bandwidth=config.color_jit.max_freqs, stochastic=True
            )
            augmentations.append(color)

        if config.enable_aug.rand_filter:
            filt = RandomFilter(
                kernel_size=config.rand_filter.kernel_size,
                sigma=config.rand_filter.sigma, stochastic=True
            )
            augmentations.append(filt)
        
        prime_module = GeneralizedPRIMEModule(
            preprocess=TransformLayer(dataset.mean, dataset.std),
            mixture_width=config.augmix.mixture_width,
            mixture_depth=config.augmix.mixture_depth,
            no_jsd=config.augmix.no_jsd, max_depth=3,
            aug_module=PRIMEAugModule(augmentations),
        )

    # Load model
    base_model = get_model(config.dataset, config.model.name)(
        num_classes=dataset.num_classes, pretrained=False
    )
    if "tree" in config.model.name:
        if config.use_prime:
            model = TreeAugClassifier(
                model=AugModel(model=base_model, aug=prime_module),
                no_jsd=config.augmix.no_jsd, 
            )
        elif config.use_augmix:
            model = TreeAugClassifier(
                model=base_model, no_jsd=config.augmix.no_jsd, 
            )
        else:
            model = TreeClassifier(model=base_model)
    else:
        if config.use_prime:
            model = AugClassifier(
                model=AugModel(model=base_model, aug=prime_module),
                no_jsd=config.augmix.no_jsd, 
            )
        elif config.use_augmix:
            model = AugClassifier(
                model=base_model, no_jsd=config.augmix.no_jsd, 
            )
        else:
            model = Classifier(model=base_model)

    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Initialize trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1,
        accelerator=config.accelerator,
        benchmark=True,
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.test(model, dataloaders=dataset.test_dataloader())
    print(f"Test Results: {test_results}")

    # Evaluate on common corruptions
    print("Evaluating on common corruptions...")
    dataset_c = get_cc_dataset(config.dataset)(
        config.cc_dir, batch_size=config.test_batch_size,
        num_workers=config.test_num_workers
    )
    transforms = [] if 'cifar' in config.dataset else [T.ToTensor()]
    transforms.append(T.Normalize(dataset.mean, dataset.std))
    dataset_c.transform = T.Compose(transforms)
    dataset_c.prepare_data()
    dataset_c.setup()

    cc_loaders = dataset_c.test_dataloader()
    keys = list(cc_loaders.keys())
    avg_acc = 0.
    for key in keys:
        res = trainer.test(model, test_dataloaders=cc_loaders[key])
        acc = res[0]["test.acc"]
        wandb.run.summary["test.%s" % key] = acc
        avg_acc += acc
    wandb.run.summary["test_avg.acc"] = avg_acc / len(keys)
    print(f"Average Accuracy on Common Corruptions: {avg_acc / len(keys):.4f}")

if __name__ == "__main__":
    config_flags.DEFINE_config_file('config')
    flags.DEFINE_string('checkpoint_path', None, 'Path to the trained model checkpoint')
    FLAGS = flags.FLAGS

    def main(_):
        if FLAGS.checkpoint_path is None:
            raise ValueError("Please provide a valid checkpoint path using --checkpoint_path")
        test_model(FLAGS.config, FLAGS.checkpoint_path)

    app.run(main)