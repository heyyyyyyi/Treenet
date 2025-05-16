# tree_model
## privacy and robustness test 

## model
- CNN
  - original
  - 2 - 4,6 model
  - 10 - 4,6 model
 
## reference
- prime augumentation: https://github.com/amodas/PRIME-augmentations.git
  - python -u train.py --config=config/cifar10_cfg.py --config.model.name=lighttreeresnet20 --config.use_prime=True --config.cc_dir=./data/cifar10c/CIFAR-10-C --config.save_dir=./PRIME/linear_lr/

- data adversary robustness: https://github.com/imrahulr/adversarial_robustness_pytorch.git
  - python train-tree.py --data-dir ./data     --log-dir ./log1     --desc lighttreeresnet_compare_linear     --data cifar10s     --batch-size 1024     --model lighttreeresnet20     --num-adv-epochs 100
  - 

