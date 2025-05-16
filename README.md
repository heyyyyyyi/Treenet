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
  - python train-tree.py --data-dir ./data     --log-dir ./log2     --desc lighttreeresnet_2112     --data cifar10s     --batch-size 1024     --model lighttreeresnet20     --num-adv-epochs 100
  - 


  
python train.py --data-dir ./data     --log-dir ./log2/origin     --desc lightresnet     --data cifar10s     --batch-size 1024     --model lightresnet20     --num-adv-epochs 100 && python train-tree.py --data-dir ./data     --log-dir ./log2/2112     --desc lighttreeresnet_2112     --data cifar10s     --batch-size 1024  --model lighttreeresnet20     --num-adv-epochs 100 && cd ../PRIME-augmentations/ && conda activate prime && python -u train.py --config=config/cifar10_cfg.py --config.model.name=lightresnet20 --config.use_prime=True --config.cc_dir=./data/cifar10c/CIFAR-10-C --config.save_dir=./PRIME/origin/
