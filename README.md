# tree_model
## privacy and robustness test 

## model
- CNN
  - original
  - 2 - 4,6 model
  - 10 - 4,6 model
 
## reference
- prime augumentation: https://github.com/amodas/PRIME-augmentations.git

```python 
python -u train.py --config=config/cifar10_cfg.py --config.model.name=lighttreeresnet20 --config.use_prime=True --config.cc_dir=./data/cifar10c/CIFAR-10-C --config.save_dir=./PRIME/linear_lr/
```


- data adversary robustness: https://github.com/imrahulr/adversarial_robustness_pytorch.git

    - train tree model with trades
    ```python 
    python train-tree.py --data-dir ./data     --log-dir ./log_test     --desc trades_tree     --data cifar10     --batch-size 1024     --model lighttreeresnet20     --num-adv-epochs 100 --adv-eval-freq 10 --beta 6  
    ```
    - train tree model with mart 

    ```python 
    python train-tree.py --data-dir ./data     --log-dir ./log_test     --desc test     --data cifar10     --batch-size 1024     --model lighttreeresnet20     --num-adv-epochs 1 --adv-eval-freq 10 --beta 6 --mart
    ```
    - train origin model 
    ```python
    python train.py --data-dir ./data     --log-dir ./log_test     --desc trades_origin     --data cifar10     --batch-size 1024     --model lightresnet20     --num-adv-epochs 100 --adv-eval-freq 10  --beta 6 
    ```
    - evaluate using autoattack
  ```python
  python eval-aa.py --data-dir ./data     --log-dir ./log_test     --desc test     --data cifar10      
  ```
    - evaluate using benchmark
  ```python 
  python eval-rb.py --data-dir ./data     --log-dir ./baseline_log/origin     --desc origin_10_classifier     --data cifar10  --threat Linf
  ```

  - category 
  python category_group.py --data-dir ./data     --log-dir ./baseline_log/origin     --desc origin_10_classifier     --data cifar10