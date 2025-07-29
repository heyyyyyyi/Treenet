import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.treetrain import TreeEnsemble
from core.utils.context import ctx_noparamgrad_and_eval
from core.utils.trades import trades_tree_loss
from core.utils.mart import mart_tree_loss

from core.metrics import accuracy, binary_accuracy, subclass_accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATreeTrainer(TreeEnsemble):
    """
    Helper class for training a tree-based model with model weight averaging.
    """
    def __init__(self, info, args):
        super(WATreeTrainer, self).__init__(info, args)
        self.wa_model = copy.deepcopy(self.model)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        self.update_steps = int(np.floor(num_samples / self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training with weight averaging.
        """
        metrics = pd.DataFrame()
        self.model.train()

        update_iter = 0
        for data in tqdm(dataloader, desc=f'Epoch {epoch}: ', disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            update_iter += 1

            x, y = data
            x, y = x.to(device), y.to(device)

            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y)

            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()

            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])], ignore_index=True)

        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()

        update_bn(self.wa_model, self.model)
        return dict(metrics.mean())

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the weight-averaged model.
        """
        acc, acc_animal, acc_vehicle = 0.0, 0.0, 0.0
        root_acc, root_acc_animal, root_acc_vehicle = 0.0, 0.0, 0.0
        root_acc_bi = 0.0

        self.wa_model.eval()

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                root_out, subroot_animal, subroot_vehicle = self.wa_model(x_adv)
                out = self.forward(x_adv)
            else:
                root_out, subroot_animal, subroot_vehicle = self.wa_model(x)
                out = self.forward(x)

            acc += accuracy(y, out)
            temp_acc_animal, temp_acc_vehicle = subclass_accuracy(y, out)
            acc_animal += temp_acc_animal
            acc_vehicle += temp_acc_vehicle

            root_acc += accuracy(y, root_out)
            temp_root_acc_animal, temp_root_acc_vehicle = subclass_accuracy(y, root_out)
            root_acc_animal += temp_root_acc_animal
            root_acc_vehicle += temp_root_acc_vehicle
            root_acc_bi += binary_accuracy(y, root_out)

        acc /= len(dataloader)
        root_acc /= len(dataloader)
        root_acc_bi /= len(dataloader)
        acc_animal /= len(dataloader)
        acc_vehicle /= len(dataloader)

        return dict(
            acc=acc,
            acc_animal=acc_animal,
            acc_vehicle=acc_vehicle,
            root_acc=root_acc,
            root_acc_animal=root_acc_animal,
            root_acc_vehicle=root_acc_vehicle,
            root_acc_bi=root_acc_bi
        )


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor

    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
