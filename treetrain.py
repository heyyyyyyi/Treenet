import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import flatten_dict
import ml_collections
import torchmetrics
animal_classes = [2, 3, 4, 5, 6, 7]
vehicle_classes = [0, 1, 8, 9]

def torch_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
    """
    Replacement for torch.isin for PyTorch < 1.10
    """
    test_elements = test_elements.to(elements.device)

    elements_shape = elements.shape
    elements_flat = elements.flatten()
    test_elements_flat = test_elements.flatten()
    result = (elements_flat[..., None] == test_elements_flat).any(-1)
    return result.reshape(elements_shape)

# alpha1 alpha2 alpha3 control loss 
# learning rate if three model ... 

class TreeClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module = None,
        optimizer_cfg: ml_collections.ConfigDict = None,
        lr_schedule_cfg: ml_collections.ConfigDict = None,
        scheduler_t: str = "cyclic",
        test_keys=None,
        alpha1: float = 0.9,
        alpha2: float = 0.1,
        alpha3: float = 0.1,
        max_epochs: int = 100,  # Total number of training epochs
        alpha_update_strategy: dict = None,  # Strategy for alpha adjustment
    ):
        super().__init__()
        self.model = model

        self.optimizer_cfg = optimizer_cfg
        self.lr_schedule_cfg = lr_schedule_cfg
        self.scheduler_t = scheduler_t

        self.train_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.val_acc = torchmetrics.Accuracy(dist_sync_on_step=True)

        if test_keys is None:
            self.test_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        else:
            self.test_acc = nn.ModuleDict({key: torchmetrics.Accuracy() for key in test_keys})

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.max_epochs = max_epochs
        self.alpha_update_strategy = alpha_update_strategy or {
            "balance_ratio": 2 / 3,  # alpha2:alpha3 ratio, e.g., 6:4 for animal vs vehicle
        }

    def update_alphas(self, current_epoch: int):
        """
        Dynamically update alpha1, alpha2, and alpha3 based on the current epoch.
        """
        progress = current_epoch / self.max_epochs  # Calculate training progress (0 to 1)
        self.alpha1 = max(0.0, 0.9 * (1 - progress))  # Decrease alpha1 from 0.9 to 0
        alpha23_total = 0.1 + 0.8 * progress  # Increase alpha2 + alpha3 from 0.1 to 0.9

        # Split alpha23_total between alpha2 and alpha3 based on the balance ratio
        balance_ratio = self.alpha_update_strategy["balance_ratio"]
        self.alpha2 = alpha23_total * balance_ratio / (1 + balance_ratio)
        self.alpha3 = alpha23_total / (1 + balance_ratio)

    def forward(self, x):
        root_logits, subroot_logits = self.model(x)
        return subroot_logits

    def configure_optimizers(self):
        """
        Configure optimizers for the root, subroot animal, and subroot vehicle models.
        """
        root_params = self.model.root_model.parameters()
        subroot_animal_params = self.model.subroot_animal.parameters()
        subroot_vehicle_params = self.model.subroot_vehicle.parameters()

        # Define separate learning rates for each model
        root_lr = self.optimizer_cfg.get("root_lr", 1e-3)
        subroot_animal_lr = self.optimizer_cfg.get("subroot_animal_lr", 1e-3)
        subroot_vehicle_lr = self.optimizer_cfg.get("subroot_vehicle_lr", 1e-3)

        # Create optimizers for each model
        root_optimizer = torch.optim.SGD(root_params, lr=root_lr)
        subroot_animal_optimizer = torch.optim.SGD(subroot_animal_params, lr=subroot_animal_lr)
        subroot_vehicle_optimizer = torch.optim.SGD(subroot_vehicle_params, lr=subroot_vehicle_lr)

        # Combine optimizers into a single list
        optimizers = [root_optimizer, subroot_animal_optimizer, subroot_vehicle_optimizer]

        # Optionally configure learning rate schedulers
        if self.scheduler_t == "piecewise_constant":
            schedulers = {
                'root_optimizer': torch.optim.lr_scheduler.StepLR(root_optimizer, **self.lr_schedule_cfg),
                'subroot_animal_optimizer': torch.optim.lr_scheduler.StepLR(subroot_animal_optimizer, **self.lr_schedule_cfg),
                'subroot_vehicle_optimizer': torch.optim.lr_scheduler.StepLR(subroot_vehicle_optimizer, **self.lr_schedule_cfg),
                'interval': 'epoch',
            }
        elif self.scheduler_t == "cyclic":
            schedulers = {
                'root_optimizer': torch.optim.lr_scheduler.OneCycleLR(root_optimizer, **self.lr_schedule_cfg),
                'subroot_animal_optimizer': torch.optim.lr_scheduler.OneCycleLR(subroot_animal_optimizer, **self.lr_schedule_cfg),
                'subroot_vehicle_optimizer': torch.optim.lr_scheduler.OneCycleLR(subroot_vehicle_optimizer, **self.lr_schedule_cfg),
                'interval': 'step',
            }
        return [optimizers], [schedulers]

    def training_step(self, batch, batch_idx):
        x, y = batch
        root_logits, subroot_logits = self.model(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss = F.cross_entropy(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = F.cross_entropy(subroot_logits[is_animal], y[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle = F.cross_entropy(subroot_logits[is_vehicle], y[is_vehicle])

        total_loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        return {'total_loss': total_loss, 'subroot_loss_animal': subroot_loss_animal, 'subroot_loss_vehicle': subroot_loss_vehicle, 'preds': preds, 'targets': y}

    def on_train_epoch_start(self):
        """
        Hook to update alphas at the start of each training epoch.
        """
        self.update_alphas(self.current_epoch)
        self.log("alpha1", self.alpha1, on_epoch=True)
        self.log("alpha2", self.alpha2, on_epoch=True)
        self.log("alpha3", self.alpha3, on_epoch=True)
    
    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]
        losses = batch_parts["loss"]

        loss = losses.mean()
        self.train_acc(preds, targets)
        self.log("train.acc", self.train_acc, on_step=True)
        self.log("train.loss", loss, on_step=True)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)
        return {"preds": preds, "targets": y}
    
    def validation_step_end(self, batch_parts):
        preds=batch_parts["preds"]
        targets=batch_parts["targets"]

        self.val_acc(preds, targets)
        self.log("val.acc", self.val_acc, on_step=False, on_epoch=True)
         
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, dict):
            preds = {}
            for key, (x, y) in flatten_dict.flatten(batch, reducer="dot").items():
                logits = self(x)
                preds[key] = torch.argmax(logits, 1)

            return {"targets": y, **preds}
        else:
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, 1)
            return {"preds": preds, "targets": y}

    def test_step_end(self, batch_parts):
        if len(batch_parts.keys()) > 2:
            targets = batch_parts.pop("targets")
            avg_acc = []
            for key, preds in batch_parts.items():
                cur_acc = self.test_acc[key](preds, targets)
                self.log(f"test.{key}", self.test_acc[key], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                avg_acc.append(cur_acc)
            self.log("test_avg.acc", torch.tensor(avg_acc).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            preds = batch_parts["preds"]
            targets = batch_parts["targets"]
            self.test_acc(preds, targets)
            self.log("test.acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
