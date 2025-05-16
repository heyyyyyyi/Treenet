import flatten_dict
import ml_collections
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
import pytorch_lightning as pl

from lightning import animal_classes, vehicle_classes

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

class TreeClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module = None,
        optimizer_cfg: ml_collections.ConfigDict = None,
        lr_schedule_cfg: ml_collections.ConfigDict = None,
        scheduler_t: str = "cyclic",
        test_keys=None,
        alpha1: float = 0.9,
        alpha2: float = 0.06,
        alpha3: float = 0.04,
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
        self.rootval_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.val_animal_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.val_vehicle_acc = torchmetrics.Accuracy(dist_sync_on_step=True)

        self.latest_rootval_acc = 0.0  

        if test_keys is None:
            self.test_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
            self.test_animal_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
            self.test_vehicle_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        else:
            self.test_acc = nn.ModuleDict({key: torchmetrics.Accuracy() for key in test_keys})
            self.test_animal_acc = nn.ModuleDict({key: torchmetrics.Accuracy() for key in test_keys})
            self.test_vehicle_acc = nn.ModuleDict({key: torchmetrics.Accuracy() for key in test_keys})

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.max_epochs = max_epochs
        self.alpha_update_strategy = alpha_update_strategy or {
            "balance_ratio": 3 / 2,  # alpha2:alpha3 ratio, e.g., 6:4 for animal vs vehicle
        }

    def update_alphas(self, current_epoch: int, root_acc: float):
        """
        Dynamically update alpha1, alpha2, and alpha3 based on the current epoch.
        """
        progress = current_epoch / self.max_epochs  # Calculate training progress (0 to 1)
        self.alpha1 = max(0.0, 0.9 * (1 - progress))  # Decrease alpha1 from 0.9 to 0
        alpha23_total = 0.1 + 0.8 * progress  # Increase alpha2 + alpha3 from 0.1 to 0.9

        # Use a custom strategy if provided
        if callable(self.alpha_update_strategy):
            self.alpha2, self.alpha3 = self.alpha_update_strategy(alpha23_total, progress)
        else:
            # Split alpha23_total between alpha2 and alpha3 based on the balance ratio
            balance_ratio = self.alpha_update_strategy["balance_ratio"]
            self.alpha2 = alpha23_total * balance_ratio / (1 + balance_ratio)
            self.alpha3 = alpha23_total / (1 + balance_ratio)
        


    # def update_alphas(self, current_epoch: int, root_acc: float):
    #     """
    #     Update alpha1, alpha2, alpha3 dynamically based on epoch and root accuracy.
    #     This ensures: alpha1 dominates early, then reduces over time, adjusted by root acc.

    #     alpha1 : 0.9 -> 0.5 -> 0.1 
    #     """
    
    #     if current_epoch < self.max_epochs * 0.15 and root_acc < 0.75:
    #         self.alpha1 = 0.9
    #     elif current_epoch > self.max_epochs * 0.7:
    #         self.alpha1 = 0.1
    #     else:
    #         self.alpha1 = 0.5

    #     balance_ratio = self.alpha_update_strategy["balance_ratio"]

    #     # Remaining portion goes to alpha2 and alpha3
    #     alpha23_total = 1.0 - self.alpha1
    #     self.alpha2 = alpha23_total * balance_ratio / (1 + balance_ratio)
    #     self.alpha3 = alpha23_total / (1 + balance_ratio)


    def forward(self, x):
        root_logits, subroot_logits = self.model(x)
        return root_logits, subroot_logits

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_cfg)
        if self.scheduler_t == "cyclic":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.lr_schedule_cfg),
                "interval": "step",
            }
        elif self.scheduler_t == "piecewise_constant":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_schedule_cfg),
                "interval": "epoch",
            }
        return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        # 1. 多 param group 支持：分别设置 lr
        #if self.model has attribute "root_model":
        if hasattr(self.model, "root_model"):
            root_params = self.model.root_model.parameters()
            animal_params = self.model.subroot_animal.parameters()
            vehicle_params = self.model.subroot_vehicle.parameters()
        else:
            root_params = self.model.model.root_model.parameters()
            animal_params = self.model.model.subroot_animal.parameters()
            vehicle_params = self.model.model.subroot_vehicle.parameters()
        
        optimizer = torch.optim.SGD([
            {"params": root_params, "lr": self.optimizer_cfg.lr * 0.5},
            {"params": animal_params, "lr": self.optimizer_cfg.lr * 1.0},
            {"params": vehicle_params, "lr": self.optimizer_cfg.lr * 1.5},
        ], 
            momentum=self.optimizer_cfg.momentum,
            weight_decay=self.optimizer_cfg.weight_decay,
            nesterov=self.optimizer_cfg.nesterov
        )


        # 2. scheduler max_lr 对应多个 param group
        if self.scheduler_t == "cyclic":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[
                        self.lr_schedule_cfg["max_lr"] * 0.5,
                        self.lr_schedule_cfg["max_lr"] * 1.0,
                        self.lr_schedule_cfg["max_lr"] * 1.5
                    ],
                    pct_start=self.lr_schedule_cfg["pct_start"],
                    steps_per_epoch=self.lr_schedule_cfg["steps_per_epoch"],
                    epochs=self.lr_schedule_cfg["epochs"]
                ),
                "interval": "step",
            }

        elif self.scheduler_t == "piecewise_constant":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_schedule_cfg),
                "interval": "epoch",
            }

        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        x, y = batch
        root_logits, subroot_logits = self(x)
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
        
        return {'loss': total_loss, 'preds': preds, 'targets': y}

    def on_train_epoch_start(self):
        """
        Hook to update alphas at the start of each training epoch.
        """
        # get root validation accuracy on last epoch
        self.update_alphas(self.current_epoch, self.latest_rootval_acc)
        
        self.log("alpha1", self.alpha1, on_epoch=True)
        self.log("alpha2", self.alpha2, on_epoch=True)
        self.log("alpha3", self.alpha3, on_epoch=True)
        #self.log("root_loss", self.root_loss, on_epoch=True)
        
    
    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]
        losses = batch_parts["loss"]

        loss = losses.mean()
        self.train_acc(preds, targets)
        #self.log("train.acc", self.train_acc, on_step=True)
        #self.log("train.loss", loss, on_step=True)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        root_logits, subroot_logits = self(x)
        root_preds = torch.argmax(root_logits, 1)
        subroot_preds = torch.argmax(subroot_logits, 1)
        return {"root_preds": root_preds, "subroot_preds": subroot_preds, "targets": y}
    
    def validation_step_end(self, batch_parts):
        preds=batch_parts["subroot_preds"]
        targets=batch_parts["targets"]

        self.val_acc(preds, targets)
        self.rootval_acc(batch_parts["root_preds"], targets)
        is_animal = torch_isin(targets, torch.tensor(animal_classes, device=targets.device))
        is_vehicle = torch_isin(targets, torch.tensor(vehicle_classes, device=targets.device))
        self.val_animal_acc(preds[is_animal], targets[is_animal])
        self.val_vehicle_acc(preds[is_vehicle], targets[is_vehicle])

        self.log("val.acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("rootval.acc", self.rootval_acc, on_step=False, on_epoch=True)
        self.log("val.animal.acc", self.val_animal_acc, on_step=False, on_epoch=True)
        self.log("val.vehicle.acc", self.val_vehicle_acc, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        """
        Hook to compute and store the latest validation root accuracy.
        """
        # Compute the accumulated root validation accuracy
        self.latest_rootval_acc = self.rootval_acc.compute().item()  # Store it as a class attribute


    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, dict):
            preds = {}
            for key, (x, y) in flatten_dict.flatten(batch, reducer="dot").items():
                _, logits = self(x)
                preds[key] = torch.argmax(logits, 1)

            return {"targets": y, **preds}
        else:
            x, y = batch
            _, logits = self(x)
            preds = torch.argmax(logits, 1)
            return {"preds": preds, "targets": y}
    
    def test_step_end(self, batch_parts):
        if len(batch_parts.keys()) > 2:
            targets = batch_parts.pop("targets")
            avg_acc = []
            for key, preds in batch_parts.items():
                cur_acc = self.test_acc[key](preds, targets)
                self.log(f"test.{key}", self.test_acc[key], on_step=False, on_epoch=True)
                avg_acc.append(cur_acc)

            self.log("test_avg.acc", torch.tensor(avg_acc).mean(), on_step=False, on_epoch=True)
            is_animal = torch_isin(targets, torch.tensor(animal_classes, device=targets.device))
            is_vehicle = torch_isin(targets, torch.tensor(vehicle_classes, device=targets.device))
            self.test_animal_acc(preds[is_animal], targets[is_animal])
            self.test_vehicle_acc(preds[is_vehicle], targets[is_vehicle])
            self.log("test.animal.acc", self.test_animal_acc, on_step=False, on_epoch=True)
            self.log("test.vehicle.acc", self.test_vehicle_acc, on_step=False, on_epoch=True)

        else:
            preds = batch_parts["preds"]
            targets = batch_parts["targets"]
            self.test_acc(preds, targets)
            self.log("test.acc", self.test_acc, on_step=False, on_epoch=True)
            
            is_animal = torch_isin(targets, torch.tensor(animal_classes, device=targets.device))
            is_vehicle = torch_isin(targets, torch.tensor(vehicle_classes, device=targets.device))
            self.test_animal_acc(preds[is_animal], targets[is_animal])
            self.test_vehicle_acc(preds[is_vehicle], targets[is_vehicle])
            self.log("test.animal.acc", self.test_animal_acc, on_step=False, on_epoch=True)
            self.log("test.vehicle.acc", self.test_vehicle_acc, on_step=False, on_epoch=True)