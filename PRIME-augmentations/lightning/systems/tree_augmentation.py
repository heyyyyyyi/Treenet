import flatten_dict
import ml_collections
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from lightning.systems.tree_classification import TreeClassifier

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


class TreeAugClassifier(TreeClassifier):
    def __init__(
        self,
        model: nn.Module = None,
        optimizer_cfg: ml_collections.ConfigDict = None,
        lr_schedule_cfg: ml_collections.ConfigDict = None,
        no_jsd: bool = False,
        scheduler_t: str = "cyclic",
        test_keys=None,
        alpha1: float = 0.9,
        alpha2: float = 0.1,
        alpha3: float = 0.1,
    ):
        super().__init__(
            model, optimizer_cfg, lr_schedule_cfg, 
            scheduler_t=scheduler_t, test_keys=test_keys, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3)
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.loss_ema = 0.0
        self.no_jsd = no_jsd
    
    def compute_subroot_loss(self,logits, logits_aug1, logits_aug2, y, class_idx):
        mask = torch_isin(y, class_idx)
        if not mask.any():
            return torch.tensor(0.0, device=y.device)

        ce_loss = F.cross_entropy(logits[mask], y[mask])
        p_clean, p_aug1, p_aug2 = F.softmax(logits[mask], dim=1), F.softmax(logits_aug1[mask], dim=1), F.softmax(logits_aug2[mask], dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

        kl_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        return ce_loss + kl_loss

    
    def training_step(self, batch, batch_idx):

        if self.no_jsd:
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
            
            return {'loss': total_loss, 'preds': preds, 'targets': y}

        else:
            x, y = batch
            if len(batch[0]) == 3: # for AugMix
                x = torch.cat(x, 0)
            
            root_logits_all, subroot_logits_all = self.model(x)
            if len(batch[0]) == 3: # for AugMix
                root_logits, root_logits_aug1, root_logits_aug2 = torch.split(
                    root_logits_all, batch[0][0].size(0))
                subroot_logits, subroot_logits_aug1, subroot_logits_aug2 = torch.split(
                    subroot_logits_all, batch[0][0].size(0))
            else:
                root_logits, root_logits_aug1, root_logits_aug2 = torch.split(
                    root_logits_all, x.size(0))
                subroot_logits, subroot_logits_aug1, subroot_logits_aug2 = torch.split(
                    subroot_logits_all, x.size(0))

            preds = torch.argmax(subroot_logits, 1)
            root_loss = F.cross_entropy(root_logits, y)

            p_clean, p_aug1, p_aug2 = F.softmax(
                root_logits, dim=1), F.softmax(
                root_logits_aug1, dim=1), F.softmax(
                root_logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            root_loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            subroot_loss_animal = torch.tensor(0.0, device=y.device)
            subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

            animal_classes_index = torch.tensor(animal_classes, device=y.device)
            vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

            is_animal = torch_isin(y, animal_classes_index)
            is_vehicle = torch_isin(y, vehicle_classes_index)

            if is_animal.any():
                subroot_loss_animal = self.compute_subroot_loss(subroot_logits, subroot_logits_aug1, subroot_logits_aug2, y, animal_classes_index)
            if is_vehicle.any():
                subroot_loss_vehicle = self.compute_subroot_loss(subroot_logits, subroot_logits_aug1, subroot_logits_aug2, y, vehicle_classes_index)

            loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
            # calculate kl divergence and add to loss todo 
            self.loss_ema = self.loss_ema * 0.9 + loss * 0.1

            return {"loss": loss, "preds": preds, "targets": y, "loss_ema": self.loss_ema}

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]
        losses = batch_parts["loss"]

        loss = losses.mean()
        self.train_acc(preds, targets)
        self.log("train.acc", self.train_acc, on_step=True)
        self.log("train.loss", loss, on_step=True)
        if not self.no_jsd:
            self.log("train.loss_ema", batch_parts["loss_ema"].mean(), on_step=True)

        return loss

    