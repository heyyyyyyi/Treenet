from .train import Trainer
from .core import animal_classes, vehicle_classes

class TreeEnsemble:
    def __init__(self, 
        info, args,
        alpha1: float = 0.9,
        alpha2: float = 0.1,
        alpha3: float = 0.1,
        max_epochs: int = 100,  # Total number of training epochs
        alpha_update_strategy: dict = None,
    ):
        trainer_lis = [
            Trainer(info, args, tree_model='lighttreeresnet20_root'),
            Trainer(info, args, tree_model='lighttreeresnet20_animal'),
            Trainer(info, args, tree_model='lighttreeresnet20_vehicle'),
        ]
        self.root_trainer = trainer_lis[0]
        self.animal_trainer = trainer_lis[1]
        self.vehicle_trainer = trainer_lis[2]
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
        root_logits, root_features = self.root_trainer.model(x)
        root_pred = torch.argmax(root_logits, dim=1)

        subroot_logits = torch.zeros_like(root_logits)
        animal_classes_index = torch.tensor(animal_classes, device=root_pred.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=root_pred.device)

        is_animal = root_pred.unsqueeze(1) == animal_classes_index
        is_animal = is_animal.any(dim=1)

        is_vehicle = root_pred.unsqueeze(1) == vehicle_classes_index
        is_vehicle = is_vehicle.any(dim=1)

        # Fix for animal subroot logits
        if is_animal.any():
            animal_rows = is_animal.nonzero(as_tuple=True)[0]
            subroot_animal_logits = self.animal_trainer.model(root_features[animal_rows])
            subroot_logits[animal_rows[:, None], animal_classes_index] = subroot_animal_logits[:, :-1]
            unknown_value = subroot_animal_logits[:, -1] / len(vehicle_classes)
            subroot_logits[animal_rows[:, None], vehicle_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(vehicle_classes))

        # Fix for vehicle subroot logits
        if is_vehicle.any():
            vehicle_rows = is_vehicle.nonzero(as_tuple=True)[0]
            subroot_vehicle_logits = self.vehicle_trainer.model(root_features[vehicle_rows])
            subroot_logits[vehicle_rows[:, None], vehicle_classes_index] = subroot_vehicle_logits[:, :-1]
            unknown_value = subroot_vehicle_logits[:, -1] / len(animal_classes)
            subroot_logits[vehicle_rows[:, None], animal_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(animal_classes))
            
            return root_logits, subroot_logits

    def train(self, dataloaders, epoch=0, adversarial=False):
        """
        Train each trainer on a given (sub)set of data.
        """
        # update alpha every epoch 
        self.update_alphas(epoch)
        
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
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
            
            #metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])], ignore_index=True)

        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        return dict(metrics.mean())

    def standard_loss(self, x, y):
        """
        Standard training.
        """
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss = self.root_trainer.standard_loss(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = self.subroot_animal.standard_loss(subroot_logits[is_animal], y[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle = self.subroot_vehicle.standard_loss(subroot_logits[is_vehicle], y[is_vehicle])

        total_loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss = self.root_trainer.adversarial_loss(root_logits, y)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = self.subroot_animal.adversarial_loss(subroot_logits[is_animal], y[is_animal])
        if is_vehicle.any():
            subroot_loss_vehicle = self.subroot_vehicle.adversarial_loss(subroot_logits[is_vehicle], y[is_vehicle])

        total_loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss = self.root_trainer.trades_loss(root_logits, y, beta=beta)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = self.subroot_animal.trades_loss(subroot_logits[is_animal], y[is_animal], beta=beta)
        if is_vehicle.any():
            subroot_loss_vehicle = self.subroot_vehicle.trades_loss(subroot_logits[is_vehicle], y[is_vehicle], beta=beta)

        total_loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def mart_loss(self, x, y, beta):
        """
        MART training.
        """
        root_logits, subroot_logits = self(x)
        preds = torch.argmax(subroot_logits, 1)

        root_loss = self.root_trainer.mart_loss(root_logits, y, beta=beta)

        subroot_loss_animal = torch.tensor(0.0, device=y.device)
        subroot_loss_vehicle = torch.tensor(0.0, device=y.device)

        animal_classes_index = torch.tensor(animal_classes, device=y.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=y.device)

        is_animal = torch_isin(y, animal_classes_index)
        is_vehicle = torch_isin(y, vehicle_classes_index)

        if is_animal.any():
            subroot_loss_animal = self.subroot_animal.mart_loss(subroot_logits[is_animal], y[is_animal], beta=beta)
        if is_vehicle.any():
            subroot_loss_vehicle = self.subroot_vehicle.mart_loss(subroot_logits[is_vehicle], y[is_vehicle], beta=beta)

        total_loss = self.alpha1 * root_loss + self.alpha2 * subroot_loss_animal + self.alpha3 * subroot_loss_vehicle 
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.root_trainer.model.eval()
        self.animal_trainer.model.eval()
        self.vehicle_trainer.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                _, out = self(x_adv)
            else:
                _, out = self(x)
            acc += accuracy(y, out)
        
        return acc / len(dataloader.dataset) 