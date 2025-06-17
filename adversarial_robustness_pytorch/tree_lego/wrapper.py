from core import animal_classes, vehicle_classes
import torch
class root_wrapper(object):
    """
    Wrapper class for the tree root model.
    """
    def __init__(self, model):
        self.model = model
    def forward(self, x):
        """
        Forward pass through the model.
        """
        root_logits, root_features = self.model(x)
        return root_logits
    def __call__(self, x):
        """
        Call the forward method.
        """
        return self.forward(x)
    

class model_wrapper(object):
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        root_logits, subroot_animal_logits, subroot_vehicle_logits = self.model(x)
        root_pred = torch.argmax(root_logits, dim=1)

        subroot_logits = torch.zeros_like(root_logits)
        animal_classes_index = torch.tensor(animal_classes, device=x.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=x.device)

        # Fix for torch.isin usage
        is_animal = torch.isin(root_pred, animal_classes_index).to(x.device)
        is_vehicle = torch.isin(root_pred, vehicle_classes_index).to(x.device)

        # Fix for animal subroot logits
        if is_animal.any():
            subroot_logits[is_animal] = subroot_animal_logits[is_animal]

        if is_vehicle.any():
            subroot_logits[is_vehicle] = subroot_vehicle_logits[is_vehicle]

        return subroot_logits
    
    def __call__(self, x):
        return self.forward(x)

class animal_wrapper(object):
    def __init__(self, root_model, subroot_model):
        self.root_model = root_model
        self.subroot_model = subroot_model

    def forward(self, x):
        # Freeze root_model during forward pass
        with torch.no_grad():
            _, root_features = self.root_model(x)
        subroot_logits_animal = self.subroot_model(root_features)

        logits_animal = torch.zeros((x.size(0), 10), device=x.device)
        animal_classes_index = torch.tensor(animal_classes, device=x.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=x.device)

        logits_animal[:, animal_classes_index] = subroot_logits_animal[:, :-1]
        unknown_value = subroot_logits_animal[:, -1] / len(vehicle_classes)
        logits_animal[:, vehicle_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(vehicle_classes))

        return logits_animal

    def __call__(self, x):
        return self.forward(x)


class vehicle_wrapper(object):
    def __init__(self, root_model, subroot_model):
        self.root_model = root_model
        self.subroot_model = subroot_model

    def forward(self, x):
        # Freeze root_model during forward pass
        with torch.no_grad():
            _, root_features = self.root_model(x)
        subroot_logits_vehicle = self.subroot_model(root_features)

        logits_vehicle = torch.zeros((x.size(0), 10), device=x.device)
        animal_classes_index = torch.tensor(animal_classes, device=x.device)
        vehicle_classes_index = torch.tensor(vehicle_classes, device=x.device)

        logits_vehicle[:, vehicle_classes_index] = subroot_logits_vehicle[:, :-1]
        unknown_value = subroot_logits_vehicle[:, -1] / len(animal_classes)
        logits_vehicle[:, animal_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(animal_classes))

        return logits_vehicle

    def __call__(self, x):
        return self.forward(x)