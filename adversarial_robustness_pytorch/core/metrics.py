import torch
from core import animal_classes, vehicle_classes

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()/float(true.size(0))
    return accuracy.item()

def binary_accuracy(true, preds):
    """
    Computes binary accuracy if label in animal_classes (1) if in vehicle_classes (0). Used in root_acc_bi.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Binary accuracy.
    """
    # Convert true labels to binary: 1 for animal_classes, 0 for vehicle_classes
    true_binary = torch.isin(true, torch.tensor(animal_classes, device=true.device)).long()

    # Convert predicted labels to binary: 1 for animal_classes, 0 for vehicle_classes
    preds_binary = torch.isin(preds.argmax(dim=1), torch.tensor(animal_classes, device=preds.device)).long()

    # Compute binary accuracy
    accuracy = (true_binary == preds_binary).sum().float() / float(true.size(0))
    return accuracy.item()

def subclass_accuracy(true, preds):
    """
    Computes subroot accuracy. Based on true labels, acc_animal computes accuracy of samples in animal_classes,
    and acc_vehicle computes accuracy of samples in vehicle_classes.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Subroot accuracy.
    """
    animal_indices = torch.isin(true, torch.tensor(animal_classes, device=true.device))
    vehicle_indices = torch.isin(true, torch.tensor(vehicle_classes, device=true.device))
    if animal_indices.sum() > 0:
        acc_animal = (
            torch.softmax(preds[animal_indices], dim=1).argmax(dim=1) == true[animal_indices]
        ).sum().float() / float(true[animal_indices].size(0))
    else:
        acc_animal = torch.tensor(0.0, device=true.device)

    if vehicle_indices.sum() > 0:
        acc_vehicle = (
            torch.softmax(preds[vehicle_indices], dim=1).argmax(dim=1) == true[vehicle_indices]
        ).sum().float() / float(true[vehicle_indices].size(0))
    else:
        acc_vehicle = torch.tensor(0.0, device=true.device)

    return acc_animal.item(), acc_vehicle.item()

    
