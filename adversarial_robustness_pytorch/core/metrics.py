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


    
