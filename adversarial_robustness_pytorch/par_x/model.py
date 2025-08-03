from core.models.resnet import BasicBlock, Bottleneck
from core.models.resnet import Normalization, LightResnet
import torch.nn as nn
import torch.nn.functional as F
import torch

animal_classes = [ 2, 3, 4, 5, 6, 7]  # 6种动物
vehicle_classes = [0, 1, 8, 9]  # 4种交通工具

# samilar to TreeResNet, change the root model to LightRootResnet, and subroot model to LightSubRootResNet
# 10 dim 
class LightTreeResNet(nn.Module):
    def __init__(self, block, subroot_num_blocks, num_classes=10, device='cpu'):
        super(LightTreeResNet, self).__init__()
        self.subroot_animal = LightResnet(block, subroot_num_blocks, num_classes=6,device=device)
        self.subroot_vehicle = LightResnet(block, subroot_num_blocks, num_classes=4,device=device)
        self.num_classes = num_classes

    def forward(self, x):

        subroot_logits_animal = self.subroot_animal(x)
        subroot_logits_vehicle = self.subroot_vehicle(x)

        logits_animal = torch.zeros_like(self.num_classes)
        logits_vehicle = torch.zeros_like(self.num_classes)

        animal_classes_index = torch.tensor(animal_classes)
        vehicle_classes_index = torch.tensor(vehicle_classes)

        logits_animal[:, animal_classes_index] = subroot_logits_animal[:,:]
        logits_vehicle[:, vehicle_classes_index] = subroot_logits_vehicle[:,:]

        return logits_animal, logits_vehicle

    def load_subroot_animal(self, path):
        """
        Load pre-trained weights for the subroot animal model with error handling.
        """
        try:
            checkpoint = torch.load(path)
            self.subroot_animal.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load subroot animal model from {path}: {e}")

    def load_subroot_vehicle(self, path):
        """
        Load pre-trained weights for the subroot vehicle model with error handling.
        """
        try:
            checkpoint = torch.load(path)
            self.subroot_vehicle.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load subroot vehicle model from {path}: {e}")

# 4+1 dim / 6+1 dim 
class LightTreeResNet_Unknown(nn.Module):
    def __init__(self, block, subroot_num_blocks, num_classes=10,  device='cpu'):
        super(LightTreeResNet_Unknown, self).__init__()
        self.subroot_animal = LightResnet(block, subroot_num_blocks, num_classes=7,device=device)
        self.subroot_vehicle = LightResnet(block, subroot_num_blocks, num_classes=5,device=device)
    
    def forward(self, x):

        subroot_logits_animal = self.subroot_animal(x)
        subroot_logits_vehicle = self.subroot_vehicle(x)
        return subroot_logits_animal, subroot_logits_vehicle

    def load_subroot_animal(self, path):
        """
        Load pre-trained weights for the subroot animal model with error handling.
        """
        try:
            checkpoint = torch.load(path)
            self.subroot_animal.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load subroot animal model from {path}: {e}")

    def load_subroot_vehicle(self, path):
        """
        Load pre-trained weights for the subroot vehicle model with error handling.
        """
        try:
            checkpoint = torch.load(path)
            self.subroot_vehicle.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load subroot vehicle model from {path}: {e}")

def lightresnet(name, num_classes=10, device='cpu', unkown_classes=False):
    """
    Returns suitable Light Resnet model from its name.
    Arguments:
        num_classes (int): number of target classes.
        pretrained (bool): whether to load pretrained weights (not implemented).
        device (str): device to use ('cpu' or 'cuda').
    Returns:
        torch.nn.Module.
    """
    if name == 'lightresnet20':
        return LightResnet(
            BasicBlock, 
            [2, 2, 2],
            num_classes=num_classes, 
            device=device
        )
    
    raise ValueError('Only lightresnet20 is supported!')
    return  

def lighttreeresnet(name, num_classes=10, device='cpu', unkown_classes=False):
    """
    Returns suitable Light Resnet model from its name.
    Arguments:
        num_classes (int): number of target classes.
        pretrained (bool): whether to load pretrained weights (not implemented).
        device (str): device to use ('cpu' or 'cuda').
    Returns:
        torch.nn.Module.
    """
    if name == 'lighttreeresnet20':
        if unkown_classes:
            return LightTreeResNet_Unknown(
                BasicBlock, 
                subroot_num_blocks=[2, 2, 2], 
                num_classes=num_classes, 
                device=device
            )
        else:
            return LightTreeResNet(
                BasicBlock, 
                subroot_num_blocks=[2, 2, 2], 
                num_classes=num_classes, 
                device=device
            )
    
    
    raise ValueError('Only lighttreeresnet20 is supported!')
    return

def create_model(name, normalize, info, device, unknown_classes=True, num_classes=10):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to load pretrained weights (not implemented).
        device (str): device to use ('cpu' or 'cuda').
    Returns:
        torch.nn.Module.
    """
    if "tree" in name:
        backbone = lighttreeresnet(name, num_classes=num_classes, unkown_classes=unknown_classes, device=device)
        if normalize:
            normalization_layer = Normalization(info['mean'], info['std']).to(device)
            backbone.subroot_animal = torch.nn.Sequential(normalization_layer, backbone.subroot_animal)
            backbone.subroot_vehicle = torch.nn.Sequential(normalization_layer, backbone.subroot_vehicle)
        else:
            backbone.subroot_animal = torch.nn.Sequential(backbone.subroot_animal)
            backbone.subroot_vehicle = torch.nn.Sequential(backbone.subroot_vehicle)
        return backbone.to(device)
    
    else: 
        backbone = lightresnet(name, num_classes=num_classes, device=device)
        if normalize:
            normalization_layer = Normalization(info['mean'], info['std']).to(device)
            backbone = torch.nn.Sequential(normalization_layer, backbone)
        else:
            backbone = torch.nn.Sequential(backbone)
        return backbone.to(device)