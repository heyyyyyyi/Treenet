from core.models.resnet import BasicBlock, Bottleneck
from core.models.resnet import Normalization, LightResnet
import torch.nn as nn
import torch.nn.functional as F
import torch

animal_classes = [ 2, 3, 4, 5, 6, 7]  # 6种动物
vehicle_classes = [0, 1, 8, 9]  # 4种交通工具

# samilar to TreeResNet, change the root model to LightRootResnet, and subroot model to LightSubRootResNet
class LightTreeResNet(nn.Module):
    def __init__(self, block, root_num_blocks, subroot_num_blocks, num_classes=10, device='cpu'):
        super(LightTreeResNet, self).__init__()
        self.root_model = LightResnet(block, root_num_blocks, num_classes=num_classes,device=device)  # 10分类 
        self.subroot_animal = LightResnet(block, subroot_num_blocks, num_classes=6,device=device)
        self.subroot_vehicle = LightResnet(block, subroot_num_blocks, num_classes=4,device=device)
    
    def forward(self, x):
        root_logits = self.root_model(x)
        subroot_logits_animal = self.subroot_animal(x)
        subroot_logits_vehicle = self.subroot_vehicle(x)

        logits_animal = torch.zeros_like(root_logits)
        logits_vehicle = torch.zeros_like(root_logits)

        animal_classes_index = torch.tensor(animal_classes)
        vehicle_classes_index = torch.tensor(vehicle_classes)

        logits_animal[:, animal_classes_index] = subroot_logits_animal[:,:]
        logits_vehicle[:, vehicle_classes_index] = subroot_logits_vehicle[:,:]

        return root_logits, logits_animal, logits_vehicle

    def load_root_model(self, path):
        """
        Load pre-trained weights for the root model with error handling.
        """
        try:
            checkpoint = torch.load(path)
            self.root_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load root model from {path}: {e}")

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


class LightTreeResNet_Unknown(nn.Module):
    def __init__(self, block, root_num_blocks, subroot_num_blocks, num_classes=10,  device='cpu'):
        super(LightTreeResNet_Unknown, self).__init__()
        self.root_model = LightResnet(block, root_num_blocks, num_classes=num_classes,device=device)  # 10分类 
        self.subroot_animal = LightResnet(block, subroot_num_blocks, num_classes=7,device=device)
        self.subroot_vehicle = LightResnet(block, subroot_num_blocks, num_classes=5,device=device)
    
    def forward(self, x):
        # print x shape
        #print(f"x shape: {x.shape}")
        root_logits = self.root_model(x)
        subroot_logits_animal = self.subroot_animal(x)
        subroot_logits_vehicle = self.subroot_vehicle(x)
        return root_logits, subroot_logits_animal, subroot_logits_vehicle

    def load_root_model(self, path):
        """
        Load pre-trained weights for the root model with error handling.
        """
        try:
            checkpoint = torch.load(path)
            self.root_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load root model from {path}: {e}")

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
                root_num_blocks=[2, 2, 2], 
                subroot_num_blocks=[2, 2, 2], 
                num_classes=num_classes, 
                device=device
            )
        else:
            return LightTreeResNet(
                BasicBlock, 
                root_num_blocks=[2, 2, 2], 
                subroot_num_blocks=[2, 2, 2], 
                num_classes=num_classes, 
                device=device
            )
    
    
    raise ValueError('Only lighttreeresnet20 is supported!')
    return

def create_model(name, normalize, info, device, unknown_classes=False):
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
        backbone = lighttreeresnet(name, num_classes=info['num_classes'], unkown_classes=unknown_classes, device=device)
        if normalize:
            normalization_layer = Normalization(info['mean'], info['std']).to(device)
            backbone.root_model = torch.nn.Sequential(normalization_layer, backbone.root_model)
            backbone.subroot_animal = torch.nn.Sequential(normalization_layer, backbone.subroot_animal)
            backbone.subroot_vehicle = torch.nn.Sequential(normalization_layer, backbone.subroot_vehicle)
        else:
            backbone.root_model = torch.nn.Sequential(backbone.root_model)
            backbone.subroot_animal = torch.nn.Sequential(backbone.subroot_animal)
            backbone.subroot_vehicle = torch.nn.Sequential(backbone.subroot_vehicle)
        return backbone.to(device)
    
    else: 
        backbone = lightresnet(name, num_classes=info['num_classes'], device=device)
        if normalize:
            normalization_layer = Normalization(info['mean'], info['std']).to(device)
            backbone = torch.nn.Sequential(normalization_layer, backbone)
        else:
            backbone = torch.nn.Sequential(backbone)
        return backbone.to(device)