from core.model.resnet import BasicBlock, Bottleneck
from core.model.resnet import Normalization
import torch.nn as nn
import torch.nn.functional as F
import torch
animal_classes = [ 2, 3, 4, 5, 6, 7]  # 6种动物
vehicle_classes = [0, 1, 8, 9]  # 4种交通工具

class LightRootResnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, linear_bias=True, bn_affine=True, device='cpu'):
        super(LightRootResnet, self).__init__()
        self.in_planes = 16

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, affine=self.bn_affine)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape) # (m, 3, 32, 32)
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape) # (m, 16, 32, 32)
        out = self.layer1(out)
        #print(out.shape) # (m, 16, 32, 32)
        feature_map = self.layer2(out)
        #print(feature_map.shape) # (m, 32, 16, 16)
        out = F.avg_pool2d(feature_map, 4)
        #print(out.shape) # (m, 32, 4, 4)
        out = out.view(out.size(0), -1)
        #print(out.shape) # (m, 512)
       
        #print(logits.shape)
        return feature_map

class LightSubRootResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, linear_bias=True, bn_affine=True, device='cpu'):
        super(LightSubRootResNet, self).__init__()
        self.in_planes = 32

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.linear = nn.Linear(64 * block.expansion * 4, num_classes, bias=self.linear_bias)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits

# samilar to TreeResNet, change the root model to LightRootResnet, and subroot model to LightSubRootResNet
class LightTreeResNet(nn.Module):
    def __init__(self, block, root_num_blocks, subroot_num_blocks, num_classes=10, linear_bias=True, bn_affine=True, device='cpu'):
        super(LightTreeResNet, self).__init__()
        self.root_model = LightRootResnet(block, root_num_blocks, num_classes=num_classes,linear_bias=linear_bias,bn_affine=bn_affine, device=device)  # 10分类 
        self.subroot_coarse = LightSubRootResNet(block, subroot_num_blocks, num_classes=num_classes,linear_bias=linear_bias,bn_affine=bn_affine, device=device)  # 2分类
        self.subroot_animal = LightSubRootResNet(block, subroot_num_blocks, num_classes=6,linear_bias=linear_bias,bn_affine=bn_affine, device=device)
        self.subroot_vehicle = LightSubRootResNet(block, subroot_num_blocks, num_classes=4,linear_bias=linear_bias,bn_affine=bn_affine, device=device)
    def forward(self, x):
        root_features = self.root_model(x)
        logits_coarse = self.subroot_coarse(root_features)  # 10分类
        subroot_logits_animal = self.subroot_animal(root_features)
        subroot_logits_vehicle = self.subroot_vehicle(root_features)

        logits_animal = torch.zeros_like(logits_coarse)
        logits_vehicle = torch.zeros_like(logits_coarse)

        animal_classes_index = torch.tensor(animal_classes)
        vehicle_classes_index = torch.tensor(vehicle_classes)

        logits_animal[:, animal_classes_index] = subroot_logits_animal[:,:]
        logits_vehicle[:, vehicle_classes_index] = subroot_logits_vehicle[:,:]

        return logits_coarse, logits_animal, logits_vehicle


def lighttreeresnet(name, num_classes=2, pretrained=False, device='cpu'):
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
        return LightTreeResNet(
            BasicBlock, 
            root_num_blocks=[2, 1], 
            subroot_num_blocks=[1, 2], 
            num_classes=num_classes, 
            device=device
        )
    
    raise ValueError('Only lighttreeresnet20 is supported!')
    return

def create_model(name, normalize, info, device):
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
    
    backbone = lighttreeresnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
    if normalize:
        normalization_layer = Normalization(info['mean'], info['std']).to(device)
        backbone.root_model = torch.nn.Sequential(normalization_layer, backbone.root_model)
        backbone.subroot_coarse = torch.nn.Sequential(normalization_layer, backbone.subroot_coarse)
        backbone.subroot_animal = torch.nn.Sequential(normalization_layer, backbone.subroot_animal)
        backbone.subroot_vehicle = torch.nn.Sequential(normalization_layer, backbone.subroot_vehicle)
    else:
        backbone.root_model = torch.nn.Sequential(backbone.root_model)
        backbone.subroot_coarse = torch.nn.Sequential(backbone.subroot_coarse)
        backbone.subroot_animal = torch.nn.Sequential(backbone.subroot_animal)
        backbone.subroot_vehicle = torch.nn.Sequential(backbone.subroot_vehicle)
    return backbone.to(device)