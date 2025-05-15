"""
    FOR TREE MODEL 
        - BASE RESNET 34
"""
from .resnet import BasicBlock, Bottleneck
import torch.nn as nn
import torch.nn.functional as F
import torch

from .. import utils

from lightning import animal_classes, vehicle_classes

class RootResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True):
        super(RootResNet, self).__init__()
        self.in_planes = 64

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=self.bn_affine)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256 * block.expansion * 4, num_classes, bias=self.linear_bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.linear_bias, self.bn_affine))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        #print(out.shape) # (m, 64, 32, 32)
        out = self.layer2(out)
        #print(out.shape) # (m, 128, 16, 16)
        feature_map = self.layer3(out)
        #print (feature_map.shape) # (m, 256, 8, 8)
        out = F.avg_pool2d(feature_map, 4)
        #print(out.shape) # (m, 256, 2, 2)
        out = out.view(out.size(0), -1)
        #print(out.shape) # (m, 1024)
        logits = self.linear(out) 
        #print(logits.shape) 
        return logits, feature_map

class SubRootResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, linear_bias=True, bn_affine=True):
        super(SubRootResNet, self).__init__()
        self.in_planes = 256

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine
        self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=self.linear_bias)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.linear_bias, self.bn_affine))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        #print(x.shape) # (m, 256, 8, 8)
        out = self.layer1(x)
        #print(out.shape) # (m, 256, 8, 8)
        out = self.layer2(out)
        #print(out.shape) # (m, 512, 4, 4)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits

class TreeResNet(nn.Module):
    def __init__(self, block, root_num_blocks, subroot_num_blocks, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True):
        super(TreeResNet, self).__init__()
        self.root_model = RootResNet(block, root_num_blocks, num_channels=num_channels, num_classes=num_classes,linear_bias=linear_bias,bn_affine=bn_affine)  # 10分类 
        self.subroot_animal = SubRootResNet(block, subroot_num_blocks, num_classes=7,linear_bias=linear_bias,bn_affine=bn_affine)  # 6种动物 + 1 none of them
        self.subroot_vehicle = SubRootResNet(block, subroot_num_blocks, num_classes=5,linear_bias=linear_bias,bn_affine=bn_affine)  # 4种交通工具 + 1 none of them 

    def forward(self, x):
        root_logits, root_features = self.root_model(x)
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
            subroot_animal_logits = self.subroot_animal(root_features[animal_rows])
            subroot_logits[animal_rows[:, None], animal_classes_index] = subroot_animal_logits[:, :-1]
            unknown_value = subroot_animal_logits[:, -1] / len(vehicle_classes)
            subroot_logits[animal_rows[:, None], vehicle_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(vehicle_classes))

        # Fix for vehicle subroot logits
        if is_vehicle.any():
            vehicle_rows = is_vehicle.nonzero(as_tuple=True)[0]
            subroot_vehicle_logits = self.subroot_vehicle(root_features[vehicle_rows])
            subroot_logits[vehicle_rows[:, None], vehicle_classes_index] = subroot_vehicle_logits[:, :-1]
            unknown_value = subroot_vehicle_logits[:, -1] / len(animal_classes)
            subroot_logits[vehicle_rows[:, None], animal_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(animal_classes))
            
            return root_logits, subroot_logits

    
@utils.register_model(dataset='cifar10', name='treeresnet34')
@utils.register_model(dataset='cifar100', name='treeresnet34')
class TreeResNet34(TreeResNet): # 【3，4，6，3】 resnet34， 【3，4，3】 treeresnet34 +【3】；
    def __init__(self, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True, **kwargs):
        super(TreeResNet34, self).__init__( 
            BasicBlock, [3, 4, 3],[3,3], 
            num_channels=num_channels, num_classes=num_classes,
            linear_bias=linear_bias, bn_affine=bn_affine
        )

@utils.register_model(dataset='cifar10', name='treeresnet18')
@utils.register_model(dataset='cifar100', name='treeresnet18')
class TreeResNet18(TreeResNet): # 【2, 2, 2, 2】 resnet18, 【2, 2, 1】 treeresnet18 +【1】；
    def __init__(self, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True, **kwargs):
        super(TreeResNet18, self).__init__( 
            BasicBlock, [2,2,1], [1,1],
            num_channels=num_channels, num_classes=num_classes,
            linear_bias=linear_bias, bn_affine=bn_affine
        )

# ---------------------------light resnet--------------------------------

class LightRootResnet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True):
        super(LightRootResnet, self).__init__()
        self.in_planes = 16

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, affine=self.bn_affine)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.linear = nn.Linear(32 * block.expansion * 16, num_classes, bias=self.linear_bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.linear_bias, self.bn_affine))
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
        logits = self.linear(out)
        #print(logits.shape)
        return logits ,feature_map

class LightSubRootResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, linear_bias=True, bn_affine=True):
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
            layers.append(block(self.in_planes, planes, stride, self.linear_bias, self.bn_affine))
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
    def __init__(self, block, root_num_blocks, subroot_num_blocks, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True):
        super(LightTreeResNet, self).__init__()
        self.root_model = LightRootResnet(block, root_num_blocks, num_channels=num_channels, num_classes=num_classes,linear_bias=linear_bias,bn_affine=bn_affine)  # 10分类 
        self.subroot_animal = LightSubRootResNet(block, subroot_num_blocks, num_classes=7,linear_bias=linear_bias,bn_affine=bn_affine)  # 6种动物 + 1 none of them
        self.subroot_vehicle = LightSubRootResNet(block, subroot_num_blocks, num_classes=5,linear_bias=linear_bias,bn_affine=bn_affine)  # 4种交通工具 + 1 none of them 

    def forward(self, x):
        root_logits, root_features = self.root_model(x)
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
            subroot_animal_logits = self.subroot_animal(root_features[animal_rows])
            subroot_logits[animal_rows[:, None], animal_classes_index] = subroot_animal_logits[:, :-1]
            unknown_value = subroot_animal_logits[:, -1] / len(vehicle_classes)
            subroot_logits[animal_rows[:, None], vehicle_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(vehicle_classes))

        # Fix for vehicle subroot logits
        if is_vehicle.any():
            vehicle_rows = is_vehicle.nonzero(as_tuple=True)[0]
            subroot_vehicle_logits = self.subroot_vehicle(root_features[vehicle_rows])
            subroot_logits[vehicle_rows[:, None], vehicle_classes_index] = subroot_vehicle_logits[:, :-1]
            unknown_value = subroot_vehicle_logits[:, -1] / len(animal_classes)
            subroot_logits[vehicle_rows[:, None], animal_classes_index] = unknown_value.unsqueeze(1).expand(-1, len(animal_classes))
            
            return root_logits, subroot_logits

@utils.register_model(dataset='cifar10', name='lighttreeresnet20')
@utils.register_model(dataset='cifar100', name='lighttreeresnet20')
class LightTreeResNet20(LightTreeResNet):
    def __init__(self, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True, **kwargs):
        super(LightTreeResNet20, self).__init__( 
            BasicBlock, [2, 1], [1, 1], 
            num_channels=num_channels, num_classes=num_classes,
            linear_bias=linear_bias, bn_affine=bn_affine
        )
