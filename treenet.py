"""
    FOR TREE MODEL 
        - BASE RESNET 18/34
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

animal_classes = [2, 3, 4, 5, 6, 7]
vehicle_classes = [0, 1, 8, 9]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, linear_bias=True, bn_affine=True):
        super(BasicBlock, self).__init__()

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=self.bn_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, linear_bias=True, bn_affine=True):
        super(Bottleneck, self).__init__()

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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

    
class TreeModel_34(TreeResNet): # 【3，4，6，3】 resnet34， 【3，4，3】 treeresnet34 +【3】；
    def __init__(self, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True, **kwargs):
        super(TreeModel_34, self).__init__( 
            BasicBlock, [3, 4, 3],[3,3], 
            num_channels=num_channels, num_classes=num_classes,
            linear_bias=linear_bias, bn_affine=bn_affine
        )

class TreeModel_18(TreeResNet): # 【2, 2, 2, 2】 resnet18, 【2, 2, 1】 treeresnet18 +【1】；
    def __init__(self, num_channels=3, num_classes=10, linear_bias=True, bn_affine=True, **kwargs):
        super(TreeModel_18, self).__init__( 
            BasicBlock, [2,2,1], [1,1],
            num_channels=num_channels, num_classes=num_classes,
            linear_bias=linear_bias, bn_affine=bn_affine
        )

