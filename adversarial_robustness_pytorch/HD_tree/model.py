from core.model.resnet import BasicBlock, Bottleneck
from core.model.resnet import Normalization
import torch.nn as nn
import torch.nn.functional as F
import torch

class LightShareResnet(nn.Module):  # Renamed from LightRootResnet to LightShareResnet
    def __init__(self, block, num_blocks,  device='cpu'):
        super(LightShareResnet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
    def __init__(self, block, num_blocks, num_classes, device='cpu'):
        super(LightSubRootResNet, self).__init__()
        self.in_planes = 32

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

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
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits

# samilar to TreeResNet, change the root model to LightShareResnet, and subroot model to LightSubRootResNet
class LightTreeResNet(nn.Module):
    def __init__(self, block, share_num_blocks, subroot_num_blocks, num_classes=10, linear_bias=True, bn_affine=True, device='cpu'):
        super(LightTreeResNet, self).__init__()
        self.share = LightShareResnet(block, share_num_blocks, device=device)  # Renamed from root to share
        self.subroot_coarse = LightSubRootResNet(block, subroot_num_blocks, num_classes=num_classes, device=device)  # 2分类
        self.subroot_animal = LightSubRootResNet(block, subroot_num_blocks, num_classes=6, device=device)
        self.subroot_vehicle = LightSubRootResNet(block, subroot_num_blocks, num_classes=4, device=device)
    def forward(self, x):
        share_features = self.share(x)  # Updated from root to share
        
        logits_coarse = self.subroot_coarse(share_features)  # 10分类
        logits_animal = self.subroot_animal(share_features)  # 6分类
        logits_vehicle = self.subroot_vehicle(share_features)  # 4分类

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
            share_num_blocks=[2, 1],  # Updated from root_num_blocks to share_num_blocks
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
        backbone.share = torch.nn.Sequential(normalization_layer, backbone.share)  # Updated from root to share
        backbone.subroot_coarse = torch.nn.Sequential(normalization_layer, backbone.subroot_coarse)
        backbone.subroot_animal = torch.nn.Sequential(normalization_layer, backbone.subroot_animal)
        backbone.subroot_vehicle = torch.nn.Sequential(normalization_layer, backbone.subroot_vehicle)
    else:
        backbone.share = torch.nn.Sequential(backbone.share)  # Updated from root to share
        backbone.subroot_coarse = torch.nn.Sequential(backbone.subroot_coarse)
        backbone.subroot_animal = torch.nn.Sequential(backbone.subroot_animal)
        backbone.subroot_vehicle = torch.nn.Sequential(backbone.subroot_vehicle)
    return backbone.to(device)


class LightResnet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10, device='cpu'):
        super(LightResnet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
        self.share = [self.layer1, self.layer2[0]]
        self.subroot = [self.layer2[1], self.layer3, self.avg_pool, self.fc]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def lightresnet(name, num_classes=10, pretrained=False, device='cpu'):
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'lightresnet20':
        return LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=device)
     
    raise ValueError('Only lightresnet20 are supported!')