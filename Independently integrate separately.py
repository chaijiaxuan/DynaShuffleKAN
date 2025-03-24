import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import time

class ShuffleNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        original = shufflenet_v2_x1_0(pretrained=True)
        

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1),              
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True))
        self.stage2 = original.features[3]  
        self.stage3 = original.features[7]  
        self.stage4 = original.features[11] 

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage2(x)  
        f2 = self.stage3(f1) 
        f3 = self.stage4(f2)
        return [f1, f2, f3]

class EnhancedShuffleNetKAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ShuffleNetFeatureExtractor()
        

        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(116, 256, 1),
            nn.Conv2d(232, 256, 1),
            nn.Conv2d(464, 256, 1)
        ])
        

        self.fusion = DynamicWeightedFeatureFusion(
            channels=256, 
            num_features=3
        )
        

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            KANLayer(256, 128),
            nn.Dropout(0.3),
            KANLayer(128, 42)  # 42
        )

    def forward(self, x):
        features = self.backbone(x)
        adjusted = [conv(feat) for conv, feat in zip(self.channel_adjust, features)]
        fused = self.fusion(adjusted)
        return self.classifier(fused)

class DynamicWeightedFeatureFusion(nn.Module):
    def __init__(self, channels, num_features=3, reduction=16):
        super().__init__()
        self.num_features = num_features
        self.channels = channels
        

        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_features)
        ])
        

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        

        self.weight_generator = nn.Sequential(
            nn.Conv2d(num_features*channels, num_features*channels, 3, 
                     padding=1, groups=num_features*channels, bias=False),
            nn.Conv2d(num_features*channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, num_features, 1),
            nn.Softmax(dim=1)
        )
        

        self.shortcut = nn.Conv2d(channels, channels, 1)

    def forward(self, features):
        aligned_features = []
        for i in range(self.num_features):
            feat = features[i]
            if i != 1:  # 以中间特征为基
                target_size = features[1].shape[2:]
                feat = F.interpolate(feat, target_size, mode='bilinear', align_corners=False)
            aligned_feat = self.align_convs[i](feat)
            aligned_features.append(aligned_feat)
        
        channel_weights = [self.channel_attention(f) for f in aligned_features]
        concatenated = torch.cat(aligned_features, dim=1)
        spatial_weights = self.weight_generator(concatenated)

        weighted_features = []
        for i in range(self.num_features):
            att_feat = aligned_features[i] * channel_weights[i]
            weighted_features.append(att_feat * spatial_weights[:, i:i+1])
        
        fused_feature = torch.sum(torch.stack(weighted_features), dim=0)
        return fused_feature + self.shortcut(aligned_features[1])


class KANLayerO(nn.Module):
    def __init__(self, input_dim, output_dim, num_basis=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        
        self.basis_weights = nn.Parameter(torch.randn(output_dim, input_dim, num_basis))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        nn.init.kaiming_normal_(self.basis_weights, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x_pows = x.unsqueeze(-1) ** torch.arange(self.num_basis, device=x.device)
        output = torch.einsum('bik,oik->bo', x_pows, self.basis_weights)
        return output + self.bias.unsqueeze(0)


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_basis=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.base_weights = nn.Parameter(torch.randn(input_dim, output_dim, num_basis))
        self.spline_coeff = nn.Parameter(torch.randn(input_dim, output_dim, num_basis))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_normal_(self.base_weights)
        nn.init.xavier_normal_(self.spline_coeff)

    def forward(self, x):

        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim, 1, 1)  
        base_weights = self.base_weights.unsqueeze(0)  
        spline_coeff = self.spline_coeff.unsqueeze(0)   
        bases = torch.sigmoid(base_weights * x + spline_coeff)  
        weighted_bases = bases * F.softmax(spline_coeff, dim=-1)
        output = torch.sum(weighted_bases, dim=(1, 3))  
        return output + self.bias.unsqueeze(0)


class EnhancedCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(64, 256, 1),   
            nn.Conv2d(128, 256, 1), 
            nn.Conv2d(256, 256, 1)   
        ])
        

        self.fusion = DynamicWeightedFeatureFusion(256, 3)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            KANLayer(256, 128),
            nn.Dropout(0.5),
            KANLayer(128, 42)
        )

    def forward(self, x):

        x1 = self.conv1(x)  
        x2 = self.conv2(x1) 
        x3 = self.conv3(x2)  
        features = [
            self.channel_adjust[0](x1), 
            self.channel_adjust[1](x2),  
            self.channel_adjust[2](x3)   
        ]
        fused = self.fusion(features) 
        return self.classifier(fused)


class CIFARNetWithShuffleKAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ShuffleNetFeatureExtractor()
        self.channel_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(116, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(232, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(464, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.fusion = DynamicWeightedFeatureFusion(256, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.kan = KANLayer(256, 42, num_basis=5)

    def forward(self, x):

        features = self.backbone(x)
        adjusted = [conv(feat) for conv, feat in zip(self.channel_adjust, features)]
        fused = self.fusion(adjusted)
        out = self.avgpool(fused)
        out = out.view(out.size(0), -1)
        return self.kan(out)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10('./data', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
    return total_loss/len(loader.dataset), 100.*correct/len(loader.dataset)
def test(model, device, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
    return total_loss/len(loader.dataset), 100.*correct/len(loader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCIFARNet().to(device)
print(model)
num_params = sum(p.numel() for p in model.parameters())
print(f'参数总量: {num_params}')
dummy_input = torch.randn(1, 3, 32, 32).to(device)


torch.onnx.export(
    model,                       
    dummy_input,                 
    "EnhancedCIFARNet.onnx",     
    export_params=True,          
    opset_version=13,            
    do_constant_folding=True,    
    input_names=["input"],       
    output_names=["output"],     
    dynamic_axes={               
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("模型已导出为 EnhancedCIFARNet.onnx")


def validate_onnx_model():
    import onnx
    from onnx import shape_inference
    

    onnx_model = onnx.load("EnhancedCIFARNet.onnx")
    

    inferred_model = shape_inference.infer_shapes(onnx_model)
    

    onnx.checker.check_model(inferred_model)
    

    print("\nONNX模型输入输出信息：")
    print(f"输入形状：{inferred_model.graph.input[0].type.tensor_type.shape}")
    print(f"输出形状：{inferred_model.graph.output[0].type.tensor_type.shape}")
    print("ONNX模型验证通过")

validate_onnx_model()