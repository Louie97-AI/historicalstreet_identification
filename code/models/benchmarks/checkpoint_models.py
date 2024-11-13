import torchvision.models as models
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.init as init
from transformers import ViTModel, ViTConfig
import timm



class QBresNet50(nn.Module):
    def __init__(self):
        super(QBresNet50,self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        resnet50.fc = nn.Sequential(nn.Linear(2048,512),
                                    nn.ReLU(),
                                    nn.Linear(512,2))
                                    
        self.resnet50 = resnet50
    
    def forward(self,x):
        
        x = self.resnet50(x)
        
        return x


class QBresnext50(nn.Module):
    def __init__(self) -> None:
        super(QBresnext50,self).__init__()
        resnext50 = models.resnext50_32x4d(pretrained=False)
        resnext50.fc = nn.Sequential(nn.Linear(2048,512),
                            nn.ReLU(),
                            nn.Linear(512,2))
        
        self.resnext50 = resnext50

    def forward(self,x):
        
        x = self.resnext50(x)
        
        return x


class QBVGG19(nn.Module):
    def __init__(self):
        super(QBVGG19,self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        self.vgg19 =  vgg19
    
    def forward(self,x):
        
        x = self.vgg19(x)
        
        return x



class VIT(nn.Module):
    def __init__(self) -> None:
        super(VIT,self).__init__()
        config = ViTConfig()
        self.vit = ViTModel(config)
        # self.relu = nn.ReLU()
        self.classhead = nn.Sequential(nn.Linear(768,256),
                                       nn.GELU(),
                                       nn.Linear(256,2))
    def forward(self,x):
        x = self.vit(x)
        x = x.last_hidden_state[:,0,:]
        x = self.classhead(x)
        
        return x


class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer,self).__init__()
        model_name = 'swin_base_patch4_window7_224'
        model = timm.create_model(model_name, pretrained=False)
        model.head.fc1 = nn.Linear(in_features=1024, out_features=256)
        model.head.fc2 = nn.Linear(in_features=256, out_features=2)

        self.swintransformer = model
        
    def forward(self,x):
        x = self.swintransformer(x)
        return x


