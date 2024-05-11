import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MLP import MLP
from model.resnet import resnet18

class resnet18_model(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet18_model, self).__init__()

        self.encoder = resnet18()
        self.fc = nn.Linear(512, num_classes)
        self.mlp = MLP(512, 128, 2)
    
    def forward(self, x_ori, x_aug1=None, x_aug2=None, eval_only=False):
        feature_ori = self.encoder(x_ori)
        logits_ori = self.fc(feature_ori)
        
        if eval_only:
            return logits_ori
        
        lowdim_feature_ori = self.mlp(feature_ori)
        
        feature_aug1 = self.encoder(x_aug1)
        lowdim_feature_aug1 = self.mlp(feature_aug1)

        feature_aug2 = self.encoder(x_aug2)
        lowdim_feature_aug2 = self.mlp(feature_aug2)

        return lowdim_feature_ori, lowdim_feature_aug1, lowdim_feature_aug2, logits_ori
    
    def forward_encoder(self, x):
        out = self.encoder(x)
        return out
    
    def pretraining_head(self, x):
        out = self.mlp(x)
        return out
    
    def classification_head(self, x):
        out = self.fc(x)
        return out
