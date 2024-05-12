import torch.nn as nn
from model.MLP import MLP
from model.resnet import resnet18

# resnet18 model of FAPT and FATEL
class Resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet18, self).__init__()

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
