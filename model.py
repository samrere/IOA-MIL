import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 512),
            nn.ReLU(),
        )
    def forward(self, x): # B,D,C,H,W (B=1)
        x = x.squeeze(0) # DCHW
        x = self.feature_extractor_part1(x) # D,50,4,4
        x = x.view(-1, 50 * 4 * 4) # D,800
        x = self.feature_extractor_part2(x) # D,512
        return x
        
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.feature_extractor = Feature_extractor()
        self.attention = nn.Linear(512, 1)
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x, criterion=None,labels=None):
        x = self.feature_extractor(x) # D,512
        raw = self.attention(x)# D,1
        raw = torch.transpose(raw, 1, 0)# 1,D
        
        A = F.softmax(raw, dim=1)# 1,D
        M = A @ x # 1,D @ D,512->1,512
        
        x = self.fc(M) # 1,1
        x = x.squeeze(-1)# 1,
        L1 = criterion(x, labels).mean()
        loss = L1
        
        return x, loss
#

class Attention_instance_loss(nn.Module):
    def __init__(self):
        super(Attention_instance_loss, self).__init__()
        self.feature_extractor = Feature_extractor()
        self.attention = nn.Linear(512, 1)
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x, criterion=None,labels=None):
        D = x.shape[1]
        x = self.feature_extractor(x) # D,512
        raw = self.attention(x)# D,1
        raw = torch.transpose(raw, 1, 0)# 1,D
        
        prob = 5*torch.tanh(raw/5)# 1,D
        A = F.softmax(prob, dim=1)# 1,D
        M = A @ x # 1,D @ D,512->1,512

        M = M.detach()
        skip = self.fc(M)
        x = self.attention(M)
        x = x.detach()
        x = x + skip
        x = x.squeeze(-1)# 1,
        L1 = criterion(x, labels).mean()
        L3 = A.squeeze(0) * criterion(raw.squeeze(0), labels.repeat(D))
        L3 = L3.mean()
        loss = L1+L3
        
        return x, loss