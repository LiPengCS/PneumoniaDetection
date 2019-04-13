import torchvision.models as models
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
import torch
import torch.nn.functional as F

def flatten(x): 
    return x.view(x.size(0), -1)

class MyAlexNet(nn.Module):
    def __init__(self, params):
        super(MyAlexNet, self).__init__()
        self.params = params
        pretrained = models.alexnet(pretrained=True)

        self.conv1 = pretrained.features[0:3]
        self.conv2 = pretrained.features[3:6]
        self.conv3 = pretrained.features[6:8]
        self.conv4 = pretrained.features[8:10]
        self.conv5 = pretrained.features[10:]

        self.fc1 = pretrained.classifier[0:4]
        self.fc2 = pretrained.classifier[4:6]
        self.fc3 = nn.Linear(4096, params.num_classes)

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        for param in self.conv5.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.conv1_out = self.conv1(x)
        self.conv2_out = self.conv2(self.conv1_out)
        self.conv3_out = self.conv3(self.conv2_out)
        self.conv4_out = self.conv4(self.conv3_out)
        self.conv5_out = self.conv5(self.conv4_out)
        
        self.flat = flatten(self.conv5_out)
        self.fc1_out = self.fc1(self.flat)
        self.fc2_out = self.fc2(self.fc1_out)
        self.fc3_out = self.fc3(self.fc2_out)

        return self.fc3_out

loss_fn = nn.CrossEntropyLoss()

def metric(y_score, y_true, test_mode=False):
    y_pred = np.argmax(y_score, axis=1)
    acc = np.mean(y_true == y_pred)
    if not test_mode:
        return acc
    else:
        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        auc = roc_auc_score(y_true, y_score[:, 1])
        return {"acc": acc, "f1": f1, "cm":cm, "auc": auc, "fpr":fpr, "tpr": tpr}


class MyAlexNetCAM(nn.Module):
    def __init__(self, params):
        super(MyAlexNetCAM, self).__init__()
        self.params = params
        pretrained = models.alexnet(pretrained=True)

        self.conv1 = pretrained.features[0:3]
        self.conv2 = pretrained.features[3:6]
        self.conv3 = pretrained.features[6:8]
        self.conv4 = pretrained.features[8:10]
        self.conv5 = pretrained.features[10:12]

        self.fc = nn.Linear(256, params.num_classes, bias=False)

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        for param in self.conv5.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.conv1_out = self.conv1(x)
        self.conv2_out = self.conv2(self.conv1_out)
        self.conv3_out = self.conv3(self.conv2_out)
        self.conv4_out = self.conv4(self.conv3_out)
        self.conv5_out = self.conv5(self.conv4_out)

        self.gap_out = torch.mean(self.conv5_out.view(self.conv5_out.size(0), self.conv5_out.size(1), -1), dim=2)
        self.fc_out = self.fc(self.gap_out)

        return self.fc_out