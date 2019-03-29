import torchvision.models as models
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score

class MyAlexNet(nn.Module):
    def __init__(self, params):
        super(MyAlexNet, self).__init__()
        self.params = params
        self.pre_layers = models.alexnet(pretrained=True)

        # delete last layer
        self.pre_layers.classifier = self.pre_layers.classifier[:-1]

        for param in self.pre_layers.parameters():
                param.requires_grad = False
        self.my_layer = nn.Linear(4096, params.num_classes)

    def forward(self, x):
        self.pre_output = self.pre_layers(x)
        output = self.my_layer(self.pre_output)
        return output

loss_fn = nn.CrossEntropyLoss()

def metric(y_score, y_true, test_mode=False):
    y_pred = np.argmax(output, axis=1)
    acc = np.mean(y == y_pred)
    if not test_mode:
        return acc
    else:
        cm = confusion_matrix(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        roc = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        return {"acc": acc, "f1": f1, "roc":roc, "auc": auc}

    