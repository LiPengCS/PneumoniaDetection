import torch
import torch.nn.functional as F
import numpy as np
import utils
import config
from model import MyAlexNet
from PIL import Image
from dataloader import transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time

def get_best_model():
    params = utils.load_params()
    model = MyAlexNet(params).to(device=params.device)
    checkpoint = os.path.join(config.model_dir, 'best.pth.tar')
    utils.load_checkpoint(checkpoint, model, params)
    return model, params

def predict(X, y=None):
    """

    Args:
        X (list): a list of numpy array. (h, w, 3)
        y (list): a list of labels (0 or 1)
    """
    model, params = get_best_model()
    model.eval()
    with torch.no_grad():
        X = torch.cat([transform(Image.fromarray(x)).unsqueeze(0) for x in X])
        output = model(X)
        pre_output = model.pre_output
        prob = F.softmax(output, dim=1)

    prob = prob.data.cpu().numpy()
    pre_output = pre_output.data.cpu().numpy()
    y_pred = np.argmax(prob, axis = 1)
    prob = prob[:, 1]

    if y is not None:
        print("Accuracy:", np.mean(y == y_pred))
    return y_pred, prob, pre_output

# def show(X, y, y_pred, prob):
#     N = len(X)
#     n_row = N // 5 + 1
#     plt.figure(figsize = (10,10))
#     for i in range(N):
#         plt.subplot(n_row, 5, i+1)
#         plt.imshow(np.array(X[i]))
#         plt.axis('off')
#         pred = 'norm' if y_pred[i] == 0 else 'pneu'
#         true = 'norm' if y[i] == 0 else 'pneu'
#         plt.title("P: {} ({:.2%}), T: {}".format(pred, prob[i], true), fontsize=10)
#     plt.tight_layout()
#     plt.show()

# X, y = [], []

# y_pred, prob, pre_output = predict(X, y)
# show(X, y, y_pred, prob)

img_path = "./data/test/pneumonia/0.jpg"

# Occlude Todo
    # input: one image (h, w)
    # resize (224, 224)
    # occlude (size, color) => [(224, 224)]
    # to RGB: X = [(224, 224, 3)]
    # to prob: prob = predict(X) 
    # to heatmap: prob difference between no occlude

# Example:
    # img = np.array(Image.open(img_path))
    # print(img.shape)
    # img[100:200, 100:200] = 1
    # plt.imshow(img)
    # plt.show()


# T-SNE Todo:
    # input: a list of images (N * h * w *3)
# X = []
# y = []

# for i in range(10):
#     img = "./data/test/normal/{}.jpg".format(i)
#     x = np.array(Image.open(img).convert('RGB'))
#     X.append(x)
#     y.append(0)

# for i in range(10):
#     img = "./data/test/pneumonia/{}.jpg".format(i)
#     x = np.array(Image.open(img).convert('RGB'))
#     X.append(x)
#     y.append(1)       

# y_pred, prob, pre_output = predict(X)
# output to t-sne: (N * 4096) = > (N * 2)  (T-sne)
# [(xc1, yc1), (xc2, yc2) ... (xcN, ycN)]
# [y1, y2, ....yN]










