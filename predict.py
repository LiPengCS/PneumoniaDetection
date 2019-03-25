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
        X (list): a list of chest x-rays in numpy array. (h, w, 1)
        y (list): a list of labels (0 or 1)
    """
    model, params = get_best_model()
    model.eval()
    with torch.no_grad():
        X = utils.preprocess(X)
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