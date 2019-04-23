import torch
import torch.nn.functional as F
import numpy as np
import utils
import config
from model import MyAlexNet, MyAlexNetCAM
from PIL import Image
from dataloader import transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time

def get_best_model(return_weights=False):
    params = utils.load_params()
    model = MyAlexNet(params).to(device=params.device)
    checkpoint = os.path.join(config.model_dir, 'last.pth.tar')
    utils.load_checkpoint(checkpoint, model, params)
    if return_weights:
        weights = {
            "conv1":model.conv1[0].weight.data,
            "conv2":model.conv2[0].weight.data,
            "conv3":model.conv3[0].weight.data,
            "conv4":model.conv4[0].weight.data,
            "conv5":model.conv5[0].weight.data,
        }
        return weights
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
        prob = F.softmax(output, dim=1)

        prob = prob.data.cpu().numpy()
        conv1 = model.conv1_out.cpu().numpy()
        conv2 = model.conv2_out.cpu().numpy()
        conv3 = model.conv3_out.cpu().numpy()
        conv4 = model.conv4_out.cpu().numpy()
        conv5 = model.conv5_out.cpu().numpy()
        fc1 = model.fc1_out.cpu()
        fc2 = model.fc2_out.cpu()

    y_pred = np.argmax(prob, axis = 1)
    prob = prob[:, 1]
    activation = {
                "conv1": conv1,
                "conv2": conv2,
                "conv3": conv3,
                "conv4": conv4,
                "conv5": conv5,
                "fc1": fc1,
                "fc2": fc2
                }

    if y is not None:
        print("Accuracy:", np.mean(y == y_pred))
    return y_pred, prob, activation

def predict_CAM(X, y=None):
    """
    Args:
        X (list): a list of chest x-rays in numpy array. (h, w, 1)
        y (list): a list of labels (0 or 1)
    """
    params = utils.load_params()
    model = MyAlexNetCAM(params).to(device=params.device)
    checkpoint = os.path.join(config.cam_model_dir, 'last.pth.tar')
    utils.load_checkpoint(checkpoint, model, params)
    model.eval()
    with torch.no_grad():
        X = utils.preprocess(X)
        output = model(X)
        prob = F.softmax(output, dim=1)

        prob = prob.data.cpu().numpy()
        conv5 = model.conv5_out.cpu().numpy()
        weights = model.fc.weight.detach().numpy()

    y_pred = np.argmax(prob, axis = 1)
    w = weights[y_pred].reshape(1, -1, 1, 1)
    cam = np.sum((conv5 * w), axis=1)

    # conv5 = conv5[0]
    # cam = np.zeros(shape=conv5.shape[1:3])
    # weight = weights[y_pred, :].reshape()
    # for i, w in enumerate(weight):
    #     cam += w * conv5[i, :, :]

    if y is not None:
        print("Accuracy:", np.mean(y == y_pred))
    return cam
