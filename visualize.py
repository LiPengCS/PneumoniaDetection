from predict import predict, get_best_model, predict_CAM
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import utils
from skimage import transform
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import pickle
from torchvision.utils import make_grid

def occlude(img, label=None, n_row=8, n_col=8, occ_color=0, show_example=False):
    """ Generate heatmap by occluding
    
    Args:
        img (np.array): image (224 * 224 * 1)
    """
    if label is not None: 
        label = [label]
    y_pred, prob, pre_output = predict([img], label)
    img_h, img_w = img.shape[0], img.shape[1]
    occ_h, occ_w = img_h // n_row, img_w // n_col
    occ_imgs = []

    for row in range(n_row):
        for col in range(n_col):
            new_img = img.copy()
            new_img[row*occ_w:(row+1)*occ_w, col*occ_h:(col+1)*occ_h] = occ_color
            occ_imgs.append(new_img)

    _, probs, _ = predict(occ_imgs)
    diff = np.abs(probs.reshape(n_row, n_col) - prob)
    result = show_heatmap(diff, img)
    return result

def silency_map(img, y):
    """ Generate silency map"""
    X = utils.preprocess([img])
    y = torch.LongTensor(y)
    model, params = get_best_model()
    model.eval()
    X.requires_grad_()
    scores = model(X)
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
    correct_scores.backward(torch.FloatTensor([1, 1]))
    grads = X.grad.data
    saliency, indices = torch.max(grads.abs(), dim = 1)
    result = show_heatmap(saliency.numpy()[0], img)
    return result

class tsne(object):
    """docstring for tsne"""
    def __init__(self, norm_images, pneu_images):
        super(tsne, self).__init__()
        self.n_norm = len(norm_images)
        X = norm_images + pneu_images
        self.n_embed = len(X)
        y_pred, prob, activation = predict(X)
        self.embed = activation["fc2"]

    def transform(self, x=None):
        if x is not None:
            y_pred, prob, activation = predict(x)
            x_embed = activation["fc2"]
            embed = np.concatenate([self.embed, x_embed], axis=0)
        else:
            embed = self.embed
        tsne_embed = TSNE(n_components=2, random_state=33).fit_transform(embed)
        norm_tsne = tsne_embed[:self.n_norm]
        pneu_tsne = tsne_embed[self.n_norm:self.n_embed]
        x_tsne = tsne_embed[self.n_embed:]
        return norm_tsne, pneu_tsne, x_tsne

    def show(self, x=None):
        fig = plt.figure()
        norm_tsne, pneu_tsne, x_tsne = self.transform(x)
        plt.scatter(norm_tsne[:, 0], norm_tsne[:, 1], label="normal", color='b', s=10, alpha=.55)
        plt.scatter(pneu_tsne[:, 0], pneu_tsne[:, 1], label="pneumonia", color='C1', s=10, alpha=.55)
        if len(x_tsne) > 0:
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1], label="user", color='red', s=25)
        plt.legend(fontsize=14)
        return fig_to_np(fig)

def show_heatmap(heatmap, image):
    fig = plt.figure()
    img_h, img_w = image.shape[0:2]
    heatmap = transform.resize(heatmap, (img_h, img_w), mode='constant', anti_aliasing=True)
    ax = plt.imshow(image, cmap='gray')
    im = plt.imshow(heatmap, alpha=0.5)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.colorbar(im)
    return fig_to_np(fig)

def get_images(img_type, size):
    paths = ["./data/test/{}/{}.jpg".format(img_type, i) for i in range(size)]
    images = [utils.load_image(img_path) for img_path in paths]
    return images

def sort_activation(a):
    n = a.shape[0]
    a_reshape = a.reshape(n, -1)
    a_sum = a_reshape.sum(axis=1)
    indices = np.argsort(a_sum)[::-1]
    sort_a = np.array([a[i] for i in indices])
    return sort_a

def plot_activation(activation, n=10):
    fig = plt.figure()
    count = 1

    for l in range(5):
        a = activation['conv{}'.format(l+1)][0]
        a = sort_activation(a)

        for i in range(n):
            plt.subplot(n, 5,  5*i + l + 1)
            ax = plt.imshow(a[i])

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    return fig_to_np(fig)

def plot_kernel(weights, n=10):
    count = 1
    figs = []
    for l in range(5):
        fig = plt.figure()
        w = weights['conv{}'.format(l+1)]
        n_c = w.shape[1]
        if n_c != 3:
            w = w[:, 0:1, :, :]
        nrow = int(np.sqrt(w.shape[0]))
        img = make_grid(w, nrow=nrow, normalize=True, pad_value=1, padding=1)
        ax = plt.imshow(img.permute(1, 2, 0))

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        count += 1

        figs.append(fig_to_np(fig))
    return figs

def plot_conv1(activation):
    a = activation['conv1'][0]
    plt.figure()
    a = sort_activation(a)
    for i in range(64):
        plt.subplot(8, 8, i+1)
        fig = plt.imshow(a[i])

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

def fig_to_np(fig):
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    return X

def tsne_init():
    norm_images = get_images("normal", 100)
    pneu_images = get_images("pneumonia", 100)
    t = tsne(norm_images, pneu_images)
    pickle.dump(t, open("./tsne.obj", 'wb'))

def cam(img, y=None):
    cam_value = predict_CAM([img], y)
    cam_value = cam_value - np.min(cam_value)
    cam_value = cam_value / np.max(cam_value)
    cam_value = np.uint8(255 * cam_value)
    return show_heatmap(cam_value[0], img)

def visualize(img, y=None):
    y_pred, prob, activation = predict([img], y)

    # occlude
    occ_result = occlude(img)

    # # silency map
    silency_result = silency_map(img, y_pred)

    # CAM
    cam_result = cam(img, y)

    # tsne
    norm_images = get_images("normal", 100)
    pneu_images = get_images("pneumonia", 100)
    t = tsne(norm_images, pneu_images)
    tsne_result = t.show([img])

    # activation map
    activation_result = plot_activation(activation)

    # kernel
    weights = get_best_model(return_weights=True)
    kernel_result = plot_kernel(weights)

    visualize_result = {
        "occlude": occ_result,
        "silency_map": silency_result,
        "tsne": tsne_result,
        "activation": activation_result,
        "kernel": kernel_result,
        "cam": cam_result
    }

    return y_pred, prob, visualize_result
