from predict import predict, get_best_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import utils
from skimage import transform
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def occlude(orig_img, orig_prob, n_row=8, n_col=8, occ_color=0):
    """ Generate heatmap by occluding
    
    Args:
        orig_img (np.array): image (224 * 224 * 1)

    """
    img_h, img_w = orig_img.shape
    occ_h, occ_w = img_h // n_row, img_w // n_col
    occ_imgs = []

    for row in range(n_row):
        for col in range(n_col):
            new_img = orig_img.copy()
            new_img[row*occ_w:(row+1)*occ_w, col*occ_h:(col+1)*occ_h] = occ_color
            occ_imgs.append(new_img)

    _, probs, _ = predict(occ_imgs)
    diff = probs.reshape(n_row, n_col) - orig_prob
    # plt.figure()
    # for i, occ in enumerate(occ_imgs):
    #     plt.subplot(n_row, n_col, i+1)
    #     plt.imshow(occ, cmap='gray')
    return diff

def silency_map(X, y):
    X = utils.preprocess(X)
    y = torch.LongTensor(y)
    model, params = get_best_model()
    model.eval()
    X.requires_grad_()
    scores = model(X)
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
    correct_scores.backward(torch.FloatTensor([1, 1]))
    grads = X.grad.data
    saliency, indices = torch.max(grads.abs(), dim = 1)
    return saliency.numpy()

def tsne(X):
    y_pred, prob, embed = predict(X)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(embed)
    return X_tsne

def show_heatmap(heatmap, image):
    img_h, img_w = image.shape
    heatmap = transform.resize(heatmap, (img_h, img_w), mode='constant', anti_aliasing=True)
    plt.imshow(image, cmap='gray')
    im = plt.imshow(heatmap, alpha=0.5)
    plt.tight_layout()
    plt.colorbar(im)

def get_images(img_type, size):
    paths = ["./data/test/{}/{}.jpg".format(img_type, i) for i in range(size)]
    images = [utils.load_image(img_path) for img_path in paths]
    return images

norm_images = get_images("normal", 5)
pneu_images = get_images("pneumonia", 5)

# img = norm_images[2]
# y_pred, prob, pre_output = predict([img], [0])
# diff = occlude(img, prob)
# plt.figure()
# show_heatmap(diff, img)
# plt.title("Normal X-Ray")
# plt.show()

img = pneu_images[0]
y_pred, prob, pre_output = predict([img], [1])
diff = occlude(img, prob)
plt.figure()
show_heatmap(diff, img)
plt.title("Pneumonia X-Ray")
plt.show()


# y_pred, prob, pre_output = predict([img], [1])
# print(prob, y_pred)
# diff = occlude(img, prob)
# show_heatmap(diff, img)



# n = 100
# norm_images = get_images("normal", n)
# pneu_images = get_images("pneumonia", n)

# images = norm_images + pneu_images
# labels = np.array([0]*n + [1]*n)
# index = np.random.permutation(2*n)
# X_shuf = [images[i] for i in index]
# y_shuf = [labels[i] for i in index]
# X_tsne = tsne(X_shuf)

# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_shuf, alpha=0.6, 
#             cmap=plt.cm.get_cmap('rainbow', 10))
# plt.show()


# plt.scatter(X_tsne[:n, 0], X_tsne[:n, 1], label="normal")
# plt.scatter(X_tsne[n:, 0], X_tsne[n:, 1], label="pneumonia")
# plt.legend()
# plt.show()

# img_path = "./data/test/pneumonia/744.jpg"
# img = utils.load_image(img_path)
# y_pred, prob, pre_output = predict([img], [1])
# saliency = silency_map([img], [1])[0]
# print(y_pred, prob)
# show_heatmap(saliency, img)

# Todo:
# 1. CAM:
# 2. Retrieving 
# 3. Deconvolve
# 4. tsne

# Web UI (Front end, server) 
#   Front end: HTML 
#   Server (Javascript): (DOM) 
#       1. Get Image => X
#       2. Call python function occlude => image
#       3. Show image on HTML