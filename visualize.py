from predict import predict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import utils
from skimage import transform

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

    print(diff)
    # plt.figure()
    # for i, occ in enumerate(occ_imgs):
    #     plt.subplot(n_row, n_col, i+1)
    #     plt.imshow(occ, cmap='gray')

    heatmap = transform.resize(diff, (img_h, img_w))
    plt.figure()
    plt.imshow(orig_img, cmap='gray')
    im = plt.imshow(heatmap, alpha=0.5)
    plt.tight_layout()
    plt.colorbar(im)
    plt.show()

img_path = "./data/test/normal/0.jpg"
img = utils.load_image(img_path)
y_pred, prob, pre_output = predict([img], [0])
occlude(img, prob)