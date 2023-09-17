import numpy as np
import matplotlib.pyplot as plt

def image_to_matrix(image_file, grays=False):
    """
    Convert .png image to matrix
    of values.
    params:
    image_file = str
    grays = Boolean
    returns:
    img = (color) np.ndarray[np.ndarray[np.ndarray[float]]]
    or (grayscale) np.ndarray[np.ndarray[float]]
    """
    img = plt.imread(image_file)
    # in case of transparency values
    if len(img.shape) == 3 and img.shape[2] > 3:
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = img[r, c, 0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = img[r, c, 0]
        img = new_img
    return img

def plot_image(img_list, title_list, figsize=(15, 10)):
    fig, axes = plt.subplots(len(img_list)//2, 2, figsize=figsize)
    p = 0
    for i, ax in enumerate(axes):
        for col in range(len(img_list)//2):
            axes[i, col].imshow(img_list[p])
            axes[i, col].set_title(title_list[p])
            axes[i, col].axis('off')
            p += 1