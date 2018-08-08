from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os 
import random

from skimage.transform import rescale, resize, downscale_local_mean

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def calc_lbp_hist(file_path):
    image = io.imread(file_path, as_gray=True)
    image = resize(image, (28, 28), anti_aliasing=True)
    lbp = local_binary_pattern(image, n_points, radius, METHOD)


    values = lbp
    heights,bins = np.histogram(values,bins=10)
    heights = heights/sum(heights)
    plt.clf()
    plt.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="blue", alpha=0.5)
    plt.ylim((0, 1))
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    return plt

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def main():
    data_dir = "./data/noise"
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            file_path = os.path.join(data_dir, file)
            ax = calc_lbp_hist(file_path)
            ax.savefig("./figure/fig_" + file +".png")

            # heights, bins = np.histogram(img, bins=np.arange(0, 255, 10))

            # heights = heights/sum(heights)

            # plt.bar(bins[:-1], heights, width=(max(bins) - min(bins))/len(bins), color="blue", alpha=0.5)
            # plt.title("Gaussian Histogram")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.ylim(0, 1)
            # plt.grid(True)
            # plt.savefig("./figure/" + str(random.randint(1, 9999)) +".png")
            # plt.clf()


if __name__ == "__main__":
    main()