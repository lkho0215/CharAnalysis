import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

from skimage import io
import os
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.externals import joblib
from skimage.util import random_noise

from scipy import ndimage

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

import math

def calc_hog_vec(file_path):
    image = io.imread(file_path)

    aspect = image.shape[1]/image.shape[0]
    if aspect > 1:
        size = (math.floor(60/aspect), 60)
        padding =(60 - size[0])//2
        npad = ((padding, padding), (0, 0), (0,0))
    else:
        size = (60, math.floor(60*aspect))
        padding =(60 - size[1])//2
        npad = ((0, 0), (padding, padding), (0,0))

    image = resize(image, size, anti_aliasing=True)
    image = np.pad(image, npad,  mode='constant', constant_values=0)
    image = resize(image, (60,60), anti_aliasing=True)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(6, 6), block_norm='L2-Hys',
                        cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=True)

    # print(fd.shape)

    # plt.clf()
    # plt.imshow(hog_image)
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()
    return fd


def get_data():
    X = []
    y = []
    data_dir = "./data/noise"
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".jpg"):
            file_path = os.path.join(data_dir, file)
            hog_vec = calc_hog_vec(file_path)
            X.append(hog_vec)
            y.append(0)

    data_dir = "./data/letter"
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".jpg"):
            file_path = os.path.join(data_dir, file)
            hog_vec = calc_hog_vec(file_path)
            X.append(hog_vec)
            y.append(1)


    X = np.array(X)
    y = np.array(y)
    y = y.flatten()

    return X, y


def train(X, y):

    # clf = svm.SVC(probability=True)
    # clf.fit(X ,y)
    # joblib.dump(clf, 'svc.pkl')


    clf = LDA()
    clf.fit(X, y)
    joblib.dump(clf, 'lda.pkl')

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    #                          algorithm="SAMME",
    #                          n_estimators=200)

    # clf.fit(X, y)

    # joblib.dump(clf, 'ada.pkl')
    
    return clf







def test():
    clf = joblib.load('svc.pkl') 
    test_x = []
    test_y = []
    data_dir = "./data/test_noise"
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            file_path = os.path.join(data_dir, file)
            hog_vec = calc_hog_vec(file_path)
            # image = io.imread(file_path)
            # image = resize(image, (60, 60), anti_aliasing=True)
            # print(clf.predict([hog_vec]))
            # print(clf.predict_proba([hog_vec]))
            # print("=================")
            # plt.clf()
            # plt.imshow(image)
            # plt.show()
            print(clf.predict([hog_vec]))
            test_x.append(hog_vec)
            test_y.append(0)

    data_dir = "./data/test_letter"
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            file_path = os.path.join(data_dir, file)
            hog_vec = calc_hog_vec(file_path)
            test_x.append(hog_vec)
            test_y.append(1)
            if(clf.predict([hog_vec]) == [0]):
                image = io.imread(file_path)
                image = resize(image, (60, 60), anti_aliasing=True)
                io.imsave(os.path.join("./wrong", file), image)
                print(clf.predict_proba([hog_vec]), file)
                # print(clf.predict_proba([hog_vec]))
                # print("=================")
                # plt.clf()
                # plt.imshow(image)
                # plt.show()



    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_y = test_y.flatten()
    print(clf.predict(test_x))

    # print(clf.predict_proba(test_x))

def rough_classify():
    clf = joblib.load('svc.pkl') 
    data_dir = "./char"
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            file_path = os.path.join(data_dir, file)
            hog_vec = calc_hog_vec(file_path)
            if(clf.predict([hog_vec]) == [0]):
                image = io.imread(file_path)
                io.imsave(os.path.join("./rough/noise", file), image)

            if(clf.predict([hog_vec]) == [1]):
                image = io.imread(file_path)
                io.imsave(os.path.join("./rough/letter", file), image)


def main():

    X, y = get_data()
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = train(X_train, y_train)
        print(clf.score(X_test, y_test))

   #train()
   #test()
   #rough_classify()







if __name__ == "__main__":
    main()