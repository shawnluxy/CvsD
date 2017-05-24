import numpy as np
import cv2
import os
import csv
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

sift = cv2.xfeatures2d.SIFT_create(nfeatures=50)
image_train = []
image_test = []


def getSIFTFeatures(image, sift):
    kp, des = sift.detectAndCompute(image, None)
    return des


def HOGFeatures(image):
    winSize = (16, 16)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 2.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(image, winStride, padding, locations)
    return hist


def Classifier(features, labels, classifier):
    (X_train, X_test, Y_train, Y_test) = train_test_split(features, labels, test_size=0.25, random_state=42)
    classifier.fit(X_train, Y_train)
    score = classifier.score(X_test, Y_test)
    return score

# path = '/root/Downloads/ECSE-415/X_Train'
# # p_len = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
# p_len = 100
# Features = []
# Descriptors = []
# Labels = []
# for i in range(0, p_len):
#     image = cv2.imread(path+"/"+str(i)+".jpg")
#     descriptor = getSIFTFeatures(image, sift)
#     # descriptor = HOGFeatures(image)
#     # descriptor = descriptor.reshape(descriptor.shape[0])
#     # Features.append(descriptor)
#     Descriptors.append(descriptor)
#
# flann_params = dict(algorithm=1, trees=5)
# matcher = cv2.FlannBasedMatcher(flann_params, {})
# bow_extract = cv2.BOWImgDescriptorExtractor(sift, matcher)
# bow_train = cv2.BOWKMeansTrainer(20)
# for des in Descriptors:
#     bow_train.add(des)
# voc = bow_train.cluster()
# bow_extract.setVocabulary(voc)
#
# for i in range(0, p_len):
#     image = cv2.imread(path+"/"+str(i)+".jpg")
#     feature = bow_extract.compute(image, sift.detect(image))
#     Features.extend(feature)
#
#
# with open('/root/Downloads/ECSE-415/Y_Train.csv') as csvfile:
#     read = csv.reader(csvfile, delimiter=',')
#     for row in read:
#         Labels.append(row[1])
#     Labels = Labels[1:101]
#
# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes"]
# classifiers = [
#     KNeighborsClassifier(5),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB()]
#
# for name, clf in zip(names, classifiers):
#     scores = Classifier(np.array(Features), np.array(Labels), clf)
#     print name, scores*100


def extractForeground(image):
    h, w, d = image.shape
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (int(h*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    # plt.imshow(image), plt.colorbar(), plt.show()
    return image


def getBOW(path, images):
    Descriptors = []
    for i, image in enumerate(images):
        kp, des = sift.detectAndCompute(image, None)
        if des is None:
            print("Empty des for " + str(i))
            img = cv2.imread(path + "/" + str(i) + ".jpg")
            kp, des = sift.detectAndCompute(img, None)
        Descriptors.append(des)
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    bow_extract = cv2.BOWImgDescriptorExtractor(sift, matcher)
    bow_train = cv2.BOWKMeansTrainer(20)
    for i, des in enumerate(Descriptors):
        print("BOW train " + str(i))
        bow_train.add(des)
    voc = bow_train.cluster()
    bow_extract.setVocabulary(voc)
    return bow_extract


def getImageFeatures(path, images, bow_extract):
    Features = []
    for i, image in enumerate(images):
        feature = bow_extract.compute(image, sift.detect(image))
        if feature is None:
            print("Error for " + str(i))
            img = cv2.imread(path + "/" + str(i) + ".jpg")
            feature = bow_extract.compute(img, sift.detect(img))
        Features.extend(feature)
    return Features


def getTrainLabels(file):
    Labels_train = []
    with open(file) as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        for row in read:
            Labels_train.append(row[1])
    Labels_train = Labels_train[1:]
    return np.array(Labels_train, dtype=np.int32)


def train(classifier):
    path1 = '/root/Downloads/ECSE-415/X_Train'
    path2 = '/root/Downloads/ECSE-415/X_Test'
    p_len1 = len([f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))])
    p_len2 = len([f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))])
    # p_len1 = 100
    # p_len2 = 100
    for i in range(0, p_len1):
        print("Extract1 for:" + str(i))
        image = cv2.imread(path1 + "/" + str(i) + ".jpg")
        image = extractForeground(image)
        image_train.append(image)
    for i in range(0, p_len2):
        print("Extract2 for:" + str(i))
        image = cv2.imread(path2 + "/" + str(i) + ".jpg")
        image = extractForeground(image)
        image_test.append(image)

    bow_extract = getBOW(path1, image_train)
    Features_train = getImageFeatures(path1, image_train, bow_extract)
    Labels_train = getTrainLabels('/root/Downloads/ECSE-415/Y_Train.csv')

    clf = classifier.fit(Features_train, Labels_train)
    return bow_extract, clf


def predict(image_test, bow_extract, clf):
    path2 = '/root/Downloads/ECSE-415/X_Test'
    Features_predict = getImageFeatures(path2, image_test, bow_extract)
    prediction = clf.predict(Features_predict)

    with open('/root/Downloads/ECSE-415/Y_Test.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Image', 'Label'])
        for i, feature in enumerate(Features_predict):
            prediction = clf.predict(feature)
            name = str(i) + ".jpg"
            writer.writerow([name, str(prediction[0])])

bow_extract, clf = train(svm.LinearSVC())
predict(image_test, bow_extract, clf)

# extractForeground(cv2.imread('/root/Downloads/ECSE-415/X_Train/42.jpg'))
