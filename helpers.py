import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
import pickle


class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = SVC()

    def cluster(self):
        """
		cluster using KMeans algorithm,

		"""
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)
        # pickle.dump(self.kmeans_obj, open('model_kmeans_f.pkl', 'wb'))


    def developVocabulary(self, n_images, descriptor_list):

        """
		Each cluster denotes a particular visual word
		Every image can be represeted as a combination of multiple
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word

		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images

		"""

        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                idx = self.kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print("Vocabulary Histogram Generated")

    def standardize(self, std=None):
        """

		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.

		"""
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            # pickle.dump(self.scale , open('scale_f.pkl', 'wb'))
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        """
		restructures list into vstack array of shape
		M samples x N features for sklearn

		"""
        vStack = np.array(l[0])
        for remaining in l[1:]:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return

    def train(self, train_labels):
        """
		uses sklearn.svm.SVC classifier (SVM)


		"""
        print("Training SVM")
        print(self.clf)
        print("Train labels", train_labels)
        self.clf.fit(self.mega_histogram, train_labels)
        # pickle.dump(self.clf, open('model_f.pkl', 'wb'))
        print("Training completed")

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary=None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


class FileHelpers:

    def __init__(self):
        pass

    def getFiles(self, path, train=True):
        """
		- returns  a dictionary of all files
		having key => value as  objectname => image path

		- returns total number of files.

		"""
        tybe = ""
        if (train):
            tybe = "Train"
        else:
            tybe = "Test"

        imlist = {}
        count = 0
        for each in os.listdir(path):
            print(" #### Reading image category ", each, " ##### ")
            imlist[each] = []
            for imagefile in os.listdir(path + '/' + each + '/' + tybe):
                if imagefile.__contains__(".csv"):
                    print("Reading file ", imagefile)
                else:
                    print("Reading file ", imagefile)

                    im = cv2.imread(path + '/' + each + '/' + tybe + '/' + imagefile, 0)
                    imlist[each].append(im)
                    count += 1




        return [imlist, count]
    def getFiles1(self, path):
            """
            - returns  a dictionary of all files
            having key => value as  objectname => image path

            - returns total number of files.

            """



            imlist = {}
            count = 0
            for imagefile in os.listdir(path):
                imlist[imagefile[0:7]]=[]

            for imagefile in os.listdir(path  ):

                if imagefile.__contains__(".csv"):
                    print("Reading file ", imagefile)
                else:
                    print("Reading file ", imagefile)

                    im = cv2.imread(path  + '/' + imagefile, 0)
                    imlist[imagefile[0:7]].append(im)
                    count += 1




            return [imlist, count]






