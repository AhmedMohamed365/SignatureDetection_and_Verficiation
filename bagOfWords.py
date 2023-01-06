import cv2
import numpy as np
from glob import glob
import argparse
from helpers import *
from matplotlib import pyplot as plt
import pickle
from sklearn import externals



class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.test_path_f=None

    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print("Computing Features for ", word)
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                self.descriptor_list.append(des)

            label_count += 1

        # perform clustering
        self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        self.bov_helper.plotHist()

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)

    def recognize(self, test_img, test_image_path=None):

        """
        This method recognizes a single image
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)
        # print kp


        # generate vocab for test image
        vocab = np.array([[0 for i in range(self.no_clusters)]])
        vocab = np.array(vocab, 'float32')
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        # print (vocab)

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)
        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        # pickled_model = pickle.load(open('model.pkl', 'rb'))
        # lb = pickled_model.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb
    def recognize_t(self, test_img, test_image_path=None):

        """
        This method recognizes a single image
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)
        # print kp


        # generate vocab for test image
        vocab = np.array([[0 for i in range(self.no_clusters)]])
        vocab = np.array(vocab, 'float32')
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        pickled_model = pickle.load(open('model_kmeans_f.pkl', 'rb'))
        # des=np.array(des)
        test_ret=pickled_model.predict(des)
        # test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        # print (vocab)

        # Scale the features
        scale = pickle.load(open('scale_f.pkl', 'rb'))
        vocab = scale.transform(vocab)
        # predict the class of the image
        pickled_model = pickle.load(open('model_f.pkl', 'rb'))

        lb=pickled_model.predict(vocab)
        # lb = self.bov_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    def testModel(self):
        """
        This method is to test the trained classifier

        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image

        """
        correctClassifications = 0
        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path,False)

        predictions = []

        for word, imlist in self.testImages.items():
            print("processing ", word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                print(im.shape)
                cl = self.recognize(im)
                print(cl)
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]
                })

                if (self.name_dict[str(int(cl[0]))] == word):
                    correctClassifications = correctClassifications + 1

        print("Test Accuracy = " + str((correctClassifications / self.testImageCount) * 100))
        # print (predictions)
        for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            #
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()
    def testModel_save(self):
        """
        This method is to test the trained classifier

        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image

        """
        correctClassifications = 0
        self.testImages, self.testImageCount = self.file_helper.getFiles1(self.test_path_f)

        predictions = []

        for word, imlist in self.testImages.items():

            for im in imlist:
                # print imlist[0].shape, imlist[1].shape

                cl = self.recognize_t(im)

                if cl[0]==0:
                    h='personA'
                elif cl[0]==1:
                    h='personB'
                elif cl[0]==2:
                    h='personC'
                elif cl[0]==3:
                    h='personD'
                elif cl[0]==4:
                    h='personE'

                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': h
                })
                print("h:",h)
                print("word",word)
                if (h == word):
                    correctClassifications = correctClassifications + 1

        print("Test Accuracy = " + str((correctClassifications / self.testImageCount) * 100))

        for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            #
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()

    def print_vars(self):
        pass
    def testModel_save_new(self, path):
        im = cv2.imread(path , 0)
        cl = self.recognize_t(im)
        return cl[0]




def bow(path):
    # parse cmd args
    parser = argparse.ArgumentParser(
        description=" Bag of visual words example"
    )

    parser.add_argument('--test_path1', default="SignatureTestSamples", action="store", dest="test_path1")
    args = vars(parser.parse_args())


    bov = BOV(no_clusters=65)

    bov.test_path_f = args['test_path1']

    x = bov.testModel_save_new(path)
    a = []
    a.append(int(x))
    return a


