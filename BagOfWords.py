__author__ = 'KSM'

"""
Author: Kunjan Mhaske

This program extracts SIFT features from given dataset and plots
the keypoints on the respected image to visualize the features.
Further, it creates Bag of Words using KMeans clustering to learn
the features specifications with respect to specific category of the
images. 
Additionally, the SVM classifier learns from those clusters to predict
the class/category of given query images.
"""

import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def getImages(dataset_dir):
    '''
    This method takes name of directory and returns the list of files
    with their address
    :param dataset_dir: name of directory
    :return: list of image names in given directory
    '''
    folders = os.listdir(dataset_dir)
    dataset = []
    for folder in folders:
        files = os.listdir(dataset_dir + "/" + folder)
        each_dir = []
        for file in files:
            each_dir.append(dataset_dir + "/" + folder + "/" + file)
        each_dir.sort()
        dataset.append(each_dir)
    return dataset

def matchKeypoints(path, image1, image2):
    '''
    This method matches the keypoints between given two images
    :param path: path of given image
    :param image1: base image information to compare
    :param image2: target image information to compare
    :return: None
    '''
    img1name = image1[0]
    img1keyp = image1[1]
    img1desc = image1[2]
    img1 = image1[3]

    img2name = image2[0]
    img2keyp = image2[1]
    img2desc = image2[2]
    img2 = image2[3]

    print("Matching ", img1name, "and", img2name)
    # Take BruteForceMatcher object
    # NORM_L1 = manhattan distance
    # NORM_L2 = euclidean distance
    bfm = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    # match the descripters
    match = bfm.match(img1desc, img2desc)

    # sort the matches according to distances
    match = sorted(match, key=lambda i: i.distance)

    # draw top 20 matches
    match_img = cv2.drawMatches(img1, img1keyp, img2, img2keyp, match[:20], img2.copy(), flags=0)

    category_name = img1name.split('/')[1] + "_"
    save_name1 = img1name.split('/')[-1].split('.')[0]
    save_name2 = img2name.split('/')[-1].split('.')[0]
    save_name = category_name + save_name1 + save_name2
    save_name = ''.join(save_name)
    cv2.imwrite(os.path.join(path, save_name + "_KPMatches.png"), match_img)

def extractFeatures(path, srcImg_name, flag=False):
    '''
    This method extracts the keypoints and descriptors from
    the given images and plots them on the original image to
    visualize the keypoints as features.
    :param path: path of image
    :param srcImg_name: image name
    :param flag: True for saving images, False for not saving
    :return: keypoints, descriptors, image
    '''
    print("Extracting Features: ", srcImg_name)
    color_img = cv2.imread(srcImg_name, 1)
    # convert to gray
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # Generate the sift features
    # Install opencv-contrib-python=3.4.2.17
    # to enable SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    """
    keyp = keyPoint
    desc = SIFT descripter (128 dimensional vectors)
    """
    keyp, desc = sift.detectAndCompute(gray_img, None)

    # plt.imshow(desc, interpolation='none')
    # plt.show()
    # Visualize the features with keypoints
    kpImage = cv2.drawKeypoints(gray_img, keyp, color_img.copy())

    if flag:
        save_name = srcImg_name.split('/')[0:]
        category_name = [save_name[1], "_"]
        save_name = category_name + save_name[-1].split('.')
        save_name = ''.join(save_name[:-1])
        cv2.imwrite(os.path.join(path, save_name + "_Keypoints.png"), kpImage)
        return keyp, desc, color_img
    else:
        return keyp, desc, gray_img

def createIndividualHistogram(original_histogram):
    image_no = 0
    print("Creating Histogram for 1 image of each category...")
    x = np.arange(n_clusters)
    for folder in dataset:
        y = np.array(original_histogram[image_no,:])
        plt.bar(x,y)
        plt.xlabel('Cluster_Number')
        plt.ylabel('Frequency')
        plt.xticks(x+0.7, x)
        plt.savefig("histogram_image1_"+str(folder[0].split('/')[1])+".png")
        # plt.show()
        plt.clf()
        image_no += len(folder)

    plt.hist(kmeans_clust, bins=np.arange(n_clusters))
    plt.xlabel('Cluster_Number')
    plt.ylabel('Frequency')
    plt.savefig("Histogram_all_clusters_in_bag_of_words.png")
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    print("There must be separate folders for 'Training_Data' and 'Testing_Data'")
    print("Inside of Training_Data: Folders containing the bunch of images of same category.")
    print("Both Folders should be present in the current directory as this program.")
    print()
    print("Training_Data: \n \t\t Category_1: \n \t\t\t\t image1.jpg \n \t\t\t\t image2.jpg . . .\n \t\t Category_2: . . .")
    print("Testing_Data: \n \t\t Category_1: \n \t\t\t\t image1.jpg \n \t\t\t\t image2.jpg . . . \n \t\t Category_2: . . .")
    print()
    print("Inputs given to programs should be name of outer folder eg.: 'Training_data'")
    print("Histogram Outputs will be saved in current directory")
    print("KeyPoints plots, Keypoints Matches and Testing results will be saved in auto generated separate folders in current directory.")
    print("---------------------------------------------------------------")
    print()
    dataset_dir = input("Enter folder name of training data:")
    # fetch the paths of all images according to given category
    dataset = getImages(dataset_dir)
    # features = {category: [[img1], [img2], ....]}
    features = {}

    directory = "KeyPoints"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # extract the keypoints and descriptors from the given dataset
    for data in dataset:
        # image_list = [[path, keyp, desc, color_img]]
        image_list = []
        category = data[0].split('/')[1]
        for img_name in data:
            keyp, desc, color_img = extractFeatures(directory, img_name,True)
            image_list.append([img_name, keyp, desc, color_img])
        if features.__contains__(category):
            features[category].append(image_list.copy())
        else:
            features[category] = image_list.copy()
        image_list.clear()

    print()
    # show matches of keypoints between first image with other images from each category
    directory = "KPMatches"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for category in features:
        first = features[category][0]
        for x in range(1,len(features[category])):
            image1 = first
            image2 = features[category][x]
            matchKeypoints(directory,image1, image2)

    ############### creation of bag-of-words start #####################
    print("Creating bag of words..............")
    all_desc_list = []
    training_label = np.array([])
    label_dict = {}
    label_count = 0
    image_count = 0
    # generating training labels and desciptors list
    for category in features:
        label_dict[label_count] = str(category)
        for x in range(len(features[category])):
            training_label = np.append(training_label,label_count)
            image_count += 1
            all_desc_list.append(features[category][x][2])
        label_count += 1
    # print("descriptor: ", all_desc_list[0:10])

    # create vstack of descriptors to feed into kmeans clutering
    vStack = np.array(all_desc_list[0])
    for des in all_desc_list[1:]:
        vStack = np.vstack((vStack, des))
    # print("vStack: ", vStack[0:10])

    # perform KMeans clustering
    print("Performing clustering................ (It may take more time)")
    n_clusters = 100
    KMeans_obj = KMeans(n_clusters=n_clusters)
    kmeans_clust = KMeans.fit_predict(KMeans_obj,vStack)
    # print("KmeansClusters: ",kmeans_clust[0:10])
    print("KMeansCluster size: ", kmeans_clust.shape)
    print("Kmeans_Cluster_Centers: ",len(KMeans_obj.cluster_centers_))

    # develop vocabulary
    print("Developing vocabulary.................")
    vocab_hist = np.array([np.zeros(n_clusters) for i in range(image_count)])
    count = 0
    for i in range(image_count):
        new_arr = np.array(np.zeros([len(all_desc_list[i])]))
        # new_arr = kmeans_clust[count:len(new_arr)]
        l= len(all_desc_list[i])
        for k in range(l):
            new_arr[k]=kmeans_clust[count+k]
        tmp_arr = np.array(np.zeros(n_clusters))
        for j in new_arr:
            vocab_hist[i][int(j)] += 1
        count += len(all_desc_list[i])
    print("vocab_hist_shape", vocab_hist.shape)

    original_histogram = vocab_hist.copy()
    # show histograms
    createIndividualHistogram(original_histogram)

    print("Bag of words created.")
    print()
    ############################## bag-of-words creation completed #############################
    #------------------------------------------------------------------------------------------#
    ############################## training and testing start ##################################
    # preprocessing for training
    print("Training the model start.............")
    training_scale = StandardScaler().fit(vocab_hist)
    vocab_hist = training_scale.transform(vocab_hist)
    # print("scale of standardScalar:",training_scale)

    # training using SVC()
    training_clf = SVC(gamma='scale')
    # set gamma='scale' to suppress the FutureWarning
    training_clf.fit(vocab_hist,training_label)
    print("classifier:",training_clf)
    print("classifier classes:",training_clf.classes_)
    print("Training is Done here...")
    print("----------------------------------------------")
    print()

    # testing start:
    test_dataset_dir = input("Enter folder name of Testing data:")
    test_dataset = getImages(test_dataset_dir)
    # test_predictions = [{'Image':test_img,'class':label,'obj':label_dict[cat[0]},{...},{...}]
    test_predictions = []
    # image_list = [[keypoints, descriptor, gray_img, category, image_name],[....],[....],...]
    images_list = []

    # get keypoints and descriptors from testing dataset
    print("Testing started.............")
    for data in test_dataset:
        categ = data[0].split('/')[1]
        for img_name in data:
            keyp, desc, gray_img = extractFeatures(directory, img_name)
            images_list.append([keyp, desc, gray_img, categ, img_name])

    directory = "Results"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # recognize the image and get label for each image in testing dataset
    for image in images_list:
        kyp = image[0]
        descr = image[1]
        test_img = image[2]
        cat = image[3]
        og_image_name = image[4]
        print("Testing for image:",og_image_name)

        test_vocab = np.array([np.zeros(n_clusters) for i in range(1)])
        # clustering of image descriptors
        test_kmeans_return = KMeans_obj.predict(descr)

        for each in test_kmeans_return:
            test_vocab[0][each] += 1
        # print("test_vocab :", vocab_hist)

        # scale the features
        test_vocab = training_scale.transform(test_vocab)

        # predict the class of the image
        result_label = training_clf.predict(test_vocab)
        print()
        print("!!! result labels:",result_label)
        print("Checking given_folder_name as category:")
        print(str(cat)," : ",str(label_dict[int(result_label[0])]))

        # test_predictions = [{'Image':test_img,'class':label,'obj':label_dict[cat[0]},{...},{...}]
        test_predictions.append({ 'Image':test_img,
                                  'Class':result_label,
                                  'Category':label_dict[int(result_label[0])],
                                  'Name':og_image_name})

        print("test_predictions_size: ",len(test_predictions))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("Results: ")
    for each in test_predictions:
        print(each)
        print()
        save_name = each['Name'].split('/')[0:]
        save_name = save_name[-1].split('.')
        save_name = ''.join(save_name[:-1])
        save_name = save_name+"_"+each['Category']
        save_img = cv2.cvtColor(each['Image'], cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(directory, save_name + "_Result.png"), save_img)
        # plt.imshow(save_img)
        # plt.title(save_name)
        # plt.show()
