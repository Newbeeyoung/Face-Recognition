import cv2
import numpy as np

path="./orl_faces/"

class Face:
    def __init__(self):
        self.f_avg=None
        self.feature_matrix=None
        self.reduced_eigen=[]

    def train(self):
        final_matrix = None
        for folder_no in range(1, 41):
            for file_no in range(1, 11):

                # Open train image
                file_path = path + "s" + str(folder_no) + "/" + str(file_no) + ".pgm"
                img = cv2.imread(file_path, -1)

                # Generate MxN matrix from images, M is wxh of image, N is number of images
                img_c = np.reshape(img, (img.shape[0] * img.shape[1], 1))
                if img is not None:
                    print("Read file" + file_path)
                    print("shape:" + str(img_c.shape))
                if final_matrix is None:
                    final_matrix = img_c
                else:
                    final_matrix = np.concatenate((final_matrix, img_c), axis=1)

        # Add own train images to MxN matrix, shape change to Mx(N+n), n is number of own images
        img_train = cv2.imread("jiang1.jpeg", 0)
        img_train = cv2.resize(img_train, (92, 112))
        img_train_c = np.reshape(img_train, (img_train.shape[0] * img_train.shape[1], 1))
        final_matrix = np.concatenate((final_matrix, img_train_c), axis=1)
        print("Load own images")
        # Calculate average column from matrix
        self.f_avg = 1 / final_matrix.shape[1] * np.sum(final_matrix, axis=1)
        self.f_avg = np.reshape(self.f_avg, (self.f_avg.shape[0], 1))
        np.save("f_avg.npy",self.f_avg)
        # Obtain X, normalized matrix
        X = np.subtract(final_matrix, self.f_avg)

        # Calculate covariance matrix from X
        covariance = np.multiply(1 / final_matrix.shape[1], np.dot(X, np.transpose(X)))

        print("Computing Eigenvector...")
        # Calcuate eigenvalues and eigenvector of covariance matrix
        eigenvalue,eigenvector=np.linalg.eig(covariance)

        # Find the largest k eigenvalues and their corresponding eigenvectors
        k=200
        idx=eigenvalue.argsort()[-k:][::-1]
        print("eigenvalue:"+str(eigenvalue))
        print("sorted value:"+str(idx))

        for index in idx:
            self.reduced_eigen.append(eigenvector[:,index])

        self.reduced_eigen=np.asarray(self.reduced_eigen)
        print("Obtain Eigenvector")

        np.save("reduced_eigen.npy",self.reduced_eigen)
        # Obtain feature matrix of training images
        self.feature_matrix=np.dot(self.reduced_eigen,X)

        np.save("feature.npy",self.feature_matrix)
        print("Obtain feature matrix")
        print("Train finished")


    def load_train_data(self):
        self.f_avg=np.load("f_avg.npy")
        self.feature_matrix=np.load("feature.npy")
        self.reduced_eigen=np.load("reduced_eigen.npy")

    def test(self,img_test):

        # Open test image and obtain its feature
        # img_test=cv2.imread("jiang3.jpg",0)

        img_test = cv2.resize(img_test, (92, 112))
        img_test_c = np.reshape(img_test, (img_test.shape[0] * img_test.shape[1], 1))
        test_feature=np.dot(self.reduced_eigen,np.subtract(img_test_c,self.f_avg))

        # Find most matched image by Nearest Neighbour classifier
        min_dist=np.linalg.norm(test_feature-self.feature_matrix[:,0].reshape(self.feature_matrix.shape[0],1))
        min_dist_index=0
        for n in range(1,self.feature_matrix.shape[1]):
            dist = np.linalg.norm(test_feature - self.feature_matrix[:, n].reshape(self.feature_matrix.shape[0],1))
            if min_dist>dist:
                min_dist=dist
                min_dist_index=n+1
                # print("Min dist no:"+str(min_dist_index))
                # print("Min dist:"+str(min_dist))
        if min_dist_index==401:
            name="Jiang"
        else:
            name="Unknown"
        print("The person is "+ name)
        return name
