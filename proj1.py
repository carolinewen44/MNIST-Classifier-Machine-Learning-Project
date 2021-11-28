import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mlxtend.data import loadlocal_mnist
import platform
from matplotlib import pyplot as plt
import numpy as np


numtrain = 60000
numtest = 400 #cannot test on the full dataset because not enough memory available
              #change this to 10000 for full test set results

#load in MNIST train and test data
trainImgs, trainLabels = loadlocal_mnist(
            images_path='samples/train-images.idx3-ubyte', 
            labels_path='samples/train-labels.idx1-ubyte')
testImgs, testLabels = loadlocal_mnist(
            images_path='samples/t10k-images.idx3-ubyte', 
            labels_path='samples/t10k-labels.idx1-ubyte')

# #test to see if data is brought in correctly, plots 7th digit (it works!)
# first_image = trainImgs[7]
# first_image = np.array(first_image, dtype='float')
# pixels = first_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
# print(first_image)

# ####### (1) Naive Bayes using ML estimation with Gaussian Assumption #######
# mdl_nb = GaussianNB()
# test_pred_labels = mdl_nb.fit(trainImgs, trainLabels).predict(testImgs)
# train_pred_labels = mdl_nb.fit(trainImgs, trainLabels).predict(trainImgs)
# train_err_nb = ((trainLabels!=train_pred_labels).sum())/len(trainLabels)
# test_err_nb = ((testLabels!=test_pred_labels).sum())/len(testLabels)
# print("Naive Bayes test error rate: %f%%" % (test_err_nb*100))
# print("Naive Bayes training error rate: %f%%" % (train_err_nb*100))


# ####### (2) Nearest Neighbors #######
# N_neighbors = [1, 5, 10, 20, 50, 100]
# train_err_knn = []
# test_err_knn = []

# for i in N_neighbors:
#     #train k nearest neighbor classifier
#     mdl = KNeighborsClassifier(n_neighbors=i)
#     mdl.fit(trainImgs, trainLabels)

#     #evaluate the model on test
#     score = mdl.score(testImgs, testLabels) 
#     trainscore = mdl.score(trainImgs, trainLabels)
#     print("N_neighbor = %d, test error =%.2f%%, train error =%.2f%%" % (i, (1-score)*100, (1-trainscore)*100))
#     test_err_knn.append((1-score)*100)
#     train_err_knn.append((1-trainscore)*100)

# #plot training/test error vs N_neighbors 
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(N_neighbors, test_err_knn)
# ax1.set_title('Test Error rate vs N_neighbors')
# ax1.set(xlabel = 'N_neighbors', ylabel = 'Test Error Rate')
# ax2.plot(N_neighbors, train_err_knn)
# ax2.set_title('Train Error rate vs N_neighbors')
# ax2.set(xlabel = 'N_neighbors', ylabel = 'Train Error Rate')


# ######## (3) Fishers LDA Classifier  ########
# # digit 0 vs digit 9
# Ci = 0
# Cj = 9
# #choose only digits 0 and 9 from training set
# Ind09=np.array([ind for ind in range(numtrain) if ((trainLabels[ind]==Ci) or (trainLabels[ind]==Cj))])
# trainImg_09 = trainImgs[Ind09]
# trainLabels_09 = trainLabels[Ind09]
# #print("length of training set: ", len(Ind09))
# n_09 = len(Ind09)
# #choose only digits 0 and 9 from test set
# Ind09=np.array([ind for ind in range(10000) if ((testLabels[ind]==Ci) or (testLabels[ind]==Cj))])
# testImg_09 = testImgs[Ind09]
# testLabels_09 = testLabels[Ind09]
# #print("length of test set: ", len(Ind09))
# ntest_09 = len(Ind09)

# #train and test data for 0 and 9 - calculate error
# mdl_lda = LDA()
# mdl_lda.fit(trainImg_09, trainLabels_09)
# train_err_lda09 = 100*(np.count_nonzero(np.array(mdl_lda.predict(trainImg_09))-trainLabels_09)/n_09)
# test_err_lda09 = 100*(np.count_nonzero(np.array(mdl_lda.predict(testImg_09))-testLabels_09)/ntest_09)
# print("Digit 0 vs Digit 9: training error = %f%% , test error = %f%%" % (train_err_lda09, test_err_lda09))

# # digit 0 vs digit 8
# Ci = 0
# Cj = 8
# #choose only digits 0 and 8 from training set
# Ind08=np.array([ind for ind in range(numtrain) if ((trainLabels[ind]==Ci) or (trainLabels[ind]==Cj))])
# trainImg_08 = trainImgs[Ind08]
# trainLabels_08 = trainLabels[Ind08]
# #print("length of training set: ", len(Ind08))
# n_08 = len(Ind08)
# #choose only digits 0 and 8 from test set
# Ind08=np.array([ind for ind in range(10000) if ((testLabels[ind]==Ci) or (testLabels[ind]==Cj))])
# testImg_08 = testImgs[Ind08]
# testLabels_08 = testLabels[Ind08]
# #print("length of test set: ", len(Ind08))
# ntest_08 = len(Ind08)

# #train and test data for 0 and 8 - calculate error
# mdl_lda = LDA()
# mdl_lda.fit(trainImg_08, trainLabels_08)
# train_err_lda08 = 100*(np.count_nonzero(np.array(mdl_lda.predict(trainImg_08))-trainLabels_08)/n_08)
# test_err_lda08 = 100*(np.count_nonzero(np.array(mdl_lda.predict(testImg_08))-testLabels_08)/ntest_08)
# print("Digit 0 vs Digit 8: training error = %f%% , test error = %f%%" % (train_err_lda08, test_err_lda08))

# # digit 1 vs digit 7
# Ci = 1
# Cj = 7
# #choose only digits 1 and 7 from training set
# Ind17=np.array([ind for ind in range(numtrain) if ((trainLabels[ind]==Ci) or (trainLabels[ind]==Cj))])
# trainImg_17 = trainImgs[Ind17]
# trainLabels_17 = trainLabels[Ind17]
# #print("length of training set: ", len(Ind17))
# n_17 = len(Ind17)
# #choose only digits 1 and 7 from test set
# Ind17=np.array([ind for ind in range(10000) if ((testLabels[ind]==Ci) or (testLabels[ind]==Cj))])
# testImg_17 = testImgs[Ind17]
# testLabels_17 = testLabels[Ind17]
# #print("length of test set: ", len(Ind17))
# ntest_17 = len(Ind17)

# #train and test data for 1 and 7 - calculate error
# mdl_lda = LDA()
# mdl_lda.fit(trainImg_17, trainLabels_17)
# train_err_lda17 = 100*(np.count_nonzero(np.array(mdl_lda.predict(trainImg_17))-trainLabels_17)/n_17)
# test_err_lda17 = 100*(np.count_nonzero(np.array(mdl_lda.predict(testImg_17))-testLabels_17)/ntest_17)
# print("Digit 1 vs Digit 7: training error = %f%%, test error = %f%%" % (train_err_lda17, test_err_lda17))


##### (4) PCA for dimensionality reduction #######

#start by scaling the data (required for PCA)
scaler = MinMaxScaler()
scaler.fit(trainImgs) #fit the scaler on the training set
trainImgs_scaled = scaler.transform(trainImgs)
testImgs_scaled = scaler.transform(testImgs)

#apply PCA n_dimensions =2 and 3
pca_2 = PCA(n_components=2)
pca_2.fit(trainImgs_scaled)
trainImgs_pca_2 = pca_2.transform(trainImgs_scaled)
pca_3 = PCA(n_components=3)
pca_3.fit(trainImgs_scaled)
trainImgs_pca_3 = pca_3.transform(trainImgs_scaled)

### display all of the projected training data for n_dim = 2 and 3
plt.figure(figsize=(10,10))
plt.scatter(x=trainImgs_pca_2[:,0], y=trainImgs_pca_2[:,1], c=trainLabels, edgecolor='none',
                 cmap=plt.cm.get_cmap('Spectral',10))
plt.title("2D Scatter, PCA n_dimensions=2")
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.savefig("2D_Scatter.png")

fig=plt.figure()
ax = plt.axes(projection='3d')
p=ax.scatter3D(xs=trainImgs_pca_3[:,0], ys=trainImgs_pca_3[:,1], zs=trainImgs_pca_3[:,2], 
            c=trainLabels, edgecolor='none', alpha=.5, cmap=plt.cm.get_cmap('Spectral',10))
ax.set_title("3D Scatter, PCA n_dimensions=3")
ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
ax.set_zlabel('component 3')
fig.colorbar(p)
fig.savefig("3D_Scatter.png")


#use n_dimensions = 5,10,20,50,100 to classify with naive bayes and knn (N_neighbor = 5)
n_dimensions = [5,10,20,50,100]
nb_pca_train_err = []
nn_pca_train_err = []
nb_pca_test_err = []
nn_pca_test_err = []
mdl_nb_pca = GaussianNB()
mdl_knn_5 = KNeighborsClassifier(n_neighbors=5)

for i in n_dimensions:
    #fit and apply PCA
    pca = PCA(n_components=i)
    pca.fit(trainImgs_scaled)
    trainImgs_pca = pca.transform(trainImgs_scaled)
    testImgs_pca = pca.transform(testImgs_scaled)
    
    #classify with Naive Bayes, get train and test error rate
    test_pred_labels = mdl_nb_pca.fit(trainImgs_pca, trainLabels).predict(testImgs_pca)
    train_pred_labels = mdl_nb_pca.fit(trainImgs_pca, trainLabels).predict(trainImgs_pca)
    train_err_nb = ((trainLabels!=train_pred_labels).sum())/len(trainLabels)
    test_err_nb = ((testLabels!=test_pred_labels).sum())/len(testLabels)
    nb_pca_test_err.append(test_err_nb*100)
    nb_pca_train_err.append(train_err_nb*100)
    print("n_dim = %d with Naive Bayes: training error = %f, test error = %f" % (i, train_err_nb, test_err_nb))

    #classify with nearest neighbors, get train and test error rate
    mdl_knn_5.fit(trainImgs_pca, trainLabels)
    score = mdl_knn_5.score(testImgs_pca, testLabels)
    trainscore = mdl_knn_5.score(trainImgs_pca, trainLabels)
    train_err_nn = (1-trainscore)*100
    test_err_nn = (1-score)*100
    nn_pca_test_err.append(test_err_nn)
    nn_pca_train_err.append(train_err_nn)
    print("n_dim = %d with Nearest Neighbors (N_neighbors=5): training error = %f, test error = %f" % (i, train_err_nn, test_err_nn))
    

#plot training/test error vs N_neighbors 
fig1, (ax1, ax2) = plt.subplots(2)
ax1.plot(n_dimensions, nn_pca_test_err)
ax1.set_title('Test Error rate vs PCA n_dimensions with Nearest Neighbors')
ax1.set(xlabel = 'n_dimensions', ylabel = 'Test Error Rate')
ax2.plot(n_dimensions, nn_pca_train_err)
ax2.set_title('Train Error rate vs PCA n_dimensions with Nearest Neighbors')
ax2.set(xlabel = 'n_dimensions', ylabel = 'Train Error Rate')

fig2, (ax1, ax2) = plt.subplots(2)
ax1.plot(n_dimensions, nb_pca_test_err)
ax1.set_title('Test Error rate vs PCA n_dimensions with Naive Bayes')
ax1.set(xlabel = 'n_dimensions', ylabel = 'Test Error Rate')
ax2.plot(n_dimensions, nb_pca_train_err)
ax2.set_title('Train Error rate vs PCA n_dimensions with Naive Bayes')
ax2.set(xlabel = 'n_dimensions', ylabel = 'Train Error Rate')

plt.show()