# pyright: reportMissingImports=false

# <SETUP>
# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

# Import libraries
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
# <END SETUP>


# <LOAD DATA>
# Load facial landmark data (5-point or 68-point) for each identity
X_cal = np.load("X-68-Caltech.npy")  # Facial landmarks for each sample (num_samples x num_points x 2)
Y_cal = np.load("y-68-Caltech.npy")  # Identity labels
num_samples_cal = Y_cal.shape[0]

X_sof = np.load("X-68-SoF.npy")
Y_sof = np.load("y-68-SoF.npy")
num_samples_sof = Y_sof[0]
# <END LOAD DATA>


# <FEATURE EXTRACTION>
# Compute pairwise Euclidean distances between all landmark points for each identity
features_cal = []
for landmarks in X_cal:  # landmarks: (num_points x 2)
    distances_cal = []
    for i in range(landmarks.shape[0]):
        for j in range(landmarks.shape[0]):
            p1, p2 = landmarks[i], landmarks[j]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances_cal.append(dist)
    features_cal.append(distances_cal)

features_sof = []
for landmarks in X_sof:  # landmarks: (num_points x 2)
    distances_sof = []
    for i in range(landmarks.shape[0]):
        for j in range(landmarks.shape[0]):
            p1, p2 = landmarks[i], landmarks[j]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances_sof.append(dist)
    features_sof.append(distances_sof)

features_cal = np.array(features_cal)
features_sof = np.array(features_sof)
# <END FEATURE EXTRACTION>


# <CLASSIFICATION>
clf = KNeighborsClassifier(n_neighbors=13)  # Initialize k-NN classifier
clf_GNB = GaussianNB() #Initialize GaussianNB classifier

knn_correct = knn_incorrect = 0
gnb_correct = gnb__incorrect = 0

# Leave-One-Out Evaluation

#CalTech Dataset
for i in range(len(Y_cal)):
    query_X_cal = features_cal[i]
    query_y_cal = Y_cal[i]

    # Remove query from the dataset to serve as template set
    template_X_cal = np.delete(features_cal, i, axis=0)
    template_y_cal = np.delete(Y_cal, i)

    # Create binary labels: 1 = genuine, 0 = impostor
    binary_labels = (template_y_cal == query_y_cal).astype(int)

    # Train classifier
    clf.fit(template_X_cal, binary_labels)
    clf_GNB.fit(template_X_cal, binary_labels)

    # Predict if query is genuine (1) or impostor (0)
    knn_pred = clf.predict(query_X_cal.reshape(1, -1))[0]
    gnb_pred = clf_GNB.predict(query_X_cal.reshape(1,-1))[0]

    # Update counters

    #KNN Classifier
    if knn_pred == 1:
        knn_correct += 1
    else:
        knn_incorrect += 1

    #GNB Classifier
    if gnb_pred == 1:
        gnb_correct += 1
    else:
        gnb__incorrect += 1

#SoF Dataset
for i in range(Y_sof):
    query_X_sof = features_sof[i]
    query_Y_sof = Y_sof[i]

    
# <END CLASSIFICATION>


# <RESULTS>

#KNN
knn_accuracy = knn_correct / (knn_correct + knn_incorrect)
print("\nKNN Classifier Results:\nNum correct = %d, Num incorrect = %d, Accuracy = %.2f" %
      (knn_correct, knn_incorrect, knn_accuracy))

#GNB
gnb_accuracy = gnb_correct / (gnb_correct + gnb__incorrect)
print("\nGNB Classifier Results:\nNum correct = %d, Num incorrect = %d, Accuracy = %.2f" %
      (gnb_correct, gnb__incorrect, gnb_accuracy))
# <END RESULTS>
