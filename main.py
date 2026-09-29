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
from sklearn.neighbors import KNeighborsClassifier , NearestCentroid
from sklearn.linear_model import SGDClassifier, RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
for landmarks in X_sof:
    distances_sof = []
    for i in range(landmarks.shape[0]):
        for j in range(landmarks.shape[0]):
            p1, p2 = landmarks[i], landmarks[j]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances_sof.append(dist)
    features_sof.append(distances_sof)

features_cal = np.array(features_cal)
features_sof = np.array(features_sof)


def normalized_landmark_coordinates(landmarks):
    centered = landmarks - np.mean(landmarks, axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).max()
    return (centered / scale).ravel()


def normalized_pairwise_distances(landmarks):
    distances = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :], axis=2)
    face_scale = np.linalg.norm(np.ptp(landmarks, axis=0))
    return (distances / face_scale).ravel()


features_cal_normalized = np.array([
    normalized_landmark_coordinates(landmarks) for landmarks in X_cal
])
features_sof_ratios = np.array([
    normalized_pairwise_distances(landmarks) for landmarks in X_sof
])
# <END FEATURE EXTRACTION>


# <CLASSIFICATION>
clf = KNeighborsClassifier(n_neighbors=13)  # Initialize k-NN classifier
clf_GNB = GaussianNB() #Initialize GaussianNB classifier
clf_nc = NearestCentroid() 
clf_SGD = make_pipeline(
    StandardScaler(),
    SGDClassifier(class_weight="balanced", random_state=42, max_iter=5000)
)
clf_ridgecv = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0, 100.0))


def evaluate_identity_classification(estimator, features, labels):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.empty_like(labels)

    for train_indices, test_indices in splitter.split(features, labels):
        model = clone(estimator)
        model.fit(features[train_indices], labels[train_indices])
        predictions[test_indices] = model.predict(features[test_indices])

    return (
        accuracy_score(labels, predictions),
        balanced_accuracy_score(labels, predictions),
    )


def evaluate_verification(estimator, features, labels, gallery_indices, probe_indices):
    true_labels = []
    predicted_labels = []

    for identity in np.unique(labels):
        gallery_labels = (labels[gallery_indices] == identity).astype(int)
        model = clone(estimator)
        model.fit(features[gallery_indices], gallery_labels)
        predictions = model.predict(features[probe_indices])

        true_labels.extend((labels[probe_indices] == identity).astype(int))
        predicted_labels.extend(predictions)

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    genuine_mask = true_labels == 1
    impostor_mask = true_labels == 0
    genuine_acceptance = np.mean(predicted_labels[genuine_mask] == 1)
    impostor_rejection = np.mean(predicted_labels[impostor_mask] == 0)

    return {
        "balanced_accuracy": (genuine_acceptance + impostor_rejection) / 2,
        "genuine_acceptance": genuine_acceptance,
        "impostor_rejection": impostor_rejection,
        "false_acceptance": 1 - impostor_rejection,
        "false_rejection": 1 - genuine_acceptance,
    }


cal_gallery_indices, cal_probe_indices = train_test_split(
    np.arange(len(Y_cal)), test_size=0.3, random_state=42, stratify=Y_cal
)
sof_gallery_indices, sof_probe_indices = train_test_split(
    np.arange(len(Y_sof)), test_size=0.3, random_state=42, stratify=Y_sof
)

cal_evaluations = {
    "kNN / raw distances": (clf, features_cal),
    "GaussianNB / raw distances": (clf_GNB, features_cal),
    "RidgeCV / raw distances": (clf_ridgecv, features_cal),
    "RidgeCV / normalized coordinates": (clf_ridgecv, features_cal_normalized),
}
sof_evaluations = {
    "NearestCentroid / raw distances": (clf_nc, features_sof),
    "SGD / raw distances": (clf_SGD, features_sof),
    "SGD / normalized distance ratios": (clf_SGD, features_sof_ratios),
}

cal_metrics = {}
for name, (estimator, features) in cal_evaluations.items():
    cal_metrics[name] = {
        "identity": evaluate_identity_classification(estimator, features, Y_cal),
        "verification": evaluate_verification(
            estimator, features, Y_cal, cal_gallery_indices, cal_probe_indices
        ),
    }

sof_metrics = {}
for name, (estimator, features) in sof_evaluations.items():
    sof_metrics[name] = {
        "identity": evaluate_identity_classification(estimator, features, Y_sof),
        "verification": evaluate_verification(
            estimator, features, Y_sof, sof_gallery_indices, sof_probe_indices
        ),
    }

knn_correct = knn_incorrect = 0
gnb_correct = gnb__incorrect = 0
nc_correct = nc_incorrect = 0
sgd_correct = sgd_incorrect = 0
ridgecv_correct = ridgecv_incorrect = 0
cal_normalized_correct = cal_normalized_incorrect = 0
sof_ratios_correct = sof_ratios_incorrect = 0

# Leave-One-Out Evaluation

#CalTech Dataset
for i in range(len(Y_cal)):
    query_X_cal = features_cal[i]
    query_y_cal = Y_cal[i]

    # Remove query from the dataset to serve as template set
    template_X_cal = np.delete(features_cal, i, axis=0)
    template_y_cal = np.delete(Y_cal, i)

    # Create binary labels: 1 = genuine, 0 = impostor
    binary_labels_cal = (template_y_cal == query_y_cal).astype(int)

    # Train classifier
    clf.fit(template_X_cal, binary_labels_cal)
    clf_GNB.fit(template_X_cal, binary_labels_cal)
    clf_ridgecv.fit(template_X_cal, binary_labels_cal)
    ridgecv_pred = clf_ridgecv.predict(query_X_cal.reshape(1,-1))[0]

    template_X_cal_normalized = np.delete(features_cal_normalized, i, axis=0)
    query_X_cal_normalized = features_cal_normalized[i]
    clf_ridgecv.fit(template_X_cal_normalized, binary_labels_cal)
    normalized_pred = clf_ridgecv.predict(query_X_cal_normalized.reshape(1, -1))[0]

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

    #RidgeCV Classifier
    if ridgecv_pred == 1:
        ridgecv_correct += 1
    else:
        ridgecv_incorrect += 1

    if normalized_pred == 1:
        cal_normalized_correct += 1
    else:
        cal_normalized_incorrect += 1

#SoF Dataset
for i in range(len(Y_sof)):
    query_X_sof = features_sof[i]
    query_Y_sof = Y_sof[i]

    template_X_sof = np.delete(features_sof, i, axis = 0)
    template_y_sof = np.delete(Y_sof, i)

    binary_labels_sof = (template_y_sof == query_Y_sof).astype(int)

    clf_nc.fit(template_X_sof, binary_labels_sof)
    clf_SGD.fit(template_X_sof, binary_labels_sof)
    sgd_pred = clf_SGD.predict(query_X_sof.reshape(1, -1))[0]

    template_X_sof_ratios = np.delete(features_sof_ratios, i, axis=0)
    clf_SGD.fit(template_X_sof_ratios, binary_labels_sof)

    nc_pred = clf_nc.predict(query_X_sof.reshape(1, -1))[0]
    sof_ratios_pred = clf_SGD.predict(features_sof_ratios[i].reshape(1, -1))[0]

    if nc_pred == 1:
        nc_correct += 1
    else:
        nc_incorrect += 1

    if sgd_pred == 1:
        sgd_correct += 1
    else:
        sgd_incorrect += 1

    if sof_ratios_pred == 1:
        sof_ratios_correct += 1
    else:
        sof_ratios_incorrect += 1
    
# <END CLASSIFICATION>


# <RESULTS>
print("CalTech Dataset Results:")
#KNN
knn_accuracy = knn_correct / (knn_correct + knn_incorrect)
print("\nKNN Classifier Results:\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
      (knn_correct, knn_incorrect, knn_accuracy))

#GNB
gnb_accuracy = gnb_correct / (gnb_correct + gnb__incorrect)
print("\nGNB Classifier Results:\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
      (gnb_correct, gnb__incorrect, gnb_accuracy))

#RidgeCV
ridgecv_accuracy = ridgecv_correct / (ridgecv_correct + ridgecv_incorrect)
print("\nRidgeCV Classifier Results (raw distances):\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
      (ridgecv_correct, ridgecv_incorrect, ridgecv_accuracy))

cal_normalized_rate = cal_normalized_correct / (cal_normalized_correct + cal_normalized_incorrect)
print("\nRidgeCV with normalized coordinates:\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
    (cal_normalized_correct, cal_normalized_incorrect, cal_normalized_rate))

print("\nSoF Dataset Results:")
#nc
nc_accuracy = nc_correct / (nc_correct + nc_incorrect)
print("\nNC Classifier Results:\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
      (nc_correct, nc_incorrect, nc_accuracy))

#sgd
sgd_accuracy = sgd_correct / (sgd_correct + sgd_incorrect)
print("\nSGD Classifier Results (raw distances):\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
      (sgd_correct, sgd_incorrect, sgd_accuracy))

sof_ratios_rate = sof_ratios_correct / (sof_ratios_correct + sof_ratios_incorrect)
print("\nSGD with normalized distance ratios:\nNum accepted = %d, Num rejected = %d, Genuine acceptance = %.2f" %
    (sof_ratios_correct, sof_ratios_incorrect, sof_ratios_rate))


def print_evaluation_results(dataset_name, evaluations):
    print(f"\n{dataset_name} 5-fold identity classification and gallery/probe verification:")
    for name, results in evaluations.items():
        identity_accuracy, identity_balanced_accuracy = results["identity"]
        verification = results["verification"]
        print(
            f"{name}: identity accuracy={identity_accuracy:.3f}\n"
            f"identity balanced accuracy={identity_balanced_accuracy:.3f}\n"
            f"verification balanced accuracy={verification['balanced_accuracy']:.3f}\n"
            f"genuine acceptance={verification['genuine_acceptance']:.3f}\n"
            f"impostor rejection={verification['impostor_rejection']:.3f}\n"
            f"FAR={verification['false_acceptance']:.3f}\n"
            f"FRR={verification['false_rejection']:.3f}\n"
        )


print_evaluation_results("Caltech", cal_metrics)
print_evaluation_results("SoF", sof_metrics)
# <END RESULTS>