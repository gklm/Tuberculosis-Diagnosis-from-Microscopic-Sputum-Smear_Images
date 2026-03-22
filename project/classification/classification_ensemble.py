# Import necessary libraries
import cv2
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

# Function to load and preprocess images
def load_and_preprocess(folder, img_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img.flatten())
    return np.array(images)

# Function to create labels based on the number of images
def create_labels(num_class1, num_class2):
    return np.concatenate([np.zeros(num_class1), np.ones(num_class2)])

# Function to train a classifier and return accuracy
def train_and_evaluate_classifier(X, y, classifier, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return classifier, accuracy

# Define paths for datasets (Users should replace these with their own paths)
dataset_paths = {
    "svm": {"class1": "path/to/svm/Bacilli", "class2": "path/to/svm/NonBacilli"},
    "rf": {"class1": "path/to/rf/Bacilli", "class2": "path/to/rf/NonBacilli"},
    "xgb": {"class1": "path/to/xgb/Bacilli", "class2": "path/to/xgb/NonBacilli"}
}

# Train SVM classifier
svm_class1_images = load_and_preprocess(dataset_paths["svm"]["class1"])
svm_class2_images = load_and_preprocess(dataset_paths["svm"]["class2"])
svm_labels = create_labels(len(svm_class1_images), len(svm_class2_images))
clf_svm, acc_svm = train_and_evaluate_classifier(
    np.vstack([svm_class1_images, svm_class2_images]), svm_labels, svm.SVC()
)
print("SVM Accuracy:", acc_svm)

# Train Random Forest classifier
rf_class1_images = load_and_preprocess(dataset_paths["rf"]["class1"])
rf_class2_images = load_and_preprocess(dataset_paths["rf"]["class2"])
rf_labels = create_labels(len(rf_class1_images), len(rf_class2_images))
clf_rf, acc_rf = train_and_evaluate_classifier(
    np.vstack([rf_class1_images, rf_class2_images]), rf_labels, RandomForestClassifier()
)
print("Random Forest Accuracy:", acc_rf)

# Train XGBoost classifier
xgb_class1_images = load_and_preprocess(dataset_paths["xgb"]["class1"])
xgb_class2_images = load_and_preprocess(dataset_paths["xgb"]["class2"])
xgb_labels = create_labels(len(xgb_class1_images), len(xgb_class2_images))
clf_xgb, acc_xgb = train_and_evaluate_classifier(
    np.vstack([xgb_class1_images, xgb_class2_images]), xgb_labels, XGBClassifier()
)
print("XGBoost Accuracy:", acc_xgb)

# Create an ensemble classifier using VotingClassifier
ensemble_clf = VotingClassifier(
    estimators=[("SVM", clf_svm), ("RandomForest", clf_rf), ("XGBoost", clf_xgb)], voting="hard"
)
ensemble_clf.fit(
    np.vstack([svm_class1_images, rf_class1_images, xgb_class1_images]),
    np.concatenate([svm_labels, rf_labels, xgb_labels])
)

# Evaluate the ensemble classifier
ensemble_accuracy = ensemble_clf.score(
    np.vstack([svm_class2_images, rf_class2_images, xgb_class2_images]),
    np.concatenate([svm_labels, rf_labels, xgb_labels])
)
print("Ensemble Classifier Accuracy:", ensemble_accuracy)
