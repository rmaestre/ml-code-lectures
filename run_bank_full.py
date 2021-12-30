import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from ml.datasets.bank import Bank
from ml.models.random_forest import RandomForest
from ml.models.decision_tree import DecisionTree

from ml.metrics.classification import ClassificationMetrics


# Get data
X_train, y_train, X_test, y_test = Bank().generate()

# Oversampling
X_undersampling, y_undersampling = resample(
    X_train[y_train == 1], y_train[y_train == 1], replace=True, n_samples=150
)
ind_falses = np.random.choice(np.where(y_train == 0)[0], 150)
X_undersampling = np.vstack((X_train[ind_falses], X_undersampling))
y_undersampling = np.concatenate((y_train[ind_falses], y_undersampling))


# Init classification tree model
# ==============================
c_tree = DecisionTree(is_classification=True, max_depth=100)
# Fit model
c_tree.fit(X_undersampling, y_undersampling)
# Get predictions
y_hats = c_tree.predict(X_test)
# get accuracy
metrics = ClassificationMetrics(y_test, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_test, y_hats))
print(classification_report(y_hats, y_test))


# Init classification random forest
# =================================
rf = RandomForest(
    n_trees=10,
    max_depth=100,
    batch_size=250,
    is_classification=True,
)
# Fit model
rf.fit(X_undersampling, y_undersampling)
# Get predictions
y_hats = rf.predict(X_test)
# get accuracy
metrics = ClassificationMetrics(y_test, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_test, y_hats))
print(classification_report(y_hats, y_test))


# Init classification random forest (sklearn)
# ===========================================
clf = RandomForestClassifier(
    n_estimators=10,
)
clf.fit(X_undersampling, y_undersampling)
y_hats = clf.predict(X_test)
metrics = ClassificationMetrics(y_test, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_test, y_hats))
print(classification_report(y_hats, y_test))
