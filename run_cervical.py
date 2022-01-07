import matplotlib.pyplot as plt
import numpy as np
import pprint

from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from ml.datasets.cervical import Cervical
from ml.models.random_forest import RandomForest

from ml.metrics.classification import ClassificationMetrics


# Get data
X_train, y_train, X_test, y_test = Cervical().generate()

# Init classification random forest
# =================================
rf = RandomForest(
    n_trees=10,
    max_depth=100,
    batch_size=250,
    # max_features=10,
    is_classification=True,
)
# Fit model
rf.fit(X_train, y_train)
# Get predictions
y_hats = rf.predict(X_test)
# get accuracy
metrics = ClassificationMetrics(y_test, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_test, y_hats))
print(classification_report(y_hats, y_test))

# Predict probability
# =================================
prob_threshold = 0.1
y_hats = rf.predict(X_test, predict_proba=True) > prob_threshold
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
clf.fit(X_train, y_train)
y_hats = clf.predict(X_test)
metrics = ClassificationMetrics(y_test, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_test, y_hats))
print(classification_report(y_hats, y_test))
