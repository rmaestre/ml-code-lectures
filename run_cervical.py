import matplotlib.pyplot as plt
import pprint
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

from ml.datasets.cervical import Cervical
from ml.models.random_forest import RandomForest
from ml.metrics.classification import ClassificationMetrics


X_train, y_train, X_test, y_test = Cervical().generate()


# Init classification tree model
rf = RandomForest(n_trees=1, max_depth=4, batch_size=400, is_classification=True)

# Fit model
rf.fit(X_train, y_train)
# Get predictions
y_hats = rf.predict(X_test)
# get accuracy
metrics = ClassificationMetrics(y_test, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_test, y_hats))

# Oversampling
X_undersampling, y_undersampling = resample(
    X_train[y_train == 1],
    y_train[y_train == 1],
    replace=True,
    n_samples=X_train[y_train == 0].shape[0],
    random_state=123,
)
X_undersampling = np.vstack((X_train, X_undersampling))
y_undersampling = np.concatenate((y_train, y_undersampling))

rf = RandomForest(n_trees=10, max_depth=6, batch_size=500, is_classification=True)
# Fit model
rf.fit(X_undersampling, y_undersampling)
# Get predictions
y_hats = rf.predict(X_train)
# get accuracy
metrics = ClassificationMetrics(y_train, y_hats)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y_train, y_hats))

"""
# calculate feature importance
importances = rf.feature_importance(X_train, y_train)
pprint.pprint(importances)
# Plot bars
plt.bar(importances.keys(), importances.values())
plt.show()
"""

from sklearn.ensemble import RandomForestClassifier

# Train a random forest model
clf = RandomForestClassifier(
    n_estimators=10,
)
clf.fit(X_undersampling, y_undersampling)
print(confusion_matrix(y_train, clf.predict(X_train)))


a = 1 + 1
