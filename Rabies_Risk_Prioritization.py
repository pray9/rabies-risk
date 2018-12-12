import csv
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools

from matplotlib.pyplot import savefig
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors,cluster,model_selection
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel

risk_data = []

with open("Risk_Data.csv", "rt") as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    next(reader)
    for row in reader:
        for i in range(len(row)):
            try:
                row[i] = int(row[i])
            except:
                try:
                    row[i] = float(row[i])
                except:
                    pass
        risk_data.append(row)

labeled = []
unlabeled = []

for i in risk_data:
    if i[-1] != '':
        labeled.append(i)
    else:
        unlabeled.append(i)

def test_classifier(clf, features, n_splits=3):
    """
    Test basic classifier using Stratified K-fold feature selection. The best train/test
    split is used to create the confusion matrix
    """
    Y = np.asarray(labeled)
    folds = list(StratifiedKFold(n_splits, shuffle = True).split(np.asarray(labeled)[:,:-1], Y[:,-1]))
    scores = cross_val_score(clf, features, Y[:,-1], scoring="accuracy", cv=folds)
    print(np.average(scores))
    print(scores)
    total_Y = []
    total_Y_predict = []
    for i in range(n_splits):
        best_fold_train, best_fold_test = folds[i]
        clf.fit(features[best_fold_train], Y[best_fold_train, -1])
        for j in Y[best_fold_test, -1]:
            total_Y.append(j)
        Y_predict = clf.predict(features[best_fold_test])
        for k in Y_predict:
            total_Y_predict.append(k)
  
    cm = confusion_matrix(total_Y, total_Y_predict)
    plot_confusion_matrix(cm)
    print(classification_report(total_Y, total_Y_predict))
    new_predict = clf.predict(np.asarray(unlabeled)[:,:-1])
    print(new_predict)
    #return np.average(scores)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=['low','med','high'], normalize=True):
    """
    Method to visualize confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    savefig('sample.pdf')

data = np.asarray(labeled)[:,:-1]
test_classifier(GradientBoostingClassifier(n_estimators=200,learning_rate=0.5, max_depth=7), data)

#test_classifier(RandomForestClassifier(), data)
#test_classifier(neighbors.KNeighborsClassifier(n_neighbors = 10), data
#test_classifier(cluster.KMeans(n_clusters=3), data)
#test_classifier(ExtraTreesClassifier(), data)
#test_classifier(AdaBoostClassifier(n_estimators=150,learning_rate=0.3), data)
#test_classifier(DecisionTreeClassifier(), data)

