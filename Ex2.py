from sklearn.svm import SVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from MultiClassSVM import OneVsAllClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

DIR_PATH = "A3_Dataset/MNIST.csv"
data_file = pd.read_csv(DIR_PATH)
data_file = data_file.iloc[-4200:, :]
label = np.array(data_file['label'])
data = np.array(data_file.drop(['label'], 1))
# Use a threshold to binarize data
data[data <= 100] = -1
data[data > 100] = 1

data_train = data[:-840, :]
data_test = data[-840:, :]
label_train = label[:-840]
label_test = label[-840:]

print('**************** From Scratch MultiClass One-vs-Rest SVM with rbf kernel')

clf_OneVsAll = OneVsAllClassifier(SVC(kernel='rbf', C=10), n_classes=10)
clf_OneVsAll.fit(data_train, label_train)
y_test_predict2 = clf_OneVsAll.predict(data_test)
print('Accuracy rate of multiclass SVM in test set:'+str(accuracy_score(label_test, y_test_predict2)))
print("\nTest set confusion matrix:\n")
print(metrics.confusion_matrix(label_test, y_test_predict2))


# Specify range of Grid search
C_range = np.outer(np.logspace(-1, 0, 2), np.array([1, 5]))
C_range = C_range.flatten()  # flatten matrix, change to 1D numpy array
gamma_range = np.outer(np.logspace(-2, -1, 2), np.array([1, 5]))
gamma_range = gamma_range.flatten()
tuned_parameters = [{"estimator__C": C_range, "estimator__gamma": gamma_range}]
# Create a SVC SVM classifier object and tune it using GridSearchCV  function.
# Change the following line as appropriate
optimiser = GridSearchCV(OneVsRestClassifier(SVC(kernel='rbf')),
                         param_grid=[{"estimator__C": C_range, "estimator__gamma": gamma_range}],
                         n_jobs=-1,
                         verbose=2)
optimiser.fit(data_train, label_train)
predictions_svm = optimiser.predict(data_test)

# print the details of the best model and its accuracy
print('**************** MultiClass One-vs-Rest SVM with rbf kernel')

print("Best model:", optimiser)
print("Best params learned via GridSearch", optimiser.best_estimator_)
print("Accuracy of learned model", optimiser.score(data_test, label_test))
print(metrics.classification_report(label_test, predictions_svm))
print(metrics.confusion_matrix(label_test, predictions_svm))

print('**************** Binary One-vs-One SVM with rbf kernel')

clf = SVC(kernel="rbf", decision_function_shape='ovo', C=5, gamma=.01)
clf.fit(data_train, label_train)
predict_label = clf.predict(data_test)
print("Accuracy of learned model", clf.score(data_test, label_test))
print(metrics.classification_report(label_test, predict_label))
print(metrics.confusion_matrix(label_test, predict_label))
