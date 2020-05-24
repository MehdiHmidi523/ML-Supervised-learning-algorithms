from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from ANOVA import ANOVA


def _compute(sigma, degree):
    anovaKernel = np.zeros((X_s.shape[0], X_s.shape[0]))
    for d in range(X_s.shape[1]):
        column_1 = X_s[:, d].reshape(-1, 1)
        column_2 = X_s[:, d].reshape(-1, 1)
        anovaKernel += np.exp(-sigma * (column_1 - column_2.T) ** 2) ** degree
    return anovaKernel


print('$$$$$ Fetch Data ...')
MNIST_ = np.loadtxt('A3_DataSet/mnistsub.csv', delimiter=',')
X = MNIST_[:, 0:-1]
y = MNIST_[:, -1]
print('$$$$$ Normalize ...')
X = np.divide(X, np.amax(X) - np.amin(X))
print('$$$$$ Random Permutation ...')
np.random.seed(100)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]
X_s, y_s = X[:640, :], y[:640]

c_param = [.1, 1, 10, 100, 1000, 10000]
kernels = [{'kernel': ['linear'], 'C': c_param},
           {'kernel': ['rbf'], 'C': c_param, 'gamma': c_param},
           {'kernel': ['poly'], 'C': c_param, 'degree': [2, 3, 4, 5]}]

print("\n ['ANOVA Using gram matrix'] »»»»»»»»»»»»»»»»»»")
for d in [2, 3, 4, 5]:
    for s in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for c in [.1, 1, 10, 100, 1000]:
            gram_train = _compute(s, d)
            clf = SVC(kernel='precomputed', C=c)
            clf.fit(gram_train, y_s)
            print("degree: "+str(d)+" sigma:"+str(s)+" C:"+str(c)+" Accuracy:"+str(clf.score(gram_train, y_s)))

# Anova Kernel passed as a Function to SVC
print("\n ['ANOVA Using python Function'] »»»»»»»»»»»»»»»»»»")
grid = SVC(kernel=ANOVA(), C=.025)
for c in [.1, 1, 10, 100, 1000]:
    grid = SVC(kernel=ANOVA(), C=c)
    grid.fit(X_s, y_s)
print("kernel best parameter C: " + str(c))
print("score_train: " + str(grid.score(X_s, y_s)) + "\n")

# Grid Search Cross Validation
for p in kernels:
    grid = GridSearchCV(SVC(), p, cv=5, refit=True, n_jobs=-1)
    grid.fit(X_s, y_s)
    print("\n " + str(p.get('kernel')) + ' »»»»»»»»»»»»»»»»»»\n')
    print("GridSearchCV parameters ", p)
    print("Best score: " + str(abs(grid.best_score_)))
    print("Best Parameters: " + str(grid.best_params_))

    X1 = X_s[grid.best_estimator_.support_, 0]
    X2 = X_s[grid.best_estimator_.support_, 1]
    xx, yy = np.meshgrid(np.arange(X1.min() - 0.1, X1.max() + 0.1, .01),
                         np.arange(X2.min() - 0.1, X2.max() + 0.1, .01))
    fig = plt.figure()
    title = "Accuracy " + str(round(abs(grid.best_score_), 5)) + " "
    for key in grid.best_params_:
        title = title + key + ":" + str(grid.best_params_[key]) + " "
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)
    map_ = grid.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
    map_ = map_.reshape(xx.shape)  # Map of predictions
    ax.contour(xx, yy, map_, colors='b', linewidths=0.5)  # Show the boundary
    ax.scatter(X_s[:, 0], X_s[:, 1], s=.5, c=y_s)
    plt.show()
