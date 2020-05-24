from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


def learning_curves(X_train, y_train, X_test, y_test):
    # Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . .
    # Create the figure window
    fig = plt.figure(figsize=(10, 8))
    # We will vary the training set size so that we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))
    # Create four different models based on max_depth
    for k, depth in enumerate([1, 3, 6, 10]):
        for i, s in enumerate(sizes):
            # Setup a decision tree regressor so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth=depth)
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])
            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))
        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k + 1)
        ax.plot(sizes, test_err, lw=2, label='Testing Error')
        ax.plot(sizes, train_err, lw=2, label='Training Error')
        ax.legend()
        ax.set_title('max_depth = %s' % depth)
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=18, y=1.03)
    fig.tight_layout()
    fig.show()


def model_complexity(X_train, y_train, X_test, y_test):
    # We will vary the max_depth of a decision tree model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))
    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)
        # Fit the learner to the training data
        regressor.fit(X_train, y_train)
        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))
        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, test_err, lw=2, label='Testing Error')
    plt.plot(max_depth, train_err, lw=2, label='Training Error')
    plt.legend()
    plt.xlabel('Maximum Depth')
    plt.ylabel('Total Error')
    plt.show()


def performance_metric(y_true, y_predict):
    return mean_squared_error(y_true, y_predict)


def load_data():
    train_data = np.loadtxt('A3_DataSet/fbtrain.csv', delimiter=',')
    X_train = train_data[:, 0:-1]
    y_train = train_data[:, -1]
    test_data = np.loadtxt('A3_DataSet/fbtest.csv', delimiter=',')

    # For Exercise 3) part 3)
    test_data = test_data.tolist()
    for i in list(test_data):
        if i[38] != 24:
            test_data.remove(i)
    test_data = np.asarray(test_data)
    # ! Exercise 3) part 3)

    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    return X_train, y_train, X_test, y_test


# Load Data
X_train, y_train, X_test, y_test = load_data()
# 'Read more at  : https://uksim.info/uksim2015/data/8713a015.pdf'

"""
   DecisionTree Regression
   """

# Params for Grid Search
tuned_params = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
# Cross validate and fine tune hyper parameters
optimiser = GridSearchCV(DecisionTreeRegressor(), tuned_params, n_jobs=-1)
optimiser.fit(X_train, y_train)
predictor = optimiser.predict(X_test)
# print the details of the best model and its accuracy
print('**************** DecisionTreeRegressor kernel')

print("Best params learned via GridSearch", optimiser.best_estimator_)
print("******* DecisionTreeRegressor r2_score: ", optimiser.score(X_test, y_test))
preds = optimiser.predict(X_train)
mse = np.sum((preds - y_train) ** 2) / len(preds)
print("Train MSE", mse)
print("Test MSE: ", np.sum((predictor - y_test) ** 2) / len(predictor))
learning_curves(X_train, y_train, X_test, y_test)
model_complexity(X_train, y_train, X_test, y_test)

"""
   Random Forest Regression
   """

parameters = [{'n_estimators': [20], 'criterion': ['mse'], 'min_weight_fraction_leaf': [0.25], 'n_jobs': [-1]}]
scoring_function = make_scorer(r2_score, greater_is_better=True)
# Make the GridSearchCV object
optimiser = GridSearchCV(RandomForestRegressor(n_estimators=100, n_jobs=-1), parameters, scoring=scoring_function, cv=10)
optimiser = optimiser.fit(X_train, y_train)
predictor = optimiser.predict(X_test)
print("****** RandomForestRegressor r2_score = ", optimiser.score(X_test, y_test))
preds = optimiser.predict(X_train)
mse = np.sum((preds - y_train) ** 2) / len(preds)
print("Train MSE", mse)
print("Test MSE: ", np.sum((predictor - y_test) ** 2) / len(predictor))
