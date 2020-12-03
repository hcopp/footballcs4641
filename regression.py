import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

class Regression:

    def __init__(self):
        self.data = pd.read_csv('./data/scaled.csv')
        self.labels = pd.read_csv('./data/labels.csv')['Yards']
        self.x_train = self.data.values[:int(0.8 * len(self.data))]
        self.y_train = self.labels.values[:int(0.8 * len(self.data))]
        self.x_test = self.data.values[int(0.8 * len(self.data)):]
        self.y_test = self.labels.values[int(0.8 * len(self.data)):]
        self.big_line = "------------------------------------------------"

    def decision_tree_regression(self):
        dtr = DecisionTreeRegressor()
        dtr.fit(self.x_train, self.y_train)
        y_pred_test = dtr.predict(self.x_test)
        print('Decision Tree Regression')
        print(self.big_line)
        print('Mean squared error (MSE): %.2f'
              % mean_squared_error(self.y_test, y_pred_test))
        print('Mean absolute error (MSE): %.2f'
              % mean_absolute_error(self.y_test, y_pred_test))
        print('Coefficient of determination (R^2): %.2f'
              % r2_score(self.y_test, y_pred_test))
        total_correct = 0
        for i in range(len(y_pred_test)):
            if abs(y_pred_test[i] - self.y_test[i]) <= 1:
                total_correct += 1
        print(f'Percent Correct within 1 Yard: {total_correct / len(y_pred_test) * 100}%')
        print()

    def svr(self):
        svr_model = svm.SVR(gamma='scale')
        svr_model.fit(self.x_train, self.y_train)
        y_pred_test = svr_model.predict(self.x_test)
        print('SVR')
        print(self.big_line)
        print('Mean squared error (MSE): %.2f'
              % mean_squared_error(self.y_test, y_pred_test))
        print('Mean absolute error (MSE): %.2f'
              % mean_absolute_error(self.y_test, y_pred_test))
        print('Coefficient of determination (R^2): %.2f'
              % r2_score(self.y_test, y_pred_test))
        total_correct = 0
        for i in range(len(y_pred_test)):
            if abs(y_pred_test[i] - self.y_test[i]) <= 1:
                total_correct += 1
        print(f'Percent Correct within 1 Yard: {total_correct / len(y_pred_test) * 100}%')
        print()

    def linear_regression(self):
        linear_model = LinearRegression()
        linear_model.fit(self.x_train, self.y_train)
        y_pred_test = linear_model.predict(self.x_test)
        print('Linear Regression')
        print(self.big_line)
        print('Mean squared error (MSE): %.2f'
              % mean_squared_error(self.y_test, y_pred_test))
        print('Mean absolute error (MSE): %.2f'
              % mean_absolute_error(self.y_test, y_pred_test))
        print('Coefficient of determination (R^2): %.2f'
              % r2_score(self.y_test, y_pred_test))
        total_correct = 0
        for i in range(len(y_pred_test)):
            if abs(y_pred_test[i] - self.y_test[i]) <= 1:
                total_correct += 1
        print(f'Percent Correct within 1 Yard: {total_correct/len(y_pred_test) * 100}%')
        print()

    def poly_regression(self):
        poly_model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                               ('linear', LinearRegression(fit_intercept=False))])
        poly_model.fit(self.x_train, self.y_train)
        y_pred_test = poly_model.predict(self.x_test)
        print('Polynomial Regression')
        print(self.big_line)
        print('Mean squared error (MSE): %.2f'
              % mean_squared_error(self.y_test, y_pred_test))
        print('Mean absolute error (MSE): %.2f'
              % mean_absolute_error(self.y_test, y_pred_test))
        print('Coefficient of determination (R^2): %.2f'
              % r2_score(self.y_test, y_pred_test))
        total_correct = 0
        for i in range(len(y_pred_test)):
            if abs(y_pred_test[i] - self.y_test[i]) <= 1:
                total_correct += 1
        print(f'Percent Correct within 1 Yard: {total_correct/len(y_pred_test) * 100}%')
        print()

    def ridge(self):
        ridge_model = linear_model.Ridge(alpha=.5)
        ridge_model.fit(self.x_train, self.y_train)
        y_pred_test = ridge_model.predict(self.x_test)
        print('Ridge Regression')
        print(self.big_line)
        print('Mean squared error (MSE): %.2f'
              % mean_squared_error(self.y_test, y_pred_test))
        print('Mean absolute error (MSE): %.2f'
              % mean_absolute_error(self.y_test, y_pred_test))
        print('Coefficient of determination (R^2): %.2f'
              % r2_score(self.y_test, y_pred_test))
        total_correct = 0
        for i in range(len(y_pred_test)):
            if abs(y_pred_test[i] - self.y_test[i]) <= 1:
                total_correct += 1
        print(f'Percent Correct within 1 Yard: {total_correct / len(y_pred_test) * 100}%')
        print()

    def lasso(self):
        lasso_model = linear_model.Lasso(alpha=.5)
        lasso_model.fit(self.x_train, self.y_train)
        y_pred_test = lasso_model.predict(self.x_test)
        print('Lasso Regression')
        print(self.big_line)
        print('Mean squared error (MSE): %.2f'
              % mean_squared_error(self.y_test, y_pred_test))
        print('Mean absolute error (MSE): %.2f'
              % mean_absolute_error(self.y_test, y_pred_test))
        print('Coefficient of determination (R^2): %.2f'
              % r2_score(self.y_test, y_pred_test))
        total_correct = 0
        for i in range(len(y_pred_test)):
            if abs(y_pred_test[i] - self.y_test[i]) <= 1:
                total_correct += 1
        print(f'Percent Correct within 1 Yard: {total_correct / len(y_pred_test) * 100}%')
        print()

    def all_regressions(self):
        self.decision_tree_regression()
        self.svr()
        self.linear_regression()
        self.poly_regression()
        self.ridge()
        self.lasso()

if __name__ == '__main__':
    Regression().all_regressions()

