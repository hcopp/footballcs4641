from pandas import read_csv
from numpy import set_printoptions
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def select_features(regressionType):
    data_file_name = './data/scaled.csv'
    data = read_csv(data_file_name)
    label_file_name = './data/labels.csv'
    labels = read_csv(label_file_name)
    X = data.values
    Y = labels['Yards'].values
    if regressionType == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
        fit = model.fit(X, Y)
        importance = model.feature_importances_
    elif regressionType == 'LinearRegression':
        model = LinearRegression()
        fit = model.fit(X, Y)
        importance = model.coef_
    elif regressionType == 'LogisticRegression':
        model = LogisticRegression()
        fit = model.fit(X, Y)
        importance = model.coef_[0]
    else:
        print('Regression Type Not Recognized...')
        return None
    # k_best_object = SelectKBest(f_regression, 10)
    # k_best = k_best_object.fit_transform(X, Y)
    plt.xticks(rotation=90)
    plt.bar([data.columns.values[x] for x in range(len(importance))], importance)
    plt.savefig(f'./graphs/{regressionType}.png')
    plt.show()

if __name__ == '__main__':
    select_features('DecisionTreeRegressor')
    select_features('LinearRegression')
    select_features('LogisticRegression')
