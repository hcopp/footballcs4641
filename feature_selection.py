from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def select_features(regression_type):
    data_file_name = './data/scaled.csv'
    data = read_csv(data_file_name)
    label_file_name = './data/labels.csv'
    labels = read_csv(label_file_name)
    x = data.values
    y = labels['Yards'].values
    if regression_type == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
        model.fit(x, y)
        importance = model.feature_importances_
    elif regression_type == 'LinearRegression':
        model = LinearRegression()
        model.fit(x, y)
        importance = model.coef_
    elif regression_type == 'LogisticRegression':
        model = LogisticRegression()
        model.fit(x, y)
        importance = model.coef_[0]
    else:
        print('Regression Type Not Recognized...')
        return None
    plt.xticks(rotation=90)
    plt.bar([data.columns.values[x] for x in range(len(importance))], importance)
    plt.savefig(f'./graphs/{regression_type}.png')
    plt.show()


if __name__ == '__main__':
    select_features('DecisionTreeRegressor')
    select_features('LinearRegression')
    select_features('LogisticRegression')
