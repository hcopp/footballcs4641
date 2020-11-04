from pandas import read_csv
from numpy import set_printoptions
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

data_file_name = 'scaled.csv'
data = read_csv(data_file_name)
label_file_name = 'labels.csv'
labels = read_csv(label_file_name)
X = data.values[:, 1:]
Y = labels['playResult'].values

model = LinearRegression()
fit = model.fit(X, Y)
new_features = fit.transform(X)
importance = model.coef_
kBestObject = SelectKBest(f_regression, 10)
kBest = kBestObject.fit_transform(X, Y)
print(kBest[:3, :])
print(importance)
# plt.bar([x for x in range(len(importance))], importance)
# plt.savefig('LinearRegression.png')
# plt.show()
# set_printoptions(precision=3)
# print(fit.scores_)
