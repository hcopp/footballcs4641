import sys
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from pandas import DataFrame
import csv

# Tolerance is the # of yards we can be off (+ & -)
#tolerance = 0

labels = pd.read_csv('./data/labels.csv')
data = pd.read_csv('./data/scaled.csv')

train_data = data[:24805]
train_labels = labels[:24805]['Yards']

test_data = data[24805:]
test_labels = labels[24805:]['Yards']

results = {}
for tolerance in range(0,4):#0,4
    for est in range(18,22):#0,101
        print('\nClassifying w/', est * 5, 'estimators')
        classifier = ensemble.RandomForestClassifier(n_estimators = est * 5)
        classifier.fit(train_data, train_labels)

        correct = 0
        incorrect = 0

        i = 0

        for index, rowdata in test_data.iterrows():
            val = test_labels[index]
            prediction = classifier.predict([rowdata])[0]
            if abs(prediction - val) <= tolerance:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
            i += 1
        results[est] = {
            'tolerance': tolerance,
            'correct': correct,
            'total': i,
            'percentage': correct / i,
            'estimators': est * 5,
        }
        print('Results\nTolerance:', tolerance, 'yards\nCorrect:', correct, 'plays\nTotal:', i, 'plays\nPercentage:', correct / i)

with open('./data/random_forest_results.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results.items():
       writer.writerow([key, value])
