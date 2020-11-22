import sys
import pandas as pd
from sklearn import tree

# Tolerance is the # of yards we can be off (+ & -)
tolerance = 3

labels = pd.read_csv('./data/labels.csv')
data = pd.read_csv('./data/scaled.csv')

train_data = data[:24805]
train_labels = labels[:24805]['Yards']

test_data = data[24805:]
test_labels = labels[24805:]['Yards']

classifier = tree.DecisionTreeClassifier()
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

print('Results\nTolerance:', tolerance, 'yards\nCorrect:', correct, 'plays\nTotal:', i, 'plays\nPercentage:', correct / i)
