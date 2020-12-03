from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree


labels = pd.read_csv('./data/labels.csv')
data = pd.read_csv('./data/scaled.csv')

train_data = data[:24805]
train_labels = labels[:24805]['Yards']

test_data = data[24805:]
test_labels = labels[24805:]['Yards']

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_data, train_labels)
print(f'Accuracy: {round(classifier.score(test_data, test_labels)*100, 4)}%')

with open('treeText.txt','w') as file:
    file.write(tree.export_text(classifier, feature_names=list(data.columns)))

main_splits = []
with open('treeText.txt') as file:
    for line in file:
        if line.count('|') <= 5:
            feature = line.strip()[line.index('--- ') + 4:].split()[0]
            if feature not in main_splits:
                main_splits.append(feature)
print(f'Best Split Features: {main_splits}')

# Tolerance is the # of yards we can be off (+ & -)
tolerance = 21

toleranceResults = []
for tol in range(tolerance):
    correct = 0
    for index, rowdata in test_data.iterrows():
        val = test_labels[index]
        prediction = classifier.predict([rowdata])[0]
        if abs(prediction - val) <= tol:
            correct += 1
    toleranceResults.append(correct/len(test_data.index)*100)
plt.plot(range(tolerance), toleranceResults)
plt.xlabel('Tolerance')
plt.ylabel('Percent Correct Within Tolerance Level')
plt.xticks(range(tolerance))
plt.show()

# print('Results\nTolerance:', tolerance, 'yards\nCorrect:', correct, 'plays\nTotal:', i, 'plays\nPercentage:', str(correct / i * 100) + '%')
