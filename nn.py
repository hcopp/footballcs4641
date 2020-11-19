import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

labels = pd.read_csv('./data/labels.csv')
data = pd.read_csv('./data/scaled.csv')
# combo = pd.concat([data, labels], axis=1)

train_data = data[:24805]
train_labels = labels[:24805]['Yards']
print(train_labels)

test_data = data[24805:]
test_labels = labels[24805:]['Yards']

model = Sequential()
model.add(Dense(128, kernel_initializer='normal', input_dim=len(data.columns), activation='relu'))
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='relu'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error', 'accuracy'])
model.summary()

model.fit(train_data, train_labels, epochs=1024, batch_size=256, validation_split=0.2)

print('loss: ', model.evaluate(test_data, test_labels))

print(model.predict(test_data))