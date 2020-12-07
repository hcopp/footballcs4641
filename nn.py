import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

##################
# GPU ACCELERATION
##################

# uncomment to make sure the GPU is being used
# tf.debugging.set_log_device_placement(True)

# required on RTX 30-series cards
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# NN structure
IN_NEURONS = 64
HIDDEN_NEURONS = 128
NUM_HIDDEN_LAYERS = 3

# performance
EPOCHS = 1024
BATCH = 512

labels = pd.read_csv('./data/labels.csv')
data = pd.read_csv('./data/scaled.csv')

# keras will handle the train/test split
train_data = data
train_labels = to_categorical(labels['Yards'])

test_data = data[24805:]
test_labels = to_categorical(labels[24805:]['Yards'])

model = Sequential()
model.add(Dense(IN_NEURONS, input_dim=len(data.columns), activation='relu'))
for _ in range(NUM_HIDDEN_LAYERS):
    model.add(Dense(
        HIDDEN_NEURONS,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0001)
        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        # bias_regularizer=regularizers.l2(1e-4),
        # activity_regularizer=regularizers.l2(1e-5)
    ))
    # model.add(Dropout(0.5))
model.add(Dense(test_labels.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_data, train_labels,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.2,
    shuffle=True,
    use_multiprocessing=True,
    #callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
)

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print('loss: ', model.evaluate(test_data, test_labels))

print(model.predict(test_data))