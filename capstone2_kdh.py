# -*- coding: utf-8 -*-
"""CS2v0.2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1inJDrQ5-KiWrewujsbnPRK7NDNcANYUE
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Dense
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from torch import nn, optim

df = pd.read_csv("https://raw.githubusercontent.com/donghaeK/capStone2/main/capStone2(1).csv")
original_y = df["y"].values
dates = pd.to_datetime(df["Date"])
X = list(df)[1:9]
df = df[X].astype(float)


scaler = StandardScaler()
scaler = scaler.fit(df)
stock_data_scaled = scaler.transform(df)

n_train = int(0.8*stock_data_scaled.shape[0])
train_data_scaled = stock_data_scaled[0: n_train]
train_dates = dates[0: n_train]

test_data_scaled = stock_data_scaled[n_train:]
test_dates = dates[n_train:]

pred_days = 30  # prediction period
seq_len = 3   # sequence length = past days for future prediction.
input_dim = 8  # input_dimension = ['y','Exchange Rate']

trainX = []
trainY = []
testX = []
testY = []

for i in range(seq_len, n_train-pred_days +1):
    trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
    trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

# LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(trainY.shape[1]))
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
early_stopping=EarlyStopping(monitor='loss',patience=30, verbose=1)
history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_split=0.1, verbose=1, callbacks=[early_stopping])
#history = model.fit(trainX, trainY, epochs=500, batch_size=100, validation_split=0.1, verbose=1)

prediction = model.predict(testX)

# generate array filled with means for prediction
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

# substitute predictions into the first column
mean_values_pred[:, 0] = np.squeeze(prediction)

# inverse transform
y_pred = scaler.inverse_transform(mean_values_pred)[:,0]


# generate array filled with means for testY
mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

# substitute testY into the first column
mean_values_testY[:, 0] = np.squeeze(testY)

# inverse transform
testY_original = scaler.inverse_transform(mean_values_testY)[:,0]

# Trim the dates to match the prediction and test data lengths
trimmed_dates = test_dates[seq_len:seq_len+len(testY_original)]


# plot actual vs predicted
plt.figure(figsize=(14, 5))
plt.plot(dates, original_y, color='green', label='Original tourist')
plt.plot(dates[-len(testY_original):], testY_original, color='blue', label='Actual tourist')
plt.plot(dates[-len(y_pred):], y_pred, color='red', label='Predicted tourist')
plt.xlabel('Date')
plt.ylabel('tourist')
plt.ylim(400000,2000000)
plt.title('Original, Actual and Predicted Number of tourists (unit: 1 million)')
plt.legend()
plt.show()

