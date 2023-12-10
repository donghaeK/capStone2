# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16nhtM3nMJSCkP3Mne7o1vSG_5cRiA3GQ
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/donghaeK/capStone2/main/capStone2.csv")

original_y = df['y'].values

dates = pd.to_datetime(df['Date'])

cols = list(df)[2:5]

df = df[cols].astype(float)

scaler = StandardScaler()
scaler = scaler.fit(df)
df_scaled = scaler.transform(df)

n_train = int(0.9*df_scaled.shape[0])
train_data_scaled = df_scaled[0: n_train]
train_dates = dates[0: n_train]

test_data_scaled = df_scaled[n_train:]
test_dates = dates[n_train:]

pred_days = 30
seq_len = 5
input_dim = 3

train_X = []
train_y = []
test_X = []
test_y = []

for i in range(seq_len, n_train-pred_days +1):
    train_X.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
    train_y.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    test_X.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    test_y.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

train_X, train_y = np.array(train_X), np.array(train_y)
test_X, test_y = np.array(test_X), np.array(test_y)


model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]),
               return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(train_y.shape[1]))

model.summary()

learning_rate = 0.01

optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mse')

prediction = model.predict(test_X)
print(prediction.shape, test_y.shape)

mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

mean_values_pred[:, 0] = np.squeeze(prediction)

y_pred = scaler.inverse_transform(mean_values_pred)[:,0]
print(y_pred.shape)

mean_values_test_y = np.repeat(scaler.mean_[np.newaxis, :], test_y.shape[0], axis=0)

mean_values_test_y[:, 0] = np.squeeze(test_y)

test_y_original = scaler.inverse_transform(mean_values_test_y)[:,0]
print(test_y_original.shape)

plt.figure(figsize=(14, 5))

plt.plot(dates, original_y, color='green', label='Original Tourist')

plt.plot(test_dates[seq_len:], test_y_original, color='blue', label='Actual Tourist')
plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label='Predicted Tourist')
plt.xlabel('Date')
plt.ylabel('tourist')
plt.title('Future tourist')
plt.legend()
plt.show()






zoom_start = len(test_dates) - 50
zoom_end = len(test_dates)

plt.figure(figsize=(14, 5))

adjusted_start = zoom_start - seq_len

plt.plot(test_dates[zoom_start:zoom_end],
         test_y[adjusted_start:zoom_end - zoom_start + adjusted_start],
         color='blue',
         label='Actual Open Price')

plt.plot(test_dates[zoom_start:zoom_end],
         y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start ],
         color='red',
         linestyle='--',
         label='Predicted Open Price')

plt.xlabel('Date')
plt.ylabel('tourist')
plt.title('Zoomed In Predicted Future tourist')
plt.legend()
plt.show()