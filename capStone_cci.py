import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/donghaeK/capStone2/main/capStone2(1).csv')
df['Date'] = pd.to_datetime(df['Date'])
select_columns = ['y', 'Exchange Rate', 'Oil', 'dust', 'Temperature', 'Precipitation', 'Humidity', 'Windy']
df_select = df[select_columns]
df_select.index = df['Date']
train_split = 200

dataset = df_select.values
dataset = (dataset-(dataset[:train_split].mean(axis=0)))/(dataset[:train_split].std(axis=0))


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)


past_history = 200
future_target = 20
STEP = 1

x_train, y_train = multivariate_data(dataset, dataset[:, 0], 0, train_split, past_history, future_target, STEP)
x_val, y_val = multivariate_data(dataset, dataset[:, 0], train_split, None, past_history, future_target, STEP)

BATCH_SIZE = 30
BUFFER_SIZE = 10
EPOCHS = 20
EVALUATION_INTERVAL = 50
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:]))
multi_step_model.add(tf.keras.layers.Dense(1))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics='mse')

for x, y in val_data.take(1):
    print(multi_step_model.predict(x).shape)

history = multi_step_model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data, validation_steps=50)

plt.plot(np.arange(len(history.history['loss'])),history.history['loss'], label='train loss')
plt.plot(np.arange(len(history.history['mse'])),history.history['mse'], label='train mse')
plt.legend()
plt.ylabel('Error')
plt.title('loss,mse graph')
plt.show()

y_pred = multi_step_model.predict(x_val)
print(y_pred)
# y_pred = scaler.inverse_transform(y_pred)