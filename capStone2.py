
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import font_manager ,rc
plt.rc('font', family='Malgun Gothic')
print(plt.rcParams['font.family'])

# CSV 파일에서 데이터 읽기


data = pd.read_csv('https://raw.githubusercontent.com/donghaeK/capStone2/main/capStone2.csv' ,encoding='cp949')
# data = pd.read_csv('C:/csv/관광시설/capStone2.csv', encoding='cp949')
data = data.dropna()

# 입력 및 출력 데이터 설정
X = data[['년도']]
y = data[['서울 방한객 수', '호텔이용관광객', '쇼핑', '관광장소 및 시설(서울)']]

# 데이터 정규화
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 다중 출력 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(y.shape[1])  # 출력 데이터의 차원을 설정
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200)

# 예측
y_pred = model.predict(X_test)

# 정규화된 예측값을 원래 스케일로 복원
y_pred = scaler_y.inverse_transform(y_pred)

# 연도 데이터를 예측
input_Year = np.array([[2040]])  # 연도에 해당하는 입력 데이터를 생성
input_Year = scaler_X.transform(input_Year)  # 입력 데이터 정규화
input_Year1 = np.array([[2060]])  # 연도에 해당하는 입력 데이터를 생성
input_Year1 = scaler_X.transform(input_Year1)  # 입력 데이터 정규화
input_Year2 = np.array([[2080]])  # 연도에 해당하는 입력 데이터를 생성
input_Year2 = scaler_X.transform(input_Year2)  # 입력 데이터 정규화
input_Year3 = np.array([[2100]])  # 연도에 해당하는 입력 데이터를 생성
input_Year3 = scaler_X.transform(input_Year3)  # 입력 데이터 정규화

# 모델을 사용하여 연도 데이터 예측
output_normalized = model.predict(input_Year)
output_normalized1 = model.predict(input_Year1)
output_normalized2 = model.predict(input_Year2)
output_normalized3 = model.predict(input_Year3)

# 예측값을 원래 스케일로 복원
output_Year = scaler_y.inverse_transform(output_normalized)
output_Year1 = scaler_y.inverse_transform(output_normalized1)
output_Year2 = scaler_y.inverse_transform(output_normalized2)
output_Year3 = scaler_y.inverse_transform(output_normalized3)

print("2040년 예측 결과:")
print("서울 방한객 수:", output_Year[0, 0])
print("호텔이용관광객:", output_Year[0, 1])
print("쇼핑:", output_Year[0, 2])
print("관광장소 및 시설(서울):", output_Year[0, 3])

categories = ['서울 방한객 수', '호텔이용관광객', '쇼핑', '관광장소 및 시설(서울)']

# 막대 차트 생성
plt.figure(figsize=(15, 15))
plt.bar(categories, output_Year[0], color="black")
plt.xlabel('카테고리')
plt.ylabel('예측 값')
plt.title('2040년 예측 값')
plt.xticks(rotation=45)
plt.show()

print("2060년 예측 결과:")
print("서울 방한객 수:", output_Year1[0, 0])
print("호텔이용관광객:", output_Year1[0, 1])
print("쇼핑:", output_Year1[0, 2])
print("관광장소 및 시설(서울):", output_Year1[0, 3])

plt.figure(figsize=(15, 15))
plt.bar(categories, output_Year1[0], color="black")
plt.xlabel('카테고리')
plt.ylabel('예측 값')
plt.title('2060년 예측 값')
plt.xticks(rotation=45)
plt.show()

print("2080년 예측 결과:")
print("서울 방한객 수:", output_Year2[0, 0])
print("호텔이용관광객:", output_Year2[0, 1])
print("쇼핑:", output_Year2[0, 2])
print("관광장소 및 시설(서울):", output_Year2[0, 3])

plt.figure(figsize=(15, 15))
plt.bar(categories, output_Year2[0], color="black")
plt.xlabel('카테고리')
plt.ylabel('예측 값')
plt.title('2080년 예측 값')
plt.xticks(rotation=45)
plt.show()

print("2100년 예측 결과:")
print("서울 방한객 수:", output_Year3[0, 0])
print("호텔이용관광객:", output_Year3[0, 1])
print("쇼핑:", output_Year3[0, 2])
print("관광장소 및 시설(서울):", output_Year3[0, 3])

plt.figure(figsize=(15, 15))
plt.bar(categories, output_Year3[0], color="black")
plt.xlabel('카테고리')
plt.ylabel('예측 값')
plt.title('2100년 예측 값')
# plt.xticks(rotation=45)
plt.show()
# plt.plot(x=[df.bmi[df.diabetes==0],df.bmi[df.diabetes==1]], bins=30, histtype='barstacked',label=['normal','diabetes'])