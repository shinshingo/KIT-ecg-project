# ============================================================
# 이 파일은 ECG(심전도) 데이터의 이상 탐지를 위한 오토인코더(AutoEncoder) 기반 딥러닝 모델을 구현한 코드입니다.
# 1. 공개 ECG 데이터를 다운로드하여 전처리합니다.
# 2. 정상/비정상 데이터를 분리하여 학습 및 테스트셋을 구성합니다.
# 3. 오토인코더 모델을 정의하고 정상 데이터로 학습합니다.
# 4. 학습된 모델의 재구성 오차를 기반으로 이상 탐지 임계값을 설정합니다.
# 5. 테스트 데이터에 대해 이상 탐지 성능을 평가합니다.
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# 1. ECG 데이터셋 다운로드 및 로드
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# 2. 라벨(정상/비정상)과 신호 데이터 분리
labels = raw_data[:, -1]  # 마지막 열이 라벨
data = raw_data[:, 0:-1]  # 나머지 열이 ECG 신호

# 3. 학습/테스트 데이터 분할
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# 4. 데이터 정규화 (0~1 범위로 스케일링)
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# 5. 라벨을 불리언 타입으로 변환
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

# 6. 정상/비정상 데이터 분리
normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]
anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

# 7. 정상 ECG 데이터 시각화
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()

# 8. 비정상 ECG 데이터 시각화
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()

# 9. 오토인코더 모델 정의
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

# 10. 모델 컴파일 및 학습
autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)

# 11. 학습 과정 시각화
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

# 12. 정상 데이터 재구성 결과 시각화
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# 13. 비정상 데이터 재구성 결과 시각화
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# 14. 재구성 오차 기반 임계값 설정
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

# 15. 테스트 데이터 재구성 오차 시각화
reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

# 16. 이상 탐지 함수 정의
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

# 17. 성능 평가 함수 정의
def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# 18. 테스트 데이터에 대한 성능 평가
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)