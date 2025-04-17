# =============================================================================
# MIT-BIH ECG 데이터셋의 R-peak 세그먼트에 대해 오토인코더(AutoEncoder)를 학습하여
# 정상/이상 심전도 신호를 판별하는 코드입니다.
# 주요 동작:
# 1. MIT-BIH R-peak 세그먼트 데이터(.npz) 로드 및 전처리(샘플별 정규화)
# 2. 정상/이상 데이터 분리 및 오토인코더 모델 정의
# 3. 정상 데이터로 오토인코더 학습
# 4. 재구성 오차 기반 임계값(threshold) 산출
# 5. 테스트 데이터에 대해 이상 탐지 및 성능 평가(정확도, 정밀도, 재현율)
# 6. 학습된 모델 저장
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# NPZ 데이터 로드
npz = np.load('MIT_BIH_rpeak_segments.npz')
data = npz['x']            # (samples, 360)
labels = npz['y']          # (samples,)
symbols = npz['symbols']   # (samples,) — 필요 시 사용

sample_length = data.shape[1]   

print(f'Total samples: {data.shape[0]}')

# 학습/테스트 데이터 분할
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

print(f'Train samples: {train_data.shape[0]}')
print(f'Test samples: {test_data.shape[0]}')

# 샘플별 정규화 함수 정의
def normalize_per_sample(X):
    """
    각 샘플별로 min-max 정규화를 수행합니다.
    """
    X_norm = np.empty_like(X)
    for i in range(X.shape[0]):
        x = X[i]
        x_min, x_max = np.min(x), np.max(x)
        if x_max > x_min:
            X_norm[i] = (x - x_min) / (x_max - x_min)
        else:
            X_norm[i] = x  # 모든 값이 같은 경우 예외 처리
    return X_norm

# 각 샘플별 정규화 적용
train_data = normalize_per_sample(train_data).astype(np.float32)
test_data = normalize_per_sample(test_data).astype(np.float32)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

# 정상/이상 데이터 분리
normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

# # 정상 심전도 샘플 시각화 예시
# plt.grid()
# plt.plot(np.arange(360), normal_train_data[0])
# plt.title("A Normal ECG")
# plt.show()

# # 이상 심전도 샘플 시각화 예시
# plt.grid()
# plt.plot(np.arange(360), anomalous_train_data[0])
# plt.title("An Anomalous ECG")
# plt.show()

class AnomalyDetector(Model):
    """
    오토인코더 기반 이상 탐지 모델 정의
    """
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        # 인코더 정의
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])
        # 디코더 정의
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(sample_length, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 오토인코더 모델 생성 및 컴파일
autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')

# 정상 데이터로 오토인코더 학습
history = autoencoder.fit(
    normal_train_data, normal_train_data, 
    epochs=200, 
    batch_size=512,
    validation_data=(test_data, test_data),
    shuffle=True
)

# 학습 및 검증 손실 시각화
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# 정상 테스트 데이터 재구성 결과 시각화
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(sample_length), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# 이상 테스트 데이터 재구성 결과 시각화
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(sample_length), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# 정상 학습 데이터의 재구성 손실 분포 확인
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

# 임계값(threshold) 계산
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

# 이상 테스트 데이터의 재구성 손실 분포 확인
reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

def predict(model, data, threshold):
    """
    입력 데이터에 대해 재구성 손실이 임계값 미만이면 정상(True), 이상(False)으로 판별
    """
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    """
    예측 결과에 대한 정확도, 정밀도, 재현율 출력
    """
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

# 테스트 데이터에 대해 이상 탐지 및 성능 평가
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)

# 모델 저장
autoencoder.export('ecg_autoencoder_savedmodel')

