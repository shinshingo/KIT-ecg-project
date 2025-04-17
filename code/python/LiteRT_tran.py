"""
이 파일은 ECG(심전도) 신호의 오토인코더 모델을 TensorFlow Lite 형식(.tflite)으로 변환하고,
변환된 모델을 이용해 ECG 신호의 이상 탐지(Anomaly Detection)를 수행하는 전체 파이프라인을 구현합니다.

주요 동작:
1. 저장된 TensorFlow SavedModel을 TFLite 모델로 변환 및 저장
2. 변환된 TFLite 모델을 로드하여 추론 환경 준비
3. ECG 데이터(.npz) 로드 및 정규화
4. TFLite 모델을 이용한 신호 재구성 및 reconstruction loss 계산
5. 정상/이상 판별 임계값(threshold) 산출
6. 이상 탐지 결과 평가(정확도, 정밀도, 재현율)
7. 결과 시각화(ECG 신호 비교, reconstruction loss 히스토그램)
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

# 1. TensorFlow SavedModel을 TFLite 모델로 변환
converter = tf.lite.TFLiteConverter.from_saved_model("ecg_autoencoder_savedmodel")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# 변환 실행
tflite_model = converter.convert()
# 변환된 모델을 파일로 저장
with open("ecg_autoencoder_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite 변환 완료: ecg_autoencoder_float32.tflite")
size_kb = os.path.getsize("ecg_autoencoder_float32.tflite") / 1024
print(f"모델 크기: {size_kb:.2f} KB")

# 2. 변환된 TFLite 모델 로드 및 추론 환경 준비
interpreter = tf.lite.Interpreter(model_path="ecg_autoencoder_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print(f"TFLite 모델 입력 shape: {input_shape}")

# 3. ECG 데이터(.npz) 로드 및 정규화
data = np.load("MIT_BIH_rpeak_segments.npz")
X = data["x"].astype(np.float32)
y = data["y"].astype(bool)

# 샘플별 min-max 정규화 함수
def normalize_per_sample(X):
    X_norm = np.empty_like(X)
    for i in range(X.shape[0]):
        x_min, x_max = np.min(X[i]), np.max(X[i])
        if x_max > x_min:
            X_norm[i] = (X[i] - x_min) / (x_max - x_min)
        else:
            X_norm[i] = X[i]  # 값이 모두 같은 경우 예외처리
    return X_norm

X = normalize_per_sample(X)

# 4. TFLite 모델을 이용한 추론 함수 정의
def tflite_predict(input_data):
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # (1, 180)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# 전체 데이터에 대해 재구성 및 reconstruction loss 계산
recons = np.array([tflite_predict(x) for x in X])
losses = np.mean(np.abs(recons - X), axis=1)

# ECG 입력과 재구성 결과를 비교 시각화하는 함수
def plot_ecg_comparison(input_, output_, title="ECG", color="blue"):
    plt.figure(figsize=(10, 3))
    plt.plot(input_, label='Input', color=color)
    plt.plot(output_, label='Reconstruction', color='red', linestyle='--')
    plt.fill_between(np.arange(len(input_)), input_, output_, color='lightcoral', alpha=0.5)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_ecg_comparison(X[0], recons[0], title="Normal ECG Reconstruction", color="green")

# 5. 정상 샘플의 평균+표준편차를 임계값(threshold)으로 설정
threshold = np.mean(losses[y]) + np.std(losses[y])
print(f"Threshold 설정: {threshold:.5f}")

# 6. reconstruction loss가 threshold 미만이면 정상으로 예측
preds = losses < threshold  # 정상으로 판단
labels = y  # 실제 값

# 7. 평가 지표 출력
print("TFLite 모델 평가 결과:")
print("Accuracy =", accuracy_score(labels, preds))
print("Precision =", precision_score(labels, preds))
print("Recall =", recall_score(labels, preds))

# 8. 정상/이상 샘플의 reconstruction loss 히스토그램 시각화
plt.hist(losses[labels], bins=50, alpha=0.6, label="Normal")
plt.hist(losses[~labels], bins=50, alpha=0.6, label="Anomaly")
plt.axvline(threshold, color='red', linestyle='--', label="Threshold")
plt.legend()
plt.title("Reconstruction Loss Histogram")
plt.xlabel("Loss")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# .tflite 모델을 C 배열로 변환하는 명령어 예시
# xxd -i ecg_autoencoder_float16.tflite > ecg_autoencoder_float16.cc
