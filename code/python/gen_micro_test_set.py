# -----------------------------------------------------------------------------
# 이 파일은 MIT-BIH ECG 데이터셋에서 정상 및 비정상 샘플을 선택하여,
# TFLite로 변환된 오토인코더 모델을 통해 재구성 결과와 에러를 계산하고,
# 결과를 시각화 및 C 배열 형태로 출력하는 테스트 스크립트입니다.
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# NPZ 데이터셋 로드
# MIT_BIH_rpeak_segments.npz 파일에서 ECG 신호와 라벨, 심볼 정보를 불러옵니다.
data = np.load("MIT_BIH_rpeak_segments.npz")
X = data["x"]
y = data["y"].astype(bool)  # True = 정상, False = 비정상
sym = data["symbols"]  # 필요 시 사용

# 샘플 선택
# 정상 샘플 1개, 비정상 샘플 1개(60번째)를 선택합니다.
normal_sample = X[y][0]        # 첫 번째 정상 샘플
anomaly_sample = X[~y][60]      # 첫 번째 비정상 샘플
symbol = sym[~y][60]  # 비정상 샘플의 심볼

# 정규화 함수 정의
# 입력 신호를 0~1 범위로 정규화합니다.
def normalize(sample):
    return (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

# 정규화 수행
normal_sample = normalize(normal_sample)
anomaly_sample = normalize(anomaly_sample)

# TFLite 모델 로드
# 오토인코더 TFLite 모델을 메모리에 로드합니다.
interpreter = tf.lite.Interpreter(model_path="ecg_autoencoder_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 추론 함수 정의
# 입력 샘플을 오토인코더에 통과시켜 재구성 결과를 반환합니다.
def run_inference(sample):
    input_data = np.expand_dims(sample.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

# 추론 실행
# 정상/비정상 샘플 각각에 대해 오토인코더 추론을 수행합니다.
normal_output = run_inference(normal_sample)
anomaly_output = run_inference(anomaly_sample)

# reconstruction error 계산
# 입력과 출력의 차이(재구성 오차) 및 평균 오차를 계산합니다.
def compute_error(input_, output_):
    diff = np.abs(input_ - output_)
    avg = np.mean(diff)
    return diff, avg

diff_n, avg_n = compute_error(normal_sample, normal_output)
diff_a, avg_a = compute_error(anomaly_sample, anomaly_output)

# 결과 출력
# 정상/비정상 샘플의 재구성 오차 및 일부 값을 출력합니다.
print("=======================================")
print("정상 샘플")
print(f"평균 Reconstruction Error: {avg_n:.5f}")
print("입력값 (앞 10개):", np.round(normal_sample[:10], 4))
print("출력값 (앞 10개):", np.round(normal_output[:10], 4))
print("오차값 (앞 10개):", np.round(diff_n[:10], 5))
print("---------------------------------------")
print("비정상 샘플")
print(f"평균 Reconstruction Error: {avg_a:.5f}")
print("입력값 (앞 10개):", np.round(anomaly_sample[:10], 4))
print("출력값 (앞 10개):", np.round(anomaly_output[:10], 4))
print("오차값 (앞 10개):", np.round(diff_a[:10], 5))
print("=======================================")

# 시각화 함수 정의
# 입력 신호와 재구성 신호를 비교하여 그래프로 시각화합니다.
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

# 시각화 수행
plot_ecg_comparison(normal_sample, normal_output, title="Normal ECG Reconstruction", color="green")
plot_ecg_comparison(anomaly_sample, anomaly_output, title=f"Anomalous ECG Reconstruction {symbol}", color="blue")

# C 배열 형식으로 출력하는 함수
# 입력 신호를 C 배열 형식으로 출력합니다.
def print_c_array(arr, name="sample"):
    print(f"float {name}[{len(arr)}] = {{")
    for i, val in enumerate(arr):
        end = ",\n" if (i + 1) % 10 == 0 else ", "
        print(f"  {val:.4f}", end=end)
    print("\n};")

# C 배열 출력
print("정상 샘플:")
print_c_array(normal_sample, name="normal_ecg")

print("\n비정상 샘플:")
print_c_array(anomaly_sample, name="anomaly_ecg")