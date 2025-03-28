import wfdb
import matplotlib.pyplot as plt

# MIT-BIH의 100번 레코드 불러오기
record = wfdb.rdrecord('100', pn_dir='mitdb')  # PhysioNet에서 자동 다운로드
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')  # 'atr'은 annotation type

# ECG 신호 추출 (채널 0: MLII)
ecg_signal = record.p_signal[:, 0]
fs = record.fs  # 샘플링 주파수

# 어노테이션 정보 확인
print("R-peak indices:", annotation.sample[:10])  # R-peak 위치 (샘플 인덱스)
print("Annotation symbols:", annotation.symbol[:10])  # 부정맥 클래스

# 시각화: ECG + 어노테이션 표시
plt.figure(figsize=(15, 4))
plt.plot(ecg_signal[:3000], label='ECG Signal')  # 10초 분량 표시

# R-peak 위치 표시
for i in range(len(annotation.sample)):
    sample = annotation.sample[i]
    if sample < 3000:
        plt.axvline(x=sample, color='r', linestyle='--', alpha=0.5)
        plt.text(sample, 0.5, annotation.symbol[i], rotation=90, fontsize=8, color='red')

plt.title('ECG Signal with Annotations (Record 100)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()