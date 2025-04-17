"""
MIT-BIH 부정맥 데이터베이스의 ECG 신호에서 R-peak를 중심으로 일정 길이의 세그먼트를 추출하고,
정상/이상 레이블을 부여하여 AutoEncoder 학습용 데이터셋(npz 파일)으로 저장하는 스크립트입니다.
또한, 데이터 분포 및 샘플 파형을 시각화하여 데이터셋의 특성을 확인할 수 있습니다.
"""

import wfdb  # WFDB 라이브러리: ECG 데이터 읽기 및 주석 처리
import biosppy  # BioSPPy 라이브러리: ECG 신호 분석
import os  # 파일 및 디렉토리 경로 관리
import platform  # 플랫폼 정보 확인을 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 Matplotlib
import numpy as np  # 수치 계산을 위한 NumPy
import pandas as pd  # 데이터 분석을 위한 Pandas
import seaborn as sns # 시각화를 위한 Seaborn

# 데이터 디렉토리 경로 및 사용할 레코드 ID 목록 설정
if platform.system() == 'Windows': # Windows 환경에서 데이터 경로 설정
    data_dir = './data/mit-bih-arrhythmia-database-1.0.0/'
elif platform.system() == 'Darwin': # macOS 환경에서 데이터 경로 설정
    data_dir = 'data/mit-bih-arrhythmia-database-1.0.0/'
else:
    raise Exception("Unsupported OS") # 지원하지 않는 운영체제 예외 처리

# 사용할 beat 어노테이션 심볼 목록 정의
valid_beat_symbols = {
    'N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E'
}

def extract_segments_from_record(record_id, data_dir, channel_index=0, window_size=180, plot=False):
    """
    주어진 레코드에서 R-peak를 중심으로 window_size 길이의 ECG 세그먼트 추출.
    정상/이상 레이블 및 심볼 정보도 함께 반환.
    plot=True로 설정 시, 샘플별 파형을 키보드로 넘겨가며 시각화 가능.
    """
    record_path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(record_path)  # ECG 신호 및 메타데이터 읽기
    annotation = wfdb.rdann(record_path, 'atr')  # R-peak 및 beat 어노테이션 읽기
    ecg = record.p_signal[:, channel_index]  # 선택 채널의 ECG 신호 추출
    fs = record.fs  # 샘플링 주파수
    margin = window_size // 2  # R-peak 기준 앞뒤 margin

    # ECG 신호 필터링 (노이즈 제거)
    filtered = biosppy.signals.ecg.ecg(signal=ecg, sampling_rate=fs, show=False)['filtered']

    segments, labels, symbols = [], [], []
    positions = []

    # 각 beat 어노테이션에 대해 세그먼트 추출
    for i, sym in enumerate(annotation.symbol):
        if sym not in valid_beat_symbols:
            continue

        r = annotation.sample[i]
        if r < margin or r + margin > len(filtered):
            continue

        seg = filtered[r - margin : r + margin]
        # 정상(1): 'N', 'L', 'R', 'e', 'j' / 이상(0): 그 외
        label = 1 if (sym == 'N') or (sym == 'L') or (sym == 'R') or (sym == 'e') or (sym == 'j') else 0

        segments.append(seg)
        labels.append(label)
        symbols.append(sym)
        positions.append(i)

    segments = np.array(segments)
    labels = np.array(labels)

    # 시각화 모드: 키보드로 샘플 넘기며 파형 확인
    if plot and len(segments) > 0:
        idx = [0]
        fig, ax = plt.subplots(figsize=(12, 4))

        def plot_sample():
            ax.clear()
            t = np.arange(-margin, margin)
            seg = segments[idx[0]]
            sym = symbols[idx[0]]
            label_text = 'Normal' if labels[idx[0]] == 1 else 'Anomaly'

            ax.plot(t, seg, color='black')
            ax.axvline(x=0, color='green', linestyle='--', label='R-peak')
            ax.set_title(f"ID {record_id} | Sample {idx[0]+1}/{len(segments)} | Label: '{sym}' ({label_text})")
            ax.grid(True)
            ax.legend()
            fig.canvas.draw()

        def on_key(event):
            if event.key == 'right':
                idx[0] += 1
                if idx[0] >= len(segments):
                    print("마지막 샘플입니다.")
                    idx[0] = len(segments) - 1
                plot_sample()
            elif event.key == 'left':
                idx[0] -= 1
                if idx[0] < 0:
                    print("첫 번째 샘플입니다.")
                    idx[0] = 0
                plot_sample()
            elif event.key.lower() in ['q', 'escape']:
                print("시각화 종료")
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plot_sample()
        plt.show()

    return segments, labels, symbols

def visualize_dataset_summary(labels, symbols, segments, sample_length=180, num_samples=5):
    """
    전체 데이터셋의 레이블 분포, 심볼 분포, 샘플 파형을 시각화하여 데이터셋 특성 확인.
    """
    labels = np.array(labels)
    symbols = np.array(symbols)

    # 1. 정상/이상 레이블 분포 히스토그램
    plt.figure(figsize=(5,4))
    sns.countplot(x=labels)
    plt.title("Label Distribution")
    plt.xticks([0,1], ['Anomaly', 'Normal'])
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. 심볼별 분포 바플롯
    symbol_counts = pd.Series(symbols).value_counts()
    plt.figure(figsize=(10,4))
    sns.barplot(x=symbol_counts.index, y=symbol_counts.values)
    plt.title("ECG Symbol Distribution")
    plt.xlabel("ECG Symbol")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. 정상/이상 샘플 파형 시각화 (랜덤 샘플)
    normal_idxs = np.where(labels == 1)[0]
    anomaly_idxs = np.where(labels == 0)[0]

    fig, axs = plt.subplots(2, num_samples, figsize=(3*num_samples, 5))
    for i in range(num_samples):
        if i < len(normal_idxs):
            axs[0, i].plot(np.arange(sample_length), segments[normal_idxs[i]])
            axs[0, i].set_title(f"Normal - {symbols[normal_idxs[i]]}")
        axs[0, i].grid(True)
        if i < len(anomaly_idxs):
            axs[1, i].plot(np.arange(sample_length), segments[anomaly_idxs[i]])
            axs[1, i].set_title(f"Anomaly - {symbols[anomaly_idxs[i]]}")
        axs[1, i].grid(True)

    plt.suptitle("Random ECG Segments (Normal / Anomaly)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def save_segments_to_npz(segments, labels, symbols, save_path):
    """
    추출된 세그먼트, 레이블, 심볼 정보를 npz 파일로 저장.
    """
    np.savez(save_path, x=segments, y=labels, symbols=np.array(symbols))
    print(f"NPZ 저장 완료: {save_path}.npz")

# 자동으로 레코드 ID 추출
record_ids = sorted(set(
    os.path.splitext(f)[0]
    for f in os.listdir(data_dir)
    if f.endswith('.hea')
))

print(f"레코드 ID 목록: {record_ids}")

x_total = []
y_total = []
sym_total = []

# 각 레코드별로 세그먼트 추출 및 누적
for record_id in record_ids:
    x, y, sym = extract_segments_from_record(record_id, data_dir, plot=False)
    x_total.append(x)
    y_total.append(y)
    sym_total.append(sym)
    print(f"레코드 {record_id}에서 {len(x)}개의 세그먼트 추출 완료")

# 누락 보완: 전체 데이터 합치기
x_total = np.concatenate(x_total, axis=0)
y_total = np.concatenate(y_total, axis=0)
sym_total = np.concatenate(sym_total, axis=0)

# 저장 경로 설정
npz_save_path = "MIT_BIH_rpeak_segments"

# npz 저장 실행
save_segments_to_npz(x_total, y_total, sym_total, npz_save_path)

# 시각화
visualize_dataset_summary(y_total, sym_total, x_total, sample_length=180, num_samples=6)