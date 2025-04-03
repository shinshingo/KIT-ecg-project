import wfdb  # WFDB 라이브러리: ECG 데이터 읽기 및 주석 처리
import biosppy  # BioSPPy 라이브러리: ECG 신호 분석
import os  # 파일 및 디렉토리 경로 관리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 Matplotlib

# MIT-BIH 데이터셋에서 ECG 신호를 읽고 시각화하는 코드

# 데이터 디렉토리 경로 및 사용할 레코드 ID 목록 설정
data_dir = 'data/mit-bih-arrhythmia-database-1.0.0/'
record_ids = ['101']  # 사용할 레코드 ID 목록

# 세그먼트 길이 및 ECG 채널 설정
segment_length = 3000  # 한 번에 표시할 샘플 수
channel_index = 0  # ECG 채널 선택 (MLII 채널)

# 각 레코드 ID에 대해 처리
for record_id in record_ids:
    # 1. 레코드 읽기
    record = wfdb.rdrecord(os.path.join(data_dir, record_id))  # ECG 신호 데이터 읽기
    annotation = wfdb.rdann(os.path.join(data_dir, record_id), 'atr')  # 주석 데이터 읽기
    
    # 2. 신호 데이터 추출
    ecg_signal = record.p_signal[:, channel_index]  # 선택한 채널의 신호 데이터 추출
    sampling_rate = record.fs  # 샘플링 주파수 
    total_len = len(ecg_signal)  # 신호 데이터의 전체 길이

    # 3. BioSPPy로 ECG 분석
    ecg_result = biosppy.signals.ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    rpeaks = ecg_result['rpeaks']  # R-peak 위치 추출

    # 초기 세그먼트 시작 위치
    start = 0
    fig, ax = plt.subplots(figsize=(15, 5))  # 플롯 크기 설정

    # ---------- 플롯 + 키 이벤트 핸들러 ----------
    def plot_segment():
        """현재 세그먼트를 플롯에 시각화"""
        ax.clear()  # 이전 플롯 초기화
        end = min(start + segment_length, total_len)  # 현재 세그먼트의 끝 위치 계산
        ax.plot(ecg_signal[start:end], label='ECG Signal', color='black')  # ECG 신호 플롯

        # 주석 데이터 표시
        for i in range(len(annotation.sample)):
            pos = annotation.sample[i]  # 주석 위치
            if start <= pos < end:  # 현재 세그먼트 내에 있는 주석만 표시
                rel_pos = pos - start  # 세그먼트 내 상대 위치 계산
                ax.axvline(x=rel_pos, color='red', linestyle='--', alpha=0.4)  # 주석 위치에 수직선 표시
                ax.text(rel_pos, ecg_signal[pos] + 0.2, annotation.symbol[i],  # 주석 심볼 표시
                        rotation=90, fontsize=8, color='red')

        # R-peak 데이터 표시
        for peak in rpeaks:
            if start <= peak < end:  # 현재 세그먼트 내에 있는 R-peak만 표시
                rel_peak = peak - start  # 세그먼트 내 상대 위치 계산
                ax.plot(rel_peak, ecg_signal[peak], 'go')  # R-peak 위치에 녹색 점 표시

        # 플롯 제목 및 축 레이블 설정
        ax.set_title(f'ECG [{start} - {end}]')  # 현재 세그먼트 범위 표시
        ax.set_xlabel('Sample Index (segment)')  # x축 레이블
        ax.set_ylabel('Amplitude')  # y축 레이블
        ax.grid(True)  # 그리드 표시
        ax.legend()  # 범례 표시
        fig.canvas.draw()  # 플롯 업데이트

    def on_key(event):
        """키보드 이벤트 핸들러"""
        global start
        if event.key == 'right':  # 오른쪽 키를 누르면 다음 세그먼트로 이동
            start += segment_length  # 시작 위치를 다음 세그먼트로 이동
            if start >= total_len:  # 마지막 세그먼트에 도달하면 종료
                print("마지막 세그먼트입니다. 종료합니다.")
                plt.close()
            else:
                plot_segment()  # 다음 세그먼트 플롯
        elif event.key == 'left':  # 왼쪽 화살표 키를 누르면 이전 세그먼트로 이동
            start -= segment_length
            if start < 0:
                start = 0
                print("첫 번째 세그먼트입니다.")
            plot_segment()
        elif event.key.lower() in ['q', 'escape']:  # 'q' 또는 'Escape' 키를 누르면 종료
            print("사용자 종료 요청. 종료합니다.")
            plt.close()

    # 키보드 이벤트 연결
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 초기 세그먼트 플롯
    plot_segment()
    plt.show()  # 플롯 표시

