import wfdb
import biosppy

# 1. MIT-BIH 100번 환자 신호 읽기 (2채널 ECG 중 1채널 사용)
record = wfdb.rdrecord('100', pn_dir='mitdb')  # PhysioNet에서 자동 다운로드
ecg_signal = record.p_signal[:, 1]  # MLII 채널 선택
sampling_rate = record.fs           # 보통 360Hz

# 2. BioSPPy로 ECG 분석
output = biosppy.signals.ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=True)

# 3. 결과 확인
rpeaks = output['rpeaks']
heart_rate = output['heart_rate']