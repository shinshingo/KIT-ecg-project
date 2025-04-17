# -----------------------------------------------------------------------------
# MIT-BIH 부정맥 데이터베이스의 각 레코드별로 부정맥 심볼(라벨) 출현 빈도를 집계하여
# CSV 파일로 저장하는 스크립트입니다.
# - 데이터 디렉토리에서 레코드 목록을 자동 추출합니다.
# - 각 레코드의 annotation(주석) 파일을 읽어 심볼별 개수를 셉니다.
# - 심볼별 의미(라벨명)도 함께 매핑하여 결과를 저장합니다.
# -----------------------------------------------------------------------------

import wfdb
import os
import pandas as pd
import platform
from collections import Counter

# 부정맥 심볼 ↔ 의미 매핑 딕셔너리
annotation_labels = {
    'N': 'Normal beat',
    'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat',
    'B': 'Bundle branch block beat (unspecified)',
    'A': 'Atrial premature beat',
    'a': 'Aberrated atrial premature beat',
    'J': 'Nodal (junctional) premature beat',
    'S': 'Supraventricular premature or ectopic beat (atrial or nodal)',
    'V': 'Premature ventricular contraction',
    'r': 'R-on-T premature ventricular contraction',
    'F': 'Fusion of ventricular and normal beat',
    'e': 'Atrial escape beat',
    'j': 'Nodal (junctional) escape beat',
    'n': 'Supraventricular escape beat (atrial or nodal)',
    'E': 'Ventricular escape beat',
    '/': 'Paced beat',
    'f': 'Fusion of paced and normal beat',
    'Q': 'Unclassifiable beat',
    '?': 'Beat not classified during learning',

    # Non-beat annotations
    '[': 'Start of ventricular flutter/fibrillation',
    '!': 'Ventricular flutter wave',
    ']': 'End of ventricular flutter/fibrillation',
    'x': 'Non-conducted P-wave (blocked APC)',
    '(': 'Waveform onset',
    ')': 'Waveform end',
    'p': 'Peak of P-wave',
    't': 'Peak of T-wave',
    'u': 'Peak of U-wave',
    '`': 'PQ junction',
    "'": 'J-point',
    '^': '(Non-captured) pacemaker artifact',
    '|': 'Isolated QRS-like artifact',
    '~': 'Change in signal quality',
    '+': 'Rhythm change',
    's': 'ST segment change',
    'T': 'T-wave change',
    '*': 'Systole',
    'D': 'Diastole',
    '=': 'Measurement annotation',
    '"': 'Comment annotation',
    '@': 'Link to external data',
}

# 운영체제에 따라 데이터 디렉토리 경로 설정
if platform.system() == 'Windows': # Windows
    data_dir = './data/mit-bih-arrhythmia-database-1.0.0/'
elif platform.system() == 'Darwin': # macOS
    data_dir = 'data/mit-bih-arrhythmia-database-1.0.0/'
elif platform.system() == 'Linux': # Linux
    data_dir = './data/mit-bih-arrhythmia-database-1.0.0/'
else:
    raise Exception("Unsupported OS") # 지원하지 않는 운영체제 예외 처리

# 데이터 디렉토리에서 레코드 ID 목록 자동 추출 (.hed 파일 기준)
record_ids = sorted(set(
    os.path.splitext(f)[0]
    for f in os.listdir(data_dir)
    if f.endswith('.hed')
))

# 각 레코드별로 annotation 파일을 읽어 심볼별 개수 집계
record_symbol_counts = []
all_symbols = set()

for rec in record_ids:
    try:
        ann = wfdb.rdann(os.path.join(data_dir, rec), 'atr')  # annotation 파일 읽기
        symbol_count = Counter(ann.symbol)                    # 심볼별 개수 세기
        all_symbols.update(symbol_count.keys())               # 전체 심볼 집합 갱신
        record_symbol_counts.append({'record': rec, **symbol_count})  # 결과 저장
    except Exception as e:
        print(f"Failed to load {rec}: {e}")

# 집계 결과를 DataFrame으로 변환 및 결측값 처리
df = pd.DataFrame(record_symbol_counts)
df = df.fillna(0).astype({sym: int for sym in all_symbols})
df = df[['record'] + sorted(all_symbols)]

# 라벨(심볼) 정렬 및 의미 매핑
ordered_symbols = sorted(all_symbols)
label_names = [annotation_labels.get(sym, 'Unknown') for sym in ordered_symbols]

# 컬럼명에 라벨명 추가
df_with_labels = df.copy()
df_with_labels.columns = ['record'] + [f'{sym} ({name})' for sym, name in zip(ordered_symbols, label_names)]

# 결과를 CSV 파일로 저장
output_path = os.path.join(data_dir, 'arrhythmia_label_stats.csv')
print(f"Saving to {output_path}")

df_with_labels.to_csv(output_path, index=False, encoding='utf-8-sig')  # Excel 호환 인코딩

print(f"저장완료: {output_path}")

# (선택) DataFrame을 사용자에게 표시하는 코드 (주석 처리)
# import ace_tools_open as tools; tools.display_dataframe_to_user(name="레코드별 부정맥 라벨 통계", dataframe=df)

