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

# 데이터 디렉토리 경로 및 사용할 레코드 ID 목록 설정
if platform.system() == 'Windows': # Windows
    data_dir = './data/mit-bih-arrhythmia-database-1.0.0/'
elif platform.system() == 'Darwin': # macOS
    data_dir = 'data/mit-bih-arrhythmia-database-1.0.0/'
else:
    raise Exception("Unsupported OS") # 지원하지 않는 운영체제 예외 처리

# 자동으로 레코드 ID 추출
record_ids = sorted(set(
    os.path.splitext(f)[0]
    for f in os.listdir(data_dir)
    if f.endswith('.atr') or f.endswith('.dat')
))

# 부정맥 심볼 카운트
record_symbol_counts = []
all_symbols = set()

for rec in record_ids:
    try:
        ann = wfdb.rdann(os.path.join(data_dir, rec), 'atr')
        symbol_count = Counter(ann.symbol)
        all_symbols.update(symbol_count.keys())
        record_symbol_counts.append({'record': rec, **symbol_count})
    except Exception as e:
        print(f"Failed to load {rec}: {e}")

# DataFrame 정리
df = pd.DataFrame(record_symbol_counts)
df = df.fillna(0).astype({sym: int for sym in all_symbols})
df = df[['record'] + sorted(all_symbols)]


# 라벨 정렬 순서 유지
ordered_symbols = sorted(all_symbols)

# 라벨 이름 딕셔너리 → 리스트로 매핑
label_names = [annotation_labels.get(sym, 'Unknown') for sym in ordered_symbols]

# 라벨 이름을 첫 줄에 추가해 테이블 위에 붙일 수도 있고, 새 파일로 만들 수도 있음
df_with_labels = df.copy()
df_with_labels.columns = ['record'] + [f'{sym} ({name})' for sym, name in zip(ordered_symbols, label_names)]

# 파일 저장 경로
output_path = os.path.join(data_dir, 'arrhythmia_label_stats.csv')
print(f"Saving to {output_path}")

# CSV로 저장
df_with_labels.to_csv(output_path, index=False, encoding='utf-8-sig')  # Excel에서 열기 편하도록 utf-8-sig 사용

print(f"저장완료: {output_path}")

# 출력
# import ace_tools_open as tools; tools.display_dataframe_to_user(name="레코드별 부정맥 라벨 통계", dataframe=df)

