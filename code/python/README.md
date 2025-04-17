# code/python/ 폴더 파일 설명

이 폴더는 MIT-BIH 부정맥 데이터베이스를 활용한 ECG(심전도) 신호 처리, 시각화, 데이터셋 생성, 통계 분석, 오토인코더 테스트 등 다양한 파이썬 스크립트를 포함합니다.

## 파일별 설명

- **ann_list.py**  
  MIT-BIH 데이터셋의 각 레코드별 부정맥 심볼(라벨) 출현 빈도를 집계하여 CSV로 저장합니다.  
  심볼별 의미(라벨명)도 함께 매핑합니다.

- **biosppy_feature_RP.py**  
  BioSPPy를 이용해 ECG 신호의 R-peak 검출 및 Recurrence Plot(재귀 플롯) 특성을 시각화합니다.  
  키보드 좌/우 방향키로 신호 세그먼트를 이동하며 확인할 수 있습니다.

- **gen_dataset_for_AE.py**  
  MIT-BIH ECG 신호에서 R-peak 중심 세그먼트를 추출하여 정상/이상 레이블을 부여하고, AutoEncoder 학습용 데이터셋(npz 파일)으로 저장합니다.  
  데이터 분포 및 샘플 파형 시각화 기능도 포함되어 있습니다.

- **gen_micro_test_set.py**  
  저장된 npz 데이터셋에서 정상/비정상 샘플을 선택하여 TFLite 오토인코더로 재구성 결과 및 에러를 계산하고, 결과를 시각화 및 C 배열로 출력합니다.

- **LiteRT_tran.py**  
  학습된 Keras 오토인코더 모델(h5 파일 등)을 TensorFlow Lite(.tflite) 포맷으로 변환하는 스크립트입니다.  
  변환 과정에서 양자화(quantization) 등 옵션을 적용할 수 있으며, ESP32 등 임베디드 환경에서 사용할 수 있도록 모델을 최적화합니다.

- **mit_bih_data_viewer.py**  
  WFDB와 BioSPPy를 이용해 ECG 신호, 주석(annotation), R-peak 정보를 구간별로 시각화합니다.  
  키보드(좌/우/ESC/q)로 구간을 이동하거나 종료할 수 있습니다.

- **test_ann.py**  
  MIT-BIH 데이터셋의 annotation(주석) 파일을 읽어 부정맥 심볼 및 레이블 정보를 출력합니다.  
  데이터셋의 annotation 구조를 확인하는 데 사용됩니다.

- **test_biosppy.py**  
  BioSPPy 라이브러리를 이용해 ECG 신호의 R-peak 검출 결과를 확인하고, 신호와 R-peak를 시각화합니다.  
  BioSPPy의 기본 동작을 테스트하는 예제입니다.

- **test_wfdb.py**  
  WFDB 라이브러리를 이용해 MIT-BIH 데이터셋의 신호와 annotation을 읽고, 신호와 annotation을 시각화합니다.  
  WFDB의 기본 동작을 테스트하는 예제입니다.

- **wfdb_biosppy_data_viewer.py**  
  WFDB와 BioSPPy를 이용해 ECG 신호를 읽고, 필터링 및 R-peak 검출 후 세그먼트 단위로 신호, 주석, R-peak, Recurrence Plot을 시각화합니다.  
  키보드(→, ←, q, ESC)로 세그먼트 탐색이 가능합니다.

---

각 스크립트는 MIT-BIH 데이터셋(`data/mit-bih-arrhythmia-database-1.0.0/`)을 기반으로 하며, ECG 신호 분석 및 머신러닝 데이터 준비, 시각화, 통계 분석 등에 활용됩니다.

## ESP32에서 TFLite를 활용하기 위한 파일 활용 방법

이 폴더의 파이썬 스크립트들은 MIT-BIH ECG 데이터셋을 기반으로 오토인코더(AutoEncoder) 모델 학습, 테스트 데이터셋 생성, 모델 평가 및 시각화에 사용됩니다.  
ESP32에서 TFLite 모델을 활용하려면 아래와 같은 순서로 파일을 활용할 수 있습니다.

### 1. 데이터셋 생성 및 전처리

- **gen_dataset_for_AE.py**  
  MIT-BIH 데이터셋에서 R-peak 중심의 ECG 세그먼트를 추출하고, 정상/이상 레이블을 부여하여 AutoEncoder 학습용 데이터셋(npz 파일)으로 저장합니다.  
  → 이 데이터셋을 사용해 오토인코더 모델을 학습할 수 있습니다.

### 2. 오토인코더 모델 학습

- **AE_ecg_for_MIT_BIH.py**  
  생성된 npz 데이터셋을 이용해 오토인코더(AutoEncoder) 모델을 학습할 수 있습니다.  
  학습이 완료되면 Keras 모델 파일(h5 등)로 저장할 수 있습니다.

> 참고:  
> 본 프로젝트의 오토인코더 모델 구조 및 이상감지(Anomaly Detection) 방법은  
> [TensorFlow 공식 튜토리얼 - 오토인코더를 이용한 이상감지](https://www.tensorflow.org/tutorials/generative/autoencoder?hl=ko)  
> 내용을 기반으로 구현되었습니다.

### 3. TFLite 변환

- **LiteRT_tran.py**  
  학습된 Keras 오토인코더 모델(h5 파일 등)을 TensorFlow Lite(.tflite) 포맷으로 변환합니다.  
  변환 과정에서 양자화(quantization) 등 옵션을 적용할 수 있으며, ESP32 등 임베디드 환경에서 사용할 수 있도록 모델을 최적화합니다.

### 4. 테스트 데이터셋 및 C 배열 생성

- **gen_micro_test_set.py**  
  생성된 npz 데이터셋에서 일부 샘플을 선택하여 TFLite 오토인코더로 추론을 수행하고, 결과를 시각화합니다.  
  또한, 테스트용 입력 데이터를 C 배열 형태로 출력하여 ESP32 코드에 바로 사용할 수 있도록 합니다.

### 5. ESP32에 적용

- 변환된 `.tflite` 모델 파일을 **C 배열(.cc 파일)**로 변환해야 합니다.
- **LiteRT_tran.py** 파일의 마지막 부분에 안내된 명령어를 사용하세요:

  ```
  xxd -i ecg_autoencoder_float16.tflite > ecg_autoencoder_float16.cc
  ```

- 이렇게 생성된 `ecg_autoencoder_float16.cc` 파일의 C 배열을 ESP32 프로젝트에 포함시킵니다.
- ESP32용 TFLite Micro 라이브러리에서 이 C 배열을 사용하여 모델을 메모리에 올리고, 추론을 실행할 수 있습니다.

---

**요약:**

1. `gen_dataset_for_AE.py`로 학습/테스트 데이터셋 생성
2. `AE_ecg_for_MIT_BIH.py`로 오토인코더 모델 학습
3. `LiteRT_tran.py`로 학습된 모델을 TFLite(.tflite)로 변환
4. `gen_micro_test_set.py`로 테스트 데이터 C 배열 생성
5. `.tflite` 모델과 C 배열을 ESP32 프로젝트에 포함
6. ESP32에서 TFLite Micro로 추론 실행
