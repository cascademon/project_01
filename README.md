# ✈️ AI 기반 항공 고객 만족도 분석 프로젝트

고객의 항공 서비스 이용 데이터를 기반으로  
고객 만족 여부를 예측하고,  
서비스 품질 향상을 위한 핵심 요인을 분석한 머신러닝 및 딥러닝 프로젝트입니다.

---

# 📌 프로젝트 개요

본 프로젝트는 항공 고객 만족도 데이터를 활용하여  
고객 만족 여부를 예측하고,  
어떤 서비스 요소가 고객 경험에 가장 큰 영향을 미치는지 분석하는 것을 목표로 진행되었습니다.

단순한 모델 정확도 향상보다  
데이터를 기반으로 실제 서비스 개선 방향을 도출하는 과정에 집중하였습니다.

---

# 🎯 프로젝트 목표

- 고객 만족 여부 예측 모델 구축
- 서비스 품질 영향 요소 분석
- 연령대별 만족 요인 차이 분석
- 데이터 기반 의사결정 경험 습득
- 머신러닝 모델 성능 비교 및 개선

---

# 🛠 사용 기술 스택

## Language
- Python

## Data Analysis
- Pandas
- NumPy

## Visualization
- Matplotlib
- Seaborn

## Machine Learning / Deep Learning
- Scikit-learn
- TensorFlow
- Keras

## Environment
- Google Colab
- Jupyter Notebook

---

# 📂 프로젝트 구조

```bash
Airline-Customer-Satisfaction/
│
├── 01 - 탐색적 데이터 분석.ipynb
├── 02 - 기본 모델링.ipynb
├── 03 - 모델 추가 학습.ipynb
│
├── data.csv
├── survey.csv
├── new_train.csv
├── new_test.csv
│
├── base_model.keras
├── scaler.pkl
│
└── README.md
```

---

# 📊 데이터 분석 과정

## 1️⃣ 탐색적 데이터 분석(EDA)

- 데이터 구조 확인
- 결측치 및 이상치 확인
- 변수별 분포 분석
- 고객 만족/불만족 비율 분석
- 상관관계 분석
- 서비스 항목별 시각화 진행

---

## 2️⃣ 데이터 전처리

- 범주형 데이터 인코딩
- 데이터 정규화(StandardScaler)
- 학습/테스트 데이터 분리
- 모델 입력 데이터 가공

---

## 3️⃣ 머신러닝 모델링

### 사용 모델
- DecisionTreeClassifier
- RandomForestClassifier

### 진행 내용
- 모델 학습
- 성능 비교
- 정확도 분석
- 변수 중요도 확인
- 예측 결과 평가

---

# 📈 주요 분석 결과

## ✔ 고객 만족도에 영향을 주는 핵심 요소 확인

다음 요소들이 고객 만족도에 큰 영향을 미치는 것으로 나타났습니다.

- 좌석 편의성
- 기내 서비스 품질
- 청결도
- 기내 와이파이 서비스
- 비행 지연 시간

---

## ✔ 연령대별 중요 요소 차이 발견

연령대에 따라 중요하게 생각하는 서비스 요소가 서로 다르게 나타났습니다.

예시:
- 젊은 연령층 → 와이파이/엔터테인먼트 중요
- 높은 연령층 → 좌석 편의성/청결도 중요

---

## ✔ 데이터 기반 서비스 개선 가능성 확인

단순 예측 모델 구축을 넘어  
고객 경험 개선 방향을 데이터 기반으로 제안할 수 있음을 확인했습니다.

---

# 🤖 모델링 결과

| 모델 | 특징 |
|---|---|
| Random Forest | 변수 중요도 해석 가능 |
| DecisionTreeClassifier | // |

---

# 🧠 딥러닝 모델 구축

본 프로젝트에서는 머신러닝(Random Forest)뿐 아니라  
딥러닝 모델을 함께 구축하여  
고객 만족도 예측 성능을 비교 분석하였습니다.

---

## 📌 딥러닝 모델 구조

TensorFlow/Keras 기반 Sequential 모델을 사용하였습니다.

### 사용 Layer
- Dense Layer
- Batch Normalization
- Activation(ReLU)
- Dropout

---

## 📌 모델 학습 과정

### 학습 단계
1. 데이터 정규화 진행
2. 학습/검증 데이터 분리
3. DNN 모델 설계
4. 모델 학습 및 검증
5. 성능 비교 및 추가 학습 진행

---

## 📌 사용 기술

```python
TensorFlow
Keras
Sequential
Dense
BatchNormalization
Dropout
Adam Optimizer
```

---

## 📌 딥러닝 모델 성능 개선

기본 모델 학습 이후 추가 학습을 통해  
과적합(Overfitting)을 줄이고 예측 성능 개선을 시도하였습니다.

### 적용 내용
- Epoch 조정
- Layer 구조 변경
- BatchNormalization 적용
- Activation 함수 조정
- Validation 기반 성능 비교

---

## 📌 모델 저장 및 재사용

학습 완료된 모델은 `.keras` 형식으로 저장하여  
추후 재사용 및 예측 시스템 구축이 가능하도록 구성하였습니다.

### 저장 파일
```bash
base_model.keras
base_model (1).keras
```

---

# 📊 머신러닝 vs 딥러닝 비교

| 구분 | Random Forest | DNN |
|---|---|---|
| 장점 | 변수 중요도 해석 가능 | 복잡한 패턴 학습 가능 |
| 특징 | 해석 중심 | 예측 성능 중심 |
| 활용 | 중요 변수 분석 | 고객 만족 예측 |

---

# 🔥 프로젝트 핵심 차별점

본 프로젝트는 단순 데이터 분석에 그치지 않고,

- 머신러닝 기반 변수 중요도 분석
- 딥러닝 기반 고객 만족 예측
- 모델 성능 비교
- 서비스 개선 방향 도출

까지 연결한 데이터 기반 AI 프로젝트입니다.

# 💡 프로젝트를 통해 배운 점

- 데이터 전처리의 중요성
- 모델별 성능 및 특성 차이 이해
- 변수 중요도 분석 경험
- 데이터 기반 문제 해결 사고 방식 습득
- 서비스 개선 관점의 데이터 해석 경험

---

# 🚀 향후 개선 방향

- XGBoost / LightGBM 모델 추가 비교
- SHAP 기반 Explainable AI 적용
- 실시간 예측 시스템 구축
- 대시보드 시각화 구현
- 사용자 맞춤형 서비스 추천 기능 확장

---

# ▶ 실행 방법

## 패키지 설치

```bash
pip install -r requirements.txt
```

---

## Jupyter Notebook 실행

```bash
jupyter notebook
```

또는

```bash
Google Colab에서 .ipynb 파일 실행
```

---

# 📷 프로젝트 예시 화면

- 데이터 시각화 그래프
- 변수 중요도 분석
- 모델 성능 비교 결과
- 만족도 분포 시각화

(추후 이미지 추가 예정)

---

# 📌 프로젝트 기대 효과

- 고객 경험 개선 방향 도출
- 데이터 기반 서비스 전략 수립
- 항공 서비스 품질 향상 지원
- 머신러닝 기반 고객 만족 예측 활용 가능

---

# 👨‍💻 작성자

김남효  
Data Analysis / Machine Learning Project

GitHub Portfolio Project
