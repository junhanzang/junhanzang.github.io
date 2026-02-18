---
title: "Working with tabular data in Python"
date: 2024-12-18 12:18:18
categories:
  - Article
---

<https://wandb.ai/mostafaibrahim17/ml-articles/reports/Working-with-tabular-data-in-Python--Vmlldzo4MTU4OTgx>

[Working with tabular data in Python](https://wandb.ai/mostafaibrahim17/ml-articles/reports/Working-with-tabular-data-in-Python--Vmlldzo4MTU4OTgx)

이 튜토리얼에서는 파이썬을 사용하여 표 형식(tabular) 데이터를 다루고 이를 통해 지진을 예측하는 방법을 탐구합니다.

**Mostafa Ibrahim**

작성일: 5월 31일  
마지막 수정일: 12월 18일

표 형식 데이터를 다루는 것은 데이터 과학 및 머신러닝 프로젝트에서 핵심 기술입니다. 트렌드를 분석하거나, 예측을 수행하거나, 숨겨진 패턴을 발견하는 등의 작업에서 구조화된 데이터는 종종 기반이 됩니다. 파이썬은 강력한 라이브러리와 도구를 통해 이러한 데이터를 조작하고 통찰을 얻는 과정을 더욱 쉽게 만들어 줍니다.

이 글에서는 파이썬을 활용해 표 형식 데이터를 다루는 방법을 탐구하고, 특히 지진 예측과 같은 실제 시나리오에 이러한 기술을 적용하는 데 초점을 맞출 것입니다. 글을 끝까지 읽으면 구조화된 데이터와 머신러닝을 활용하여 복잡한 문제를 해결하고 예측 모델을 구축하는 방법에 대한 탄탄한 이해를 얻게 될 것입니다.

### 목차

1. 파이썬에서의 표 형식 데이터 이해하기
2. 지진 예측을 위한 표 형식 데이터와 머신러닝
3. 지진 예측 모델 구축을 위한 단계별 튜토리얼
   - **1단계**: 파이썬 환경 설정
   - **2단계**: 데이터 수집 및 전처리
   - **3단계**: 탐색적 데이터 분석(EDA)
   - **4단계**: 특징 엔지니어링
   - **5단계**: 머신러닝 모델 구축
   - **6단계**: Weights & Biases를 활용한 모델 평가
4. 도전 과제와 미래 방향
5. 결론

### 파이썬에서의 표 형식 데이터 이해하기

표 형식 데이터는 현실 세계의 많은 데이터셋을 구성하는 구조화된 형식입니다. 스프레드시트처럼 행과 열로 구성되어 있어 조작, 분석, 시각화가 용이하며, 데이터 과학자와 머신러닝 엔지니어들에게 필수적인 도구로 자리 잡고 있습니다.

파이썬에서는 CSV 파일, Excel 스프레드시트, 또는 SQL 데이터베이스에 저장된 데이터 등 다양한 형식의 표 형식 데이터를 다룰 수 있습니다. pandas와 NumPy와 같은 라이브러리를 사용하면 표 형식 데이터를 효율적이고 간단하게 처리할 수 있습니다. 이러한 도구들은 구조화된 데이터셋을 손쉽게 정리하고 변환하며 탐색할 수 있는 모든 기능을 제공합니다.

이번 프로젝트에서는 Kaggle에서 제공하는 **Significant Earthquakes, 1965-2016** 데이터셋을 표 형식 데이터로 사용할 예정입니다.

![](/assets/images/posts/406/img.png)

### 지진 예측을 위한 표 형식 데이터와 머신러닝

표 형식 데이터에 대한 기본 개념을 이해했으니, 이제 이 구조화된 정보를 사용하여 머신러닝으로 지진을 예측하는 방법을 살펴보겠습니다. 데이터셋에는 지진의 규모, 위치, 깊이, 발생 시간 등의 주요 정보가 포함되어 있습니다. 이러한 요소를 분석함으로써 머신러닝 모델은 미래의 지진 발생을 예측하는 데 유용한 패턴을 발견할 수 있습니다.

예를 들어, 신경망(neural networks)은 과거 데이터를 학습하여 지진 활동을 예측할 수 있습니다. 또한, 선형 회귀(linear regression), 랜덤 포레스트(random forests), 그래디언트 부스팅(gradient boosting)과 같은 기법은 데이터를 분석하는 다양한 접근 방식을 제공합니다. 이러한 방법들을 조합하면 더 깊이 있는 통찰을 도출할 수 있습니다.

이번 프로젝트에서는 지도 학습(supervised learning)과 비지도 학습(unsupervised learning) 모두를 활용하여, 머신러닝이 지진 데이터에서 의미 있는 예측으로 변환될 수 있는 과정을 탐구할 것입니다. 파이썬과 그 강력한 라이브러리를 통해 모델을 구축하고 개선하며, 데이터 기반의 예측을 수행하는 과정을 함께 진행할 예정입니다.

### 파이썬을 사용한 지진 예측 모델 구축 단계별 튜토리얼

이제 이론적인 내용을 다뤘으니, 실습으로 넘어가 보겠습니다. 이번 튜토리얼에서는 파이썬과 표 형식 데이터를 활용하여 지진을 예측하는 AI 모델을 구축합니다.

우리는 **피드포워드 신경망(FNN)**을 모델로 사용할 것입니다. 걱정하지 마세요. 각 단계를 명확하고 흥미롭게 설명하며 안내하겠습니다. 시작해 봅시다!

### **1단계: 파이썬 환경 설정**

우선 필요한 도구와 라이브러리를 설치하고 파이썬 환경을 설정해야 합니다. 여기에는 TensorFlow, Keras, pandas, NumPy와 같은 패키지 설치 및 가져오기가 포함됩니다.  
아래 명령어를 따라 환경을 성공적으로 구성할 수 있도록 명확한 지침을 제공합니다.

```
!pip install basemap
!pip install scikeras
!pip install tensorflow
```

필요한 라이브러리를 가져옵니다.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

### **2단계: 데이터 수집 및 전처리**

다음 단계는 모델 학습에 사용할 지진 데이터를 수집하는 것입니다. 이 튜토리얼에서는 **Significant Earthquakes, 1965-2016** 데이터셋을 사용하지만, 관심 있는 다른 데이터셋을 사용해도 좋습니다.

#### **데이터셋 불러오기**

```
# 데이터셋 로드
data = pd.read_csv("database.csv")
print(data.isnull())  # null 값 확인
```

위 코드는 데이터를 파이썬으로 불러오고 결측값(null values)이 있는지 확인합니다. 데이터를 불러온 후에는 오류를 수정하고, 결측값을 채우며, 머신러닝 모델에 적합한 형태로 포맷합니다.

#### **결측값 처리**

```
# 결측값 처리
data = data.interpolate(method='linear', limit_direction='forward')
print(data)
```

위 코드는 선형 보간법(linear interpolation)을 사용해 결측값을 채우며, 데이터를 분석에 적합한 상태로 만듭니다. 이 과정을 통해 데이터를 정리하고 준비하게 됩니다.

### **3단계: 탐색적 데이터 분석(EDA)**

모델을 구축하기 전에 데이터를 이해하는 과정인 **탐색적 데이터 분석(EDA)**을 수행하는 것이 중요합니다. EDA를 통해 데이터의 트렌드, 이상치(outliers), 패턴을 시각화와 통계 요약을 통해 발견할 수 있습니다. 이러한 통찰은 더 효과적인 모델 구축에 도움을 줍니다.

#### **데이터셋의 통계 확인**

```
# 데이터셋 통계 확인
data.describe()
```

위 명령어는 데이터셋의 통계 요약을 제공합니다. 평균(mean), 표준편차(standard deviation), 범위(range)와 같은 주요 통계를 한눈에 확인할 수 있습니다.

![](/assets/images/posts/406/img_1.png)

위도 및 경도에 대한 분산형 차트

#### **지진 위치 시각화**

```
plt.figure(figsize=(10, 6))
plt.scatter(data['Longitude'], data['Latitude'], c=data['Magnitude'], cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Locations')
plt.show()
```

위 산점도(scatter plot)는 경도(Longitude)와 위도(Latitude)를 기준으로 지진 위치를 시각화하며, 지진의 규모(Magnitude)를 색상으로 표현합니다.

#### **지진 규모와 깊이의 관계**

```
# 규모와 깊이 간의 관계 시각화
plt.figure(figsize=(10, 6))
plt.scatter(data['Magnitude'], data['Depth'], alpha=0.7)
plt.xlabel('Magnitude')
plt.ylabel('Depth')
plt.title('Magnitude vs Depth')
plt.show()
```

위 코드는 지진의 **규모(Magnitude)**와 **깊이(Depth)** 간의 관계를 나타내는 산점도를 생성합니다. 이를 통해 규모와 깊이 사이의 상관관계를 시각적으로 탐구할 수 있습니다.

![](/assets/images/posts/406/img_2.png)

#### **Basemap을 활용한 지진 위치 지도 시각화**

```
from mpl_toolkits.basemap import Basemap

# Basemap 객체 생성
m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')

# 지진의 경도 및 위도 데이터
longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
x, y = m(longitudes, latitudes)

# 지도에 지진 위치 표시
fig = plt.figure(figsize=(12, 10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize=2, color='green')
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()
```

위 코드는 **Basemap 라이브러리**를 사용하여 전 세계 지진 위치를 지도 위에 시각화합니다. Basemap은 지리적 시각화를 간단하게 만들어주며, 지진 데이터를 지도 형식으로 표현하는 데 적합합니다.

![](/assets/images/posts/406/img_3.png)

### **EDA의 목적**

이와 같은 시각화는 데이터의 지리적 분포와 지진 규모, 깊이 간의 관계를 이해하는 데 도움을 줍니다. 이를 통해 데이터를 더 깊이 이해하고, 모델에 적합한 특징(feature)을 선정할 수 있습니다.

### **4단계: 특징 공학(Feature Engineering)**

모델의 입력으로 사용할 **특징(feature)**을 신중히 선택하고 준비해야 합니다. 이는 지진에 대한 이해를 활용하고 통계적 방법을 적용하는 과정을 포함합니다. 데이터셋에서 주요 특징은 **Date, Time, Latitude, Longitude, Depth, Magnitude**와 같은 열이지만, 추가 변수들을 탐구할 여지도 있습니다.

#### **주요 특징 선택**

```
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
data.head()
```

위 코드는 데이터셋에서 모델 학습에 필요한 주요 열만 선택합니다.

#### **날짜와 시간 처리**

현재 **Date**와 **Time**은 문자열 형식(string)으로 되어 있습니다. 이를 모델이 더 효과적으로 사용할 수 있도록, **datetime 형식**으로 변환해야 합니다.

```
import datetime
import time

# 타임스탬프 생성
timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d + ' ' + t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        # ValueError 처리
        timestamp.append('ValueError')

# 타임스탬프를 데이터프레임에 추가
timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values
```

위 코드는 **Date**와 **Time**을 결합해 타임스탬프 형식으로 변환합니다. 변환된 타임스탬프는 숫자 형식으로 저장되므로, 모델이 처리하기에 적합합니다.

#### **불필요한 열 제거 및 데이터 정리**

```
final_data = data.drop(['Date', 'Time'], axis=1)  # Date와 Time 열 제거
final_data = final_data[final_data.Timestamp != 'ValueError']  # ValueError 제거
final_data.head()
```

- **Date**와 **Time** 열은 더 이상 필요하지 않으므로 제거합니다.
- 변환 과정에서 발생한 **ValueError** 행을 필터링하여 데이터셋을 정리합니다.

이 과정을 통해 모델이 효율적으로 학습할 수 있도록 데이터를 준비했습니다. **Latitude, Longitude, Depth, Magnitude, Timestamp**와 같은 정리된 특징은 지진 예측을 위한 입력값으로 사용됩니다.

### **5단계: 머신러닝 모델 구축**

데이터 준비와 특징 설정이 완료되었으니, 이제 지진 예측 모델을 구축할 차례입니다. 우리는 신경망(neural network)을 활용하여 지진 데이터의 패턴을 찾아내고, 모델 구조 정의, 학습, 그리고 파라미터 튜닝 과정을 진행할 것입니다.

#### **데이터 분리**

먼저 데이터를 훈련 세트와 테스트 세트로 나누어, 실제 환경을 시뮬레이션하며 모델을 평가합니다.

```
X = final_data[['Timestamp', 'Latitude', 'Longitude']]  # 입력 데이터
y = final_data[['Magnitude', 'Depth']]  # 출력 데이터

from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.keras import WandbEvalCallback, WandbMetricsLogger

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### **데이터 정규화**

모델이 효과적으로 학습할 수 있도록 데이터를 정규화합니다.

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 훈련 데이터 스케일링
X_test = scaler.transform(X_test)  # 테스트 데이터 스케일링
```

#### **모델 정의**

신경망 모델을 정의하고, 지진 데이터의 패턴을 학습하기 위한 계층을 설계합니다.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # 첫 번째 은닉층
model.add(Dropout(0.2))  # 드롭아웃
model.add(Dense(64, activation='relu'))  # 두 번째 은닉층
model.add(Dropout(0.2))  # 드롭아웃
model.add(Dense(32, activation='relu'))  # 세 번째 은닉층
model.add(Dense(2))  # 출력층 (2개의 뉴런: Magnitude, Depth)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 요약 출력
model.summary()
```

![](/assets/images/posts/406/img_4.png)

#### **모델 학습**

설계한 모델을 학습시켜 데이터에서 패턴을 학습합니다.

```
history = model.fit(
    X_train, 
    y_train, 
    epochs=200, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=1
)
```

#### **모델 평가**

학습된 모델을 테스트 데이터로 평가하여 성능을 확인합니다.

```
test_loss = model.evaluate(X_test, y_test, verbose=1) 
print(f'Test Loss: {test_loss}')
```

#### **예측 및 평가 지표 계산**

테스트 데이터에 대한 예측을 수행하고, 성능 지표인 평균 제곱 오차(Mean Squared Error, MSE)를 계산합니다.

```
from sklearn.metrics import mean_squared_error

# 예측 수행
y_pred = model.predict(X_test)

# 평균 제곱 오차 계산
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

![](/assets/images/posts/406/img_5.png)

### **6단계: Weights & Biases를 사용한 모델 평가**

모델 구축이 끝났다고 해서 여정이 끝난 것은 아닙니다. 예측의 정확성을 최대화하려면 모델 성능을 철저히 평가해야 합니다.

이 과정에서 **Weights & Biases(W&B)**가 중요한 역할을 합니다. 강력한 추적 및 시각화 도구를 통해 **정확도(accuracy)**, **손실(loss)** 등의 주요 지표를 기록하고, 이를 바탕으로 모델을 반복적으로 미세 조정하고 개선할 수 있습니다.

```
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
wandb.init(config={"hyper": "parameter"})

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(self, validation_data, data_table_columns, pred_table_columns, num_samples=100):
        super().__init__(data_table_columns, pred_table_columns)
        self.x = validation_data[0]
        self.y = validation_data[1]
        self.num_samples = num_samples

    def add_ground_truth(self, logs=None):
        for idx, (features, label) in enumerate(zip(self.x[:self.num_samples], self.y[:self.num_samples])):
            self.data_table.add_data(idx, features.tolist(), label[0], label[1])

    def add_model_predictions(self, epoch, logs=None):
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],  # features
                self.data_table_ref.data[idx][1],  # true magnitude
                self.data_table_ref.data[idx][2],  # true depth
                pred[0],  # predicted magnitude
                pred[1]   # predicted depth
            )

    def _inference(self):
        preds = []
        for features in self.x[:self.num_samples]:
            pred = self.model.predict(tf.expand_dims(features, axis=0), verbose=0)
            preds.append(pred[0])
        return preds

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=32,
    verbose=1,
    callbacks=[
        early_stopping,
        reduce_lr,
       WandbMetricsLogger(),
        WandbClfEvalCallback(
            validation_data=(X_test, y_test),
            data_table_columns=["idx", "features", "true_magnitude", "true_depth"],
            pred_table_columns=["epoch", "features", "true_magnitude", "true_depth", "pred_magnitude", "pred_depth"],
        ),
    ]
)
```

![](/assets/images/posts/406/img_6.png)

![](/assets/images/posts/406/img_7.png)

![](/assets/images/posts/406/img_8.png)

#### **Weights & Biases의 주요 기능**

- **모델 버전 관리**: 각 실험에서 사용된 모델을 저장하고 비교할 수 있습니다.
- **실험 추적**: 실험의 모든 설정, 결과 및 지표를 한눈에 확인할 수 있습니다.
- **협업 도구**: 결과를 팀원과 쉽게 공유하고 협업할 수 있습니다.
- **모델 배포**: 실제 환경에서 지진 예측에 모델을 쉽게 배포할 수 있습니다.

이러한 기능은 우리가 모델을 지속적으로 최적화하고 정교화하는 데 도움을 줍니다. Weights & Biases를 활용하면 더 나은 성능의 모델을 구축하고, 결과를 비교 및 공유하며, 궁극적으로 실제 문제에 적용할 준비가 된 모델을 만들 수 있습니다.

### **도전 과제와 향후 방향**

이번 프로젝트를 진행하며, 지진 예측을 위해 파이썬에서 표 형식 데이터를 효과적으로 사용하는 과정에서 여러 도전 과제를 마주했습니다.

초기에는 모델을 100 에포크(epoch) 동안 학습시켜도 좋은 결과를 얻지 못했으며, 300 에포크로 늘리자 과적합(overfitting)이 발생했습니다. 이를 해결하기 위해 **얼리 스토핑(Early Stopping)** 기법을 도입하여 과적합 없이 효과적으로 모델을 학습할 수 있었습니다.

또한, 모델의 은닉층에 뉴런 수를 증가시켜 데이터를 더 잘 학습하도록 설계했지만, 이로 인해 모델의 복잡성이 증가하면서 과적합 가능성이 커졌습니다. 이를 완화하기 위해 **드롭아웃(Dropout)** 레이어를 추가하여 정규화를 강화했습니다.

향후에는 **실시간 지진 센서 데이터**를 활용하여 동적이고 지속적인 업데이트가 가능한 예측 모델을 구현할 수 있을 것입니다. 또한, **트랜스포머(Transformer)** 또는 **어텐션(attention)** 기반 아키텍처와 같은 고급 모델을 탐구하여 표 형식 데이터에서 장기 의존성과 복잡한 패턴을 더 효과적으로 학습할 가능성을 모색할 수 있습니다.

추가적으로, **지질학적 데이터**와 **판구조론 데이터**와 같은 여러 데이터 소스를 결합하면 더 포괄적인 지진 예측 모델을 구축할 수 있을 것입니다. 이 과정에서 다양한 분야의 전문가 간 협력이 중요하며, 이를 통해 표 형식 데이터와 머신러닝을 활용한 지진 예측 기술을 한 단계 더 발전시킬 수 있습니다.

### **결론**

이번 프로젝트에서는 파이썬과 머신러닝을 활용하여 지진 예측을 위한 표 형식 데이터를 처리하고 분석하는 방법을 탐구했습니다. 진행 과정에서 과적합과 같은 도전을 해결하고, 모델 구조를 개선하며, 구조화된 데이터를 통해 의미 있는 통찰을 도출하는 방법을 배웠습니다.

여기서 배운 데이터 정리, 준비, 모델링 기술은 금융 예측, 의료 데이터 분석 등 **구조화된 데이터**를 사용하는 다양한 분야에 적용할 수 있습니다. 이 튜토리얼에서 익힌 기술과 도구를 활용하면 복잡한 데이터셋을 다루고 영향력 있는 예측을 수행할 수 있습니다.

표 형식 데이터를 활용할 수 있는 가능성은 매우 넓으며, 지속적인 연습을 통해 이 기술을 다양한 실제 문제 해결에 적용할 수 있을 것입니다.
