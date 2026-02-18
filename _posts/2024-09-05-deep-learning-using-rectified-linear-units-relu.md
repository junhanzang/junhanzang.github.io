---
title: "Deep Learning using Rectified Linear Units (ReLU)"
date: 2024-09-05 21:29:19
categories:
  - 인공지능
---

<https://arxiv.org/abs/1803.08375>

[Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/abs/1803.08375)

**요약**

우리는 심층 신경망(DNN)에서 ReLU(Rectified Linear Unit)를 분류 함수로 사용하는 방법을 소개합니다. 일반적으로 ReLU는 DNN에서 활성화 함수로 사용되며, Softmax 함수가 분류 함수로 사용됩니다. 그러나 Softmax 외의 분류 함수를 사용하는 몇 가지 연구가 있었으며, 본 연구는 그러한 연구들에 추가되는 내용입니다. 우리는 신경망의 마지막에서 두 번째 층의 활성화 h\_(n−1) 을 가져와 가중치 매개변수 θ 로 곱하여 원시 점수 o\_i 를 얻습니다. 그런 다음 원시 점수 o\_i 를 0으로 임계값 처리합니다. 즉, f(o)max(0, o\_i) 이며, 여기서 f(o) 는 ReLU 함수입니다. 우리는 argmax 함수를 통해 클래스 예측 y^ 를 제공합니다. 즉, argmax f(x) 입니다.

**키워드**: 인공지능, 인공신경망, 분류, 합성곱 신경망, 딥러닝, 심층 신경망, 피드포워드 신경망, 기계학습, ReLU, Softmax, 지도학습

**1. 서론**

딥러닝 접근 방식을 사용한 여러 연구들이 이미지 분류(Krizhevsky et al., 2012), 자연어 처리(Wen et al., 2015), 음성 인식(Chorowski et al., 2015), 텍스트 분류(Yang et al., 2016)와 같은 다양한 작업에서 최첨단 성능을 달성했다고 주장해왔습니다. 이러한 딥러닝 모델들은 일반적으로 Softmax 함수를 분류 층으로 사용합니다.

그러나 Softmax 이외의 분류 함수를 사용하는 여러 연구(Agarap, 2017; Alalshekmubarak 및 Smith, 2013; Tang, 2013)가 있었으며, 본 연구도 그 중 하나입니다.

이 논문에서는 딥러닝 모델의 분류 층에 ReLU(Rectified Linear Unit)를 사용하는 방법을 소개합니다. 이 접근 방식은 본 연구에서 제시된 독창성으로, ReLU는 일반적으로 심층 신경망의 은닉층에서 활성화 함수로 사용됩니다. 우리는 신경망의 마지막에서 두 번째 층의 활성화를 가져와 이를 통해 ReLU 분류 층의 가중치 매개변수를 역전파(backpropagation)로 학습합니다.

우리는 MNIST(LeCun et al., 2010), Fashion-MNIST(Xiao et al., 2017), 그리고 Wisconsin 진단 유방암(WDBC)(Wolberg et al., 1992) 분류에서 DL-ReLU 모델과 DL-Softmax 모델의 예측 성능을 비교하여 보여줍니다. 네트워크 가중치 매개변수를 학습하기 위해 Adam(Kingma 및 Ba, 2014) 최적화 알고리즘을 사용합니다.

이 연구는 딥러닝에서 분류 함수로서 ReLU의 가능성을 탐구하고 Softmax와의 성능 비교를 중점으로 합니다.

**2. 방법론**

**2.1. 머신 인텔리전스 라이브러리**  
이 연구에서는 딥러닝 알고리즘을 구현하기 위해 Keras(Chollet et al., 2015)와 Google TensorFlow(Abadi et al., 2015) 백엔드를 사용하였으며, 이 외에도 다른 과학 계산 라이브러리인 matplotlib(Hunter, 2007), numpy(Walt et al., 2011), scikit-learn(Pedregosa et al., 2011)의 도움을 받았습니다.

**2.2. 데이터셋**  
이 절에서는 실험에 사용된 딥러닝 모델에 적용된 데이터셋을 설명합니다.

**2.2.1. MNIST**  
MNIST(LeCun et al., 2010)는 딥러닝 모델의 성능 평가를 위한 표준 데이터셋 중 하나입니다. 이 데이터셋은 60,000개의 훈련 샘플과 10,000개의 테스트 샘플을 포함한 10개의 클래스로 구성된 분류 문제입니다. 모든 이미지는 그레이스케일이며, 해상도는 28×28입니다.

**2.2.2. Fashion-MNIST**  
Xiao 등(2017)(Xiao et al., 2017)은 기존 MNIST에 대한 대안으로 새로운 Fashion-MNIST 데이터셋을 제시했습니다. 이 데이터셋은 10개의 클래스로 구성된 70,000개의 패션 제품 이미지를 포함하고 있으며, 각 클래스당 7,000개의 28×28 해상도의 그레이스케일 이미지로 이루어져 있습니다.

**2.2.3. 위스콘신 진단 유방암(WDBC)**  
WDBC 데이터셋(Wolberg et al., 1992)은 유방 종양의 세침 흡인(FNA)에서 얻은 디지털화된 이미지로부터 계산된 특징을 포함합니다. 이 데이터셋에는 총 569개의 데이터 포인트가 있으며, 이 중 212개는 악성, 357개는 양성으로 분류됩니다.

**2.3. 데이터 전처리**  
우리는 데이터셋의 특징을 아래 식 (1)을 사용해 정규화했습니다.

![](/assets/images/posts/274/img.png)

여기서 **X**는 데이터셋의 특징을, **μ**는 각 데이터셋 특징 \*\*x(i)\*\*의 평균값을, **σ**는 해당 특징의 표준 편차를 나타냅니다. 이 정규화 기법은 scikit-learn의 StandardScaler(Pedregosa et al., 2011)를 사용해 구현했습니다. MNIST와 Fashion-MNIST의 경우, 차원 축소를 위해 주성분 분석(PCA)을 사용했습니다. 즉, 이미지 데이터의 대표적인 특징을 선택하기 위해 PCA(Pedregosa et al., 2011)를 사용했습니다.

**2.4. 모델**  
우리는 피드포워드 신경망(FFNN)과 합성곱 신경망(CNN)을 구현했으며, 두 모델 모두 두 가지 다른 분류 함수를 가졌습니다. (1) Softmax, (2) ReLU.

**2.4.1. Softmax**  
딥러닝 분류 문제의 해결책으로 일반적으로 Softmax 함수가 분류 함수(마지막 층)로 사용됩니다. Softmax 함수는 **K**개의 클래스에 대해 이산 확률 분포를 정의하며, 이는 다음과 같이 나타낼 수 있습니다:

![](/assets/images/posts/274/img_1.png)

만약 **x**가 신경망의 마지막에서 두 번째 층의 활성화 값이고, **θ**가 Softmax 층의 가중치 매개변수라면, 우리는 **o**를 Softmax 층의 입력으로 가질 수 있습니다.

![](/assets/images/posts/274/img_2.png)

따라서 다음과 같이 됩니다.

![](/assets/images/posts/274/img_3.png)

따라서 예측된 클래스는 다음과 같습니다.

![](/assets/images/posts/274/img_4.png)

**2.4.2. Rectified Linear Units (ReLU)**  
ReLU는 2000년 Hahnloser 등(Hahnloser et al., 2000)이 제안한 활성화 함수로, 생물학적 및 수학적 근거가 강합니다. 2011년에는 ReLU가 심층 신경망 훈련을 더욱 개선할 수 있음이 입증되었습니다. ReLU는 값을 0으로 임계값 처리하여 작동합니다. 즉, f(x) = max(0, x) 입니다. 간단히 말해, **x**가 0보다 작으면 0을 출력하고, **x**가 0 이상이면 선형 함수 출력을 반환합니다. (Figure 1에서 시각적 표현을 참조하십시오.)

![](/assets/images/posts/274/img_5.png)

**그림 1.** Rectified Linear Unit (ReLU) 활성화 함수는 입력값 **x**가 0보다 작을 때 출력으로 0을 반환하고, **x**가 0보다 클 때는 기울기가 1인 선형 출력을 생성합니다.

우리는 ReLU를 신경망의 각 은닉층에서 활성화 함수로 사용할 뿐만 아니라, 네트워크의 마지막 층에서 분류 함수로도 사용할 것을 제안합니다. 따라서 ReLU 분류기의 예측 클래스는 \*\*y^\*\*가 됩니다.

![](/assets/images/posts/274/img_6.png)

**2.4.3. ReLU를 사용한 딥러닝**  
ReLU는 일반적으로 신경망에서 활성화 함수로 사용되며, Softmax는 분류 함수로 사용됩니다. 그러한 네트워크는 Softmax 교차 엔트로피 함수를 사용하여 신경망의 가중치 매개변수 **θ**를 학습합니다. 이 논문에서는 해당 손실 함수를 계속 사용하지만, 예측 유닛에 ReLU를 사용하는 차별점을 두었습니다(식 6 참조). **θ** 매개변수는 ReLU 분류기로부터의 그래디언트를 역전파하여 학습됩니다. 이를 달성하기 위해, 우리는 ReLU 기반의 교차 엔트로피 함수(식 7 참조)를 신경망의 마지막에서 두 번째 층의 활성화 값에 대해 미분합니다.

![](/assets/images/posts/274/img_7.png)

입력 **x**를 마지막에서 두 번째 활성화 출력 **h**로 대체합니다.

![](/assets/images/posts/274/img_8.png)

역전파 알고리즘(식 8 참조)은 기존의 Softmax 기반 딥러닝 신경망과 동일합니다.

![](/assets/images/posts/274/img_9.png)

알고리즘 1은 DL-ReLU 모델에 대한 기본적인 경사 하강법 알고리즘을 보여줍니다.

![](/assets/images/posts/274/img_10.png)

일부 실험에서 우리는 DL-ReLU 모델이 Softmax 기반 모델과 비슷한 성능을 보인다는 것을 발견했습니다.

**2.5. 데이터 분석**  
DL-ReLU 모델의 성능을 평가하기 위해 다음 지표를 사용했습니다.

(1) 교차 검증 정확도 및 표준 편차: 10-겹 교차 검증(CV) 실험의 결과.  
(2) 테스트 정확도: 훈련된 모델이 보지 못한 데이터에서의 성능.  
(3) 재현율(Recall), 정밀도(Precision), F1-스코어: 클래스 예측에 대한 분류 통계.  
(4) 혼동 행렬(Confusion Matrix): 분류 성능을 설명하는 표.

**3. 실험**  
이 연구의 모든 실험은 Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB DDR3 RAM, NVIDIA GeForce GTX 960M 4GB DDR5 GPU를 탑재한 노트북 컴퓨터에서 수행되었습니다.  
표 1은 실험에서 사용된 VGG 유사 CNN 구조(Keras(Chollet et al., 2015)에서 가져옴)를 보여줍니다. 마지막 층인 **dense\_2**는 실험에서 Softmax 분류기와 ReLU 분류기를 사용했습니다.  
Softmax 기반 모델과 ReLU 기반 모델은 동일한 하이퍼파라미터를 가졌으며, 해당 설정은 프로젝트 저장소의 Jupyter Notebook에서 확인할 수 있습니다: <https://github.com/AFAgarap/relu-classifier>.

**표 1.** Keras(Chollet et al., 2015)에서 가져온 VGG 유사 CNN 구조.

![](/assets/images/posts/274/img_11.png)

표 2는 실험에 사용된 피드포워드 신경망(FFNN)의 구조를 보여줍니다. 마지막 층인 **dense\_6**는 실험에서 Softmax 분류기와 ReLU 분류기를 사용했습니다.

**표 2.** FFNN의 구조.

![](/assets/images/posts/274/img_12.png)

![](/assets/images/posts/274/img_13.png)

**3.1. MNIST**  
우리는 표 1과 표 2에 정의된 CNN 및 FFNN을 정규화되고 PCA로 차원 축소된 특징에 대해 구현했습니다. 즉, **28×28**(784) 차원에서 **16×16**(256) 차원으로 축소했습니다.  
MNIST 분류를 위한 두 개의 은닉층을 가진 FFNN을 훈련하면서, 우리는 표 3에 설명된 결과를 발견했습니다.

**표 3. MNIST 분류.**  
FFNN-Softmax 모델과 FFNN-ReLU 모델의 정확도(%) 비교. 훈련 교차 검증은 10번의 분할에서 얻은 평균 교차 검증 정확도입니다. 테스트 정확도는 보지 않은 데이터에서의 성능을 나타냅니다. 정밀도(Precision), 재현율(Recall), F1-스코어는 보지 않은 데이터에서의 성능을 나타냅니다.

![](/assets/images/posts/274/img_14.png)

Softmax 기반 FFNN이 ReLU 기반 FFNN보다 약간 더 높은 테스트 정확도를 가졌음에도 불구하고, 두 모델 모두 F1-스코어가 0.98이었습니다. 이러한 결과는 FFNN-ReLU가 기존의 FFNN-Softmax와 동등한 성능을 보인다는 것을 의미합니다.

![](/assets/images/posts/274/img_15.png)

**그림 2.** MNIST 분류에서 FFNN-ReLU의 혼동 행렬.

![](/assets/images/posts/274/img_16.png)

**그림 3.** MNIST 분류에서 FFNN-Softmax의 혼동 행렬.

그림 2와 3은 두 모델이 MNIST 분류에서 10개의 클래스에 대해 예측 성능을 보여줍니다. 혼동 행렬에서 정확한 예측의 값들이 균형 잡혀 있는 것으로 보이며, 일부 클래스에서는 ReLU 기반 FFNN이 Softmax 기반 FFNN보다 더 나은 성능을 보였고, 반대로 Softmax 기반 FFNN이 더 나은 성능을 보인 경우도 있었습니다.

MNIST 분류를 위한 VGG 유사 CNN(Chollet et al., 2015)을 훈련하는 과정에서, 우리는 표 4에 설명된 결과를 발견했습니다.

**표 4.** MNIST 분류. CNN-Softmax 모델과 CNN-ReLU 모델의 정확도(%) 비교. 훈련 교차 검증은 10번의 분할에서 얻은 평균 교차 검증 정확도입니다. 테스트 정확도는 보지 않은 데이터에서의 성능을 나타냅니다. 정밀도(Precision), 재현율(Recall), F1-스코어는 보지 않은 데이터에서의 성능을 나타냅니다.

![](/assets/images/posts/274/img_17.png)

CNN-ReLU는 CNN-Softmax보다 성능이 낮았습니다. 이는 교차 검증에서 훈련 정확도를 확인한 결과, CNN-ReLU가 수렴 속도가 더 느렸기 때문입니다(표 5 참조). 그러나 느린 수렴 속도에도 불구하고 CNN-ReLU는 90% 이상의 테스트 정확도를 달성할 수 있었습니다. 물론 CNN-Softmax의 테스트 정확도보다 약 4% 낮지만, 추가적인 최적화를 통해 CNN-ReLU가 CNN-Softmax와 동등한 성능을 달성할 수 있을 것입니다.

**표 5.** MNIST 분류에서 CNN-ReLU에 대한 10-겹 교차 검증의 각 폴드별 훈련 정확도 및 손실.

![](/assets/images/posts/274/img_18.png)

![](/assets/images/posts/274/img_19.png)

**그림 4.** MNIST 분류에서 CNN-ReLU의 혼동 행렬.

![](/assets/images/posts/274/img_20.png)

**그림 5.** MNIST 분류에서 CNN-Softmax의 혼동 행렬.

그림 4와 5는 두 모델이 MNIST 분류에서 10개의 클래스에 대해 예측 성능을 보여줍니다. CNN-Softmax가 CNN-ReLU보다 더 빠르게 수렴했기 때문에, 클래스별로 가장 많은 정확한 예측을 기록했습니다.

**3.2. Fashion-MNIST**  
우리는 표 1과 표 2에 정의된 CNN 및 FFNN을 정규화되고 PCA로 차원 축소된 특징에 대해 구현했습니다. 즉, **28×28**(784) 차원에서 **16×16**(256) 차원으로 축소했습니다. MNIST의 차원 축소는 공정한 비교를 위해 Fashion-MNIST에서도 동일하게 적용되었습니다. 이 판단은 추가 연구를 통해 도전될 수 있습니다.  
Fashion-MNIST 분류를 위한 두 개의 은닉층을 가진 FFNN을 훈련하는 과정에서, 우리는 표 6에 설명된 결과를 발견했습니다.

**표 6. Fashion-MNIST 분류.**  
FFNN-Softmax 모델과 FFNN-ReLU 모델의 정확도(%) 비교. 훈련 교차 검증은 10번의 분할에서 얻은 평균 교차 검증 정확도입니다. 테스트 정확도는 보지 않은 데이터에서의 성능을 나타냅니다. 정밀도(Precision), 재현율(Recall), F1-스코어는 보지 않은 데이터에서의 성능을 나타냅니다.

![](/assets/images/posts/274/img_21.png)

Softmax 기반 FFNN이 ReLU 기반 FFNN보다 약간 더 높은 테스트 정확도를 가졌음에도 불구하고, 두 모델 모두 F1-스코어가 0.89였습니다. 이러한 결과는 FFNN-ReLU가 기존의 FFNN-Softmax와 동등한 성능을 보인다는 것을 의미합니다.

![](/assets/images/posts/274/img_22.png)

**그림 6.** Fashion-MNIST 분류에서 FFNN-ReLU의 혼동 행렬.

![](/assets/images/posts/274/img_23.png)

**그림 7.** Fashion-MNIST 분류에서 FFNN-Softmax의 혼동 행렬.

그림 6과 7은 두 모델이 Fashion-MNIST 분류에서 10개의 클래스에 대해 예측 성능을 보여줍니다. 혼동 행렬에서 정확한 예측 값들이 균형 잡혀 있는 것으로 보이며, 일부 클래스에서는 ReLU 기반 FFNN이 Softmax 기반 FFNN보다 더 나은 성능을 보였고, 반대로 Softmax 기반 FFNN이 더 나은 성능을 보인 경우도 있었습니다.

Fashion-MNIST 분류를 위한 VGG 유사 CNN(Chollet et al., 2015)을 훈련하는 과정에서, 우리는 표 7에 설명된 결과를 발견했습니다.

**표 7. Fashion-MNIST 분류.**  
CNN-Softmax 모델과 CNN-ReLU 모델의 정확도(%) 비교. 훈련 교차 검증은 10번의 분할에서 얻은 평균 교차 검증 정확도입니다. 테스트 정확도는 보지 않은 데이터에서의 성능을 나타냅니다. 정밀도(Precision), 재현율(Recall), F1-스코어는 보지 않은 데이터에서의 성능을 나타냅니다.

![](/assets/images/posts/274/img_24.png)

MNIST 분류에서의 결과와 유사하게, CNN-ReLU는 교차 검증에서 훈련 정확도를 확인한 결과(표 8 참조), CNN-Softmax보다 수렴 속도가 느려 성능이 낮았습니다. 테스트 정확도는 약간 낮았지만, CNN-ReLU는 CNN-Softmax와 동일한 F1-스코어 0.86을 기록했습니다. 이는 MNIST 분류에서의 결과와도 유사합니다.

**표 8.** Fashion-MNIST 분류를 위한 CNN-ReLU의 10-겹 교차 검증에서 폴드별 훈련 정확도 및 손실.

![](/assets/images/posts/274/img_25.png)

![](/assets/images/posts/274/img_26.png)

**그림 8.** Fashion-MNIST 분류에서 CNN-ReLU의 혼동 행렬.

![](/assets/images/posts/274/img_27.png)

**그림 9.** Fashion-MNIST 분류에서 CNN-Softmax의 혼동 행렬.

그림 8과 9는 두 모델이 Fashion-MNIST 분류에서 10개의 클래스에 대해 예측 성능을 보여줍니다. MNIST 분류 결과와는 달리, CNN-ReLU는 클래스별로 가장 많은 정확한 예측을 기록했습니다. 반대로, CNN-Softmax는 빠른 수렴 덕분에 클래스별로 더 높은 누적 정확한 예측을 기록했습니다.

**3.3. WDBC**  
우리는 표 2에 정의된 FFNN을 구현했으나, 두 개의 은닉층에 각각 512개의 뉴런 대신 64개의 뉴런과 32개의 뉴런을 사용했습니다. WDBC 분류의 경우, 우리는 데이터셋 특징만 정규화했습니다. WDBC는 30개의 특징만 있기 때문에 PCA 차원 축소는 효율적이지 않을 수 있습니다.  
64개의 뉴런과 32개의 뉴런을 가진 두 개의 은닉층이 있는 FFNN을 훈련하는 과정에서, 우리는 표 9에 설명된 결과를 발견했습니다.

**표 9. WDBC 분류.**  
CNN-Softmax 모델과 CNN-ReLU 모델의 정확도(%) 비교. 훈련 교차 검증은 10번의 분할에서 얻은 평균 교차 검증 정확도입니다. 테스트 정확도는 보지 않은 데이터에서의 성능을 나타냅니다. 정밀도(Precision), 재현율(Recall), F1-스코어는 보지 않은 데이터에서의 성능을 나타냅니다.

![](/assets/images/posts/274/img_28.png)

FFNN-ReLU는 WDBC 분류에서도 CNN 기반 모델을 사용한 분류 결과와 유사하게 FFNN-Softmax보다 성능이 낮았습니다. CNN 기반 모델과 일관되게, FFNN-ReLU는 FFNN-Softmax보다 수렴 속도가 느렸습니다. 그러나 두 모델 간의 F1-스코어 차이는 0.2에 불과했습니다. 이는 FFNN-ReLU가 여전히 FFNN-Softmax와 비교할 만하다는 것을 의미합니다.

![](/assets/images/posts/274/img_29.png)

**그림 10.** WDBC 분류에서 FFNN-ReLU의 혼동 행렬.

![](/assets/images/posts/274/img_30.png)

**그림 11.** WDBC 분류에서 FFNN-Softmax의 혼동 행렬.

그림 10과 11은 두 모델이 WDBC 분류에서 이진 분류에 대해 예측 성능을 보여줍니다. 혼동 행렬을 보면, FFNN-Softmax는 FFNN-ReLU보다 더 많은 \*\*거짓 음성(false negative)\*\*을 가지고 있었습니다. 반대로, FFNN-ReLU는 FFNN-Softmax보다 더 많은 \*\*거짓 양성(false positive)\*\*을 가지고 있었습니다.

**4. 결론 및 제언**  
DL-ReLU 모델에 대한 상대적으로 불리한 결과는 아마도 ReLU에서 발생하는 **죽은 뉴런 문제(dying neurons problem)** 때문일 것입니다. 즉, 뉴런을 통해 역방향으로 흐르는 그래디언트가 없어서 뉴런이 멈추고 결국 "죽게" 됩니다. 그 결과, 신경망의 학습 진행이 방해됩니다. 이 문제는 이후 ReLU의 개선된 버전들에서 해결되었습니다(예: Trottier et al., 2017). 이러한 단점에도 불구하고, DL-ReLU 모델이 여전히 기존의 Softmax 기반 DL 모델과 비교할 만하며, 경우에 따라서는 더 나을 수 있다는 점을 언급할 수 있습니다. 이는 MNIST와 Fashion-MNIST를 사용한 이미지 분류에서 DNN-ReLU 모델의 결과로 뒷받침됩니다.

향후 연구에서는 역전파 과정에서 그래디언트를 수치적으로 검토하여 DL-ReLU 모델을 철저히 조사할 수 있을 것입니다. 즉, DL-ReLU 모델의 그래디언트와 DL-Softmax 모델의 그래디언트를 비교할 수 있습니다. 또한 ReLU의 다양한 변형을 추가 비교 대상으로 고려할 수 있습니다.

**5. 감사의 글**  
이 연구에서 사용된 CNN 모델은 Keras(Chollet et al., 2015)의 VGG 유사 Convnet 소스 코드를 참고하여 사용되었음을 감사하게 생각합니다.

[1803.08375v2.pdf

0.99MB](./file/1803.08375v2.pdf)
