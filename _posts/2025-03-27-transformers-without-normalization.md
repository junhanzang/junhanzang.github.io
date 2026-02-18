---
title: "Transformers without Normalization"
date: 2025-03-27 23:50:23
categories:
  - 인공지능
tags:
  - transformers without normalization
---

<https://arxiv.org/abs/2503.10622?_bhlid=1a87c33b8185a942533ee1886e23e7f6c2d5f90d>

[Transformers without Normalization](https://arxiv.org/abs/2503.10622?_bhlid=1a87c33b8185a942533ee1886e23e7f6c2d5f90d)

정규화(Normalization) 층은 현대 신경망에서 폭넓게 사용되며 오랫동안 필수적인 요소로 여겨져 왔다. 본 연구는 정규화를 사용하지 않은 트랜스포머(Transformers)에서도 매우 간단한 기법을 통해 기존 모델과 동등하거나 더 우수한 성능을 얻을 수 있음을 보여준다. 이를 위해 본 논문은 동적 탄젠트 함수(Dynamic Tanh, DyT)를 도입하며, 이는 원소 단위(element-wise)로 적용되는 연산으로 정의된다:

![](/assets/images/posts/529/img.png)

DyT는 트랜스포머의 층 정규화(Layer normalization)가 종종 탄젠트 함수와 유사한 'S자 형태'의 입출력 매핑(input-output mapping)을 생성한다는 관찰에서 영감을 얻었다. DyT를 통해 정규화 층을 제거한 트랜스포머는 특별한 하이퍼파라미터 튜닝 없이도 기존 정규화를 적용한 모델과 동등하거나 더 뛰어난 성능을 달성할 수 있다. 본 논문은 이미지 인식에서 생성 모델까지, 지도 학습에서 자기 지도 학습까지, 그리고 컴퓨터 비전에서 자연어 처리 모델까지 다양한 분야에서 DyT를 적용한 트랜스포머의 효용성을 입증하였다. 이러한 결과는 정규화 층이 현대 신경망에서 필수불가결하다는 기존의 통념에 도전하며, 딥 네트워크에서 정규화 층이 갖는 역할에 관한 새로운 통찰을 제시한다.

<https://jiachenzhu.github.io/DyT/>

[Transformers without Normalization - DynamicTanh - DyT](https://jiachenzhu.github.io/DyT/)

## 1 서론

지난 10년 동안 정규화(normalization) 층은 현대 신경망에서 가장 근본적인 구성 요소 중 하나로 자리매김했다. 이러한 흐름은 2015년 배치 정규화(Batch Normalization, BN)의 등장으로 거슬러 올라갈 수 있다(Ioffe와 Szegedy, 2015). 배치 정규화는 시각 인식 모델의 학습 속도와 수렴 성능을 현저하게 향상시키며 빠르게 인기를 얻었다. 이후로 다양한 네트워크 구조나 응용 분야에 맞춰 정규화 층의 변형들이 다수 제안되었다(Ba 외, 2016; Ulyanov 외, 2016; Wu와 He, 2018; Zhang과 Sennrich, 2019). 현재 거의 모든 현대적 신경망들이 정규화 층을 사용하고 있으며, 특히 트랜스포머(Transformer) 구조에서는 층 정규화(Layer Normalization, LN)가 가장 널리 쓰이고 있다(Vaswani 외, 2017; Dosovitskiy 외, 2020).

정규화 층이 폭넓게 사용되는 주된 이유는 최적화 측면에서의 실질적인 이점 때문이다(Santurkar 외, 2018; Bjorck 외, 2018). 정규화 층은 모델의 성능을 높일 뿐만 아니라, 수렴 속도를 가속화하고 안정화하는 데 도움을 준다. 신경망이 점점 더 넓고 깊어짐에 따라 이러한 특성은 더욱 중요해지고 있다(Brock 외, 2021a; Huang 외, 2023). 그 결과 정규화 층은 딥 네트워크 학습의 효과적인 훈련을 위한 필수적인 요소로 간주된다. 이는 최근의 신경망 설계에서 어텐션(attention)이나 컨볼루션(convolution) 층은 다른 방식으로 대체하려고 하면서도(Tolstikhin 외, 2021; Gu와 Dao, 2023; Sun 외, 2024; Feng 외, 2024), 정작 정규화 층만큼은 거의 항상 유지하고 있는 사실을 통해서도 잘 드러난다.

본 논문은 이와 같은 통념에 도전하며, 트랜스포머 모델 내의 정규화 층을 대체할 수 있는 간단한 대안을 제시한다. 본 연구는 LN 층이 입력을 출력으로 변환할 때 종종 탄젠트(tanh) 함수와 유사한 S자 형태의 곡선으로 매핑하며, 입력 활성값을 스케일링하고 극단적인 값들을 억제한다는 관찰에서 시작되었다. 이러한 통찰에 영감을 얻어, 본 논문은 다음과 같은 원소 단위(element-wise) 연산인 **Dynamic Tanh (DyT)**를 제안한다:

![](/assets/images/posts/529/img_1.png)

여기서 α는 학습 가능한 매개변수이다. DyT는 α를 통해 적절한 스케일링을 학습하고, 경계가 있는 tanh 함수를 사용하여 극단적인 값들을 억제함으로써 LN 층과 유사한 효과를 낸다. 특히 정규화 층과 달리 활성값의 통계량을 계산할 필요 없이 이러한 효과를 달성할 수 있다.

DyT를 사용하는 방법은 매우 간단하다(그림 1 참조). 우리는 비전 및 언어 분야의 트랜스포머를 포함한 기존 신경망 구조에서 정규화 층을 DyT로 바로 교체하여 적용한다. 다양한 환경에서 수행한 실험 결과, DyT를 적용한 모델은 안정적으로 학습되었으며 우수한 최종 성능을 보였다. 심지어 기존 모델의 하이퍼파라미터 튜닝을 별도로 수행하지 않은 상태에서도 좋은 성과를 얻었다. 본 연구의 결과는 현대적 신경망 학습에서 정규화 층이 필수불가결하다는 기존의 통념을 흔들며, 정규화 층의 특성에 대한 경험적 통찰을 제공한다. 또한 예비 실험 결과, DyT가 학습 및 추론 속도까지 향상시키는 것으로 나타나, 효율성을 추구하는 네트워크 설계에서도 잠재적인 후보로 자리잡을 가능성을 보였다.

![](/assets/images/posts/529/img_2.png)

**그림 1:** (왼쪽) 원본 트랜스포머 블록. (오른쪽) 본 연구에서 제안하는 DyT 층이 적용된 트랜스포머 블록. DyT는 일반적으로 널리 사용되는 Layer Norm (Ba 외, 2016) (경우에 따라 RMSNorm (Zhang과 Sennrich, 2019)) 층을 간단히 대체한다. DyT를 적용한 트랜스포머는 기존 정규화된 모델과 동등하거나 더 뛰어난 성능을 보인다.

## 2 배경: 정규화 층(Normalization Layers)

먼저, 정규화 층(normalization layers)에 대해 간략히 살펴본다. 대부분의 정규화 층은 공통된 수식을 공유한다. 입력 데이터 x의 형태(shape)가 (B,T,C)로 주어졌다고 하자. 여기서 B는 배치 크기(batch size), T는 토큰(token)의 개수, C는 각 토큰의 임베딩 차원(embedding dimension)을 나타낸다. 이때 정규화 층의 출력은 일반적으로 다음과 같은 형태로 계산된다.

![](/assets/images/posts/529/img_3.png)

여기서 ϵ은 매우 작은 상수(small constant)이며, γ와 β는 학습 가능한(learnable) 매개변수로서 형태가 (C,)인 벡터이다. γ는 스케일링(scaling), β는 시프팅(shifting)을 담당하는 아핀(affine) 매개변수로서, 이를 통해 출력값이 임의의 범위를 가지도록 조정된다. 여기서 μ와 σ^2는 입력의 평균(mean)과 분산(variance)을 의미하며, 각 정규화 방법에 따라 계산 방식이 달라진다. 따라서 이 통계량들의 차원 또한 각 방법마다 다르며, 실제 계산에서는 브로드캐스팅(broadcasting)을 통해 차원이 맞춰진다.

배치 정규화(Batch Normalization, BN)(Ioffe와 Szegedy, 2015)는 최초로 등장한 현대적 정규화 기법이며, 주로 컨볼루션 신경망(ConvNets) 모델에서 활용되었다(Szegedy 외, 2016; He 외, 2016; Xie 외, 2017). BN의 등장은 딥러닝 모델 설계에 있어서 중요한 이정표로 평가된다. BN은 배치(batch)와 토큰(token) 차원에 걸쳐 평균과 분산을 계산한다. 구체적으로 다음과 같다:

![](/assets/images/posts/529/img_4.png)

컨볼루션 신경망에서 인기가 높은 다른 정규화 기법으로는 그룹 정규화(Group Normalization)(Wu와 He, 2018)와 인스턴스 정규화(Instance Normalization)(Ulyanov 외, 2016) 등이 있으며, 이는 원래 물체 탐지(object detection)나 이미지 스타일 변환(image stylization)과 같은 특수한 작업을 위해 제안되었다. 이들 역시 기본적인 수식 형태는 동일하지만, 평균과 분산을 계산하는 축(axes)과 범위(ranges)가 서로 다르다.

트랜스포머(Transformer) 아키텍처에서 가장 널리 쓰이는 두 가지 정규화 기법은 층 정규화(Layer Normalization, LN)(Ba 외, 2016)와 루트 평균 제곱 정규화(Root Mean Square Normalization, RMSNorm)(Zhang과 Sennrich, 2019)이다. LN은 각 샘플(sample)의 각 토큰(token)에 대해 독립적으로 통계량을 계산한다. 계산 방식은 다음과 같다:

![](/assets/images/posts/529/img_5.png)

RMSNorm(Zhang과 Sennrich, 2019)는 LN을 단순화한 것으로, 입력값에서 평균을 빼주는(mean-centering) 단계를 제거하고, 대신 입력을 다음과 같이 정규화한다:

![](/assets/images/posts/529/img_6.png)

현재 대부분의 현대적 신경망은 단순성과 범용성 때문에 LN을 사용하고 있다. 최근 들어 RMSNorm 또한 널리 사용되고 있으며, 특히 T5(Raffel 외, 2020), LLaMA(Touvron 외, 2023a, b; Dubey 외, 2024), Mistral(Jiang 외, 2023), Qwen(Bai 외, 2023; Yang 외, 2024), InternLM(Zhang 외, 2024; Cai 외, 2024), DeepSeek(Liu 외, 2024; Guo 외, 2025)와 같은 대형 언어 모델에서 자주 쓰이고 있다. 본 논문에서 실험하는 대부분의 트랜스포머 모델은 LN을 사용하며, 예외적으로 LLaMA 모델만이 RMSNorm을 사용한다.

## 3 정규화 층은 어떤 역할을 하는가?

### 분석 환경

먼저 학습된 신경망 내에서 정규화 층(normalization layers)의 동작을 실험적으로 분석하였다. 분석을 위해 다음의 세 가지 모델을 사용하였다.

- **Vision Transformer (ViT-B)** (Dosovitskiy 외, 2020): ImageNet-1K 데이터셋(Deng 외, 2009)으로 학습된 모델.
- **wav2vec 2.0 Large Transformer** (Baevski 외, 2020): LibriSpeech 데이터셋(Panayotov 외, 2015)으로 학습된 음성 모델.
- **Diffusion Transformer (DiT-XL)** (Peebles와 Xie, 2023): ImageNet-1K에서 학습된 생성 모델.

모든 모델은 각 Transformer 블록 내부와 최종 선형(linear) 프로젝션 전에 LN(Layer Normalization)을 적용하였다.

세 모델 각각에서 미니 배치(mini-batch)를 샘플링하고 네트워크의 순방향 전달(forward pass)을 수행하였다. 그리고 각 정규화 층의 입력과 출력(즉, 학습 가능한 아핀 변환(affine transformation) 이전의 값)을 측정하였다. LN은 입력 텐서의 차원을 유지하므로 입력과 출력 요소 간에 일대일 대응을 설정하여 직접적인 관계를 시각화할 수 있다. 이를 통해 얻어진 매핑을 그림 2에 나타내었다.

![](/assets/images/posts/529/img_7.png)

![](/assets/images/posts/529/img_8.png)

![](/assets/images/posts/529/img_9.png)

**그림 2**: ViT(Dosovitskiy 외, 2020), wav2vec 2.0(Baevski 외, 2020), DiT(Peebles와 Xie, 2023) 모델의 특정 LN층에 대한 입력 대비 출력 값을 나타낸 그래프. 각 모델에서 네 개의 LN층을 선택하여 미니 배치의 입력-출력 값을 시각화하였다. 출력 값은 LN에서 아핀 변환 전의 값이다. 그림에서 나타난 S자 형태 곡선은 tanh 함수와 매우 유사하다(그림 3 참조). 초반 층에서 나타나는 선형 형태 역시 tanh 곡선의 중심부로 표현 가능하다. 이 결과를 바탕으로, 서로 다른 축적을 고려하기 위해 학습 가능한 매개변수 α를 가진 **Dynamic Tanh(DyT)**를 제안하였다.

![](/assets/images/posts/529/img_10.png)

**그림 3**: 서로 다른 세 가지 값의 α에 따른 tanh⁡(αx) 함수의 형태.

### 층 정규화에서 나타난 Tanh 형태의 매핑

그림 2의 모든 모델에서 초반 LN층(첫 번째 열)에서는 입력과 출력 간 관계가 대부분 선형적(linear)으로 나타나, x-y 플롯에서 직선에 가까운 모습을 보였다. 그러나 보다 심층부의 LN층에서는 더 흥미로운 현상이 나타났다.

깊은 층에서 특히 주목할 점은, 대부분의 곡선 형태가 완전하거나 부분적인 tanh 함수 형태(S자 형태)와 매우 유사하다는 것이다(그림 3 참조). LN층이 입력 텐서를 선형적으로 변환할 것으로 예상할 수도 있는데, 이는 평균을 빼고 표준편차로 나누는 연산 자체가 본질적으로 선형적이기 때문이다. LN은 각 토큰 단위로 정규화를 수행하며 각 토큰 활성값을 개별적으로만 선형 변환한다. 그러나 실제로는 각 토큰의 평균과 표준편차가 서로 다르기 때문에 전체 텐서의 활성값을 종합적으로 볼 때 선형성은 유지되지 않는다. 그런데도 불구하고, 실제 비선형 변환이 스케일링된 tanh 함수와 매우 유사하다는 것은 놀라운 결과이다.

이러한 S자 형태의 곡선은 중심부(x 값이 0 근처)에선 여전히 선형 형태를 보이며, 대부분의 점(약 99%)은 이 선형 범위에 존재한다. 하지만 ViT 모델에서 x가 50보다 크거나 -50보다 작은 값과 같은 극단적인 범위에 놓이는 점들이 상당수 존재한다. 정규화 층의 주된 역할은 이러한 극단적인 값을 다수의 점들과 유사한 덜 극단적인 값으로 압축(squash)하는 것이다. 이러한 압축 효과는 단순한 아핀 변환으로는 근사될 수 없으며, 우리는 이와 같은 비선형적이며 불균형적인 압축 효과가 정규화 층을 중요하고 필수불가결하게 만드는 이유일 것으로 가설을 세운다.

Ni 외(2024)의 최근 연구 또한 LN층이 가져오는 강력한 비선형성의 효과를 강조하며, 이러한 비선형성이 모델의 표현 능력(representational capacity)을 증대시킨다는 사실을 지적하였다. 또한 이런 압축 현상은 생물학적 뉴런이 큰 입력값에 대해 포화(saturation)되는 현상과 유사하며, 이는 약 100년 전 처음 관찰된 현상이기도 하다(Adrian, 1926; Adrian와 Zotterman, 1926a, b).

### 토큰 단위와 채널 단위로 살펴본 정규화

LN 층이 각 토큰에 대해 선형 변환을 수행하면서도 극단적 값은 어떻게 비선형적으로 압축하는지 이해하기 위해, 데이터를 토큰과 채널 별로 그룹화하여 시각화하였다. 이를 그림 2에서 ViT 모델의 두 번째와 세 번째 그래프를 다시 그리며 명확한 관찰을 위해 일부 점만 선택하여 그림 4에 나타내었다. 시각화 시, 극단적인 값을 가진 채널도 포함되도록 선정하였다.

![](/assets/images/posts/529/img_11.png)

**그림 4**: 두 LN 층에서 입력 대비 출력 값을 나타낸 그래프. 텐서 요소를 채널과 토큰 차원별로 다른 색깔로 표현하였다. 입력 텐서의 형태는 (샘플, 토큰, 채널)이며, 동일 토큰(왼쪽 두 패널)과 동일 채널(오른쪽 두 패널)마다 일정한 색상을 사용하여 시각화하였다.

- **왼쪽 두 패널:** 같은 토큰(같은 색)의 점들은 채널별로 직선을 형성한다. 이는 LN이 각 토큰의 채널 방향으로 선형 연산을 수행하기 때문이다. 그러나 모든 토큰을 함께 나타내면 이 직선들은 tanh 형태의 비선형 곡선을 형성한다.
- **오른쪽 두 패널:** 각 채널은 입력이 다른 범위에 분포하며, tanh 형태의 곡선에 서로 다른 부분을 기여한다. 일부 채널(예: 빨강, 초록, 핑크)은 극단적으로 큰 값을 가지며, 이러한 값들이 LN에 의해 강하게 압축된다.

그림 4의 왼쪽 두 패널에서 같은 색으로 나타낸 각 토큰의 활성값은 실제로 직선을 형성한다. 그러나 각 토큰의 분산이 다르기 때문에 직선의 기울기는 서로 다르다. 입력 범위가 작은 토큰들은 표준편차가 작으므로 LN은 이들의 활성값을 작은 값으로 나누게 되어 기울기가 커진다. 이렇게 모인 직선들이 전체적으로 tanh 형태의 S자 곡선을 형성한다. 오른쪽 두 패널에서는 각 채널마다 활성값의 입력 범위가 매우 다르며, 일부 소수의 채널만이 큰 극단값을 가지는 것이 확인된다. LN 층은 이 채널들의 극단적인 값을 가장 많이 압축한다.

## 4 Dynamic Tanh (DyT)

정규화 층의 입출력 형태가 스케일링된(scaled) tanh 함수와 유사하다는 점에서 영감을 얻어, 본 연구는 정규화 층의 대체로써 **Dynamic Tanh (DyT)**를 제안한다. 입력 텐서 x가 주어졌을 때 DyT 층은 다음과 같이 정의된다:

![](/assets/images/posts/529/img_12.png)

여기서 α는 입력값의 범위(scale)에 따라 입력을 동적으로 스케일링할 수 있도록 학습되는 스칼라 매개변수로, 그림 2에서 나타난 다양한 입력 범위를 처리하기 위해 도입된 것이다. 바로 이 때문에 본 연산을 "동적(Dynamic)" Tanh라 부르게 되었다. 매개변수 γ와 β는 기존의 모든 정규화 층에서 사용되는 것과 동일하게 채널 단위로 학습되는 벡터 매개변수이며, DyT의 출력을 다시 임의의 범위로 조정하는 역할을 한다. 이러한 매개변수는 종종 별도의 아핀(affine) 층으로 간주되기도 하지만, 본 논문에서는 이를 정규화 층에서처럼 DyT 층의 일부로 간주한다. DyT 층의 구현은 PyTorch 스타일의 의사 코드(pseudocode)로 알고리즘 1에 제시하였다.

```
# 입력 x의 형태는 [B, T, C] 
# B: 배치 크기, T: 토큰 수, C: 차원 수
class DyT(Module):
    def __init__(self, C, init_alpha):
        super().__init__()
        self.alpha = Parameter(ones(1) * init_alpha)
        self.gamma = Parameter(ones(C))
        self.beta = Parameter(zeros(C))

    def forward(self, x):
        x = tanh(self.alpha * x)
        return self.gamma * x + self.beta
```

**알고리즘 1:** DyT 층의 의사 코드(pseudocode).

DyT 층을 기존 아키텍처에 통합하는 방법은 매우 간단하며, 기존의 정규화 층을 DyT 층 하나로 교체하는 방식으로 이루어진다(그림 1 참조). 이 방법은 어텐션(attention) 블록 내의 정규화 층, FFN 블록, 그리고 최종 정규화 층 모두에 적용된다. DyT는 활성화 함수(activation function)처럼 보일 수도 있지만, 본 연구에서는 GELU나 ReLU와 같은 기존 아키텍처의 활성화 함수는 전혀 변경하지 않고 오직 정규화 층만 DyT로 대체하였다. 신경망의 다른 구성 요소 역시 변경하지 않았다. 또한, 기존 모델에서 사용한 하이퍼파라미터를 DyT에 맞추기 위해 튜닝할 필요가 거의 없다는 점을 관찰하였다.

### 스케일링 매개변수(Scaling Parameters)에 관하여

DyT의 매개변수 γ는 모든 값이 1인 벡터, β는 모든 값이 0인 벡터로, 기존 정규화 층의 초기화 방법과 동일하게 설정하였다. 스케일링을 위한 매개변수 α는 일반적으로 초기값 0.5로 설정하는 것이 적절하며, 예외적으로 대형 언어 모델(LLM)을 학습할 때만 다를 수 있다. α의 초기화에 관한 상세한 분석은 7장에서 제시된다. 이후 본 논문의 실험에서는 특별한 언급이 없는 한 α 값을 0.5로 초기화하였다.

### 추가 설명(Remarks)

DyT는 엄밀히 말해 새로운 유형의 정규화 층이 아니다. DyT는 정규화 층과 달리 통계량이나 그 어떤 형태의 집계(aggregation)도 계산하지 않으며, 순방향 전달(forward pass) 시 각 입력 요소(element)를 독립적으로 처리한다. 그럼에도 불구하고 DyT는 정규화 층의 효과를 보존한다. 즉, 극단적인 값은 비선형적으로 압축하고, 입력의 중심부(대부분의 값)는 거의 선형적으로 변환하는 특성을 유지한다.

## 5 실험 (Experiments)

본 절에서는 DyT의 효용성을 입증하기 위해 트랜스포머 모델을 비롯한 여러 최신 아키텍처를 다양한 작업과 도메인에 걸쳐 실험하였다. 각 실험에서는 원본 아키텍처에서 사용된 LN이나 RMSNorm 층을 DyT 층으로 교체하고, 공식 오픈 소스 프로토콜에 따라 두 버전의 모델을 학습 및 평가하였다. 결과 재현을 위한 상세한 지침은 부록 A에 제공하였다. 특히 DyT 적용의 간단함을 강조하기 위해 하이퍼파라미터는 원본 모델과 동일하게 유지하였다. 학습률(learning rate)과 α의 초기값 튜닝과 관련된 추가 실험 결과는 부록 B에 제시하였다.

### 비전 도메인의 지도학습 (Supervised learning in vision)

ImageNet-1K(Deng 외, 2009) 분류 작업에서 Vision Transformer(ViT)(Dosovitskiy 외, 2020)와 ConvNeXt(Liu 외, 2022) 모델을 "Base" 및 "Large" 크기로 학습하였다. 두 모델은 각각 attention(ViT)과 convolution(ConvNeXt)이라는 서로 다른 연산 방식을 사용하며 널리 쓰이는 모델이다. 실험 결과는 표 1에 나타냈다. DyT는 모든 아키텍처와 모델 크기에서 LN과 비슷하거나 약간 더 나은 성능을 보였다. 추가로 ViT-B와 ConvNeXt-B 모델의 훈련 손실(training loss) 그래프를 그림 5에 나타냈다. 두 그래프는 DyT와 LN 모델이 매우 유사한 수렴 경향을 보임을 나타낸다.

**표 1:** ImageNet-1K에서의 지도학습 분류 정확도 (Top-1 accuracy).

![](/assets/images/posts/529/img_13.png)

![](/assets/images/posts/529/img_14.png)

![](/assets/images/posts/529/img_15.png)

**그림 5:** ViT-B와 ConvNeXt-B 모델의 훈련 손실 곡선. 두 모델에서 LN과 DyT의 손실 곡선은 유사한 양상을 보이며, 학습 역학(dynamics)이 유사할 수 있음을 시사한다.

### 비전 도메인의 자기 지도 학습 (Self-supervised learning in vision)

비전 도메인의 대표적인 자기 지도 학습 방법인 Masked Autoencoder(MAE, He 외, 2022)와 DINO(Caron 외, 2021)를 사용하여 벤치마크하였다. 두 방법 모두 백본(backbone)으로 ViT를 사용하지만, MAE는 재구성(reconstruction) 손실을, DINO는 공동 임베딩(joint-embedding) 손실을 사용한다. 표준적인 자기 지도 학습 프로토콜에 따라 ImageNet-1K에서 라벨 없이 사전 학습(pretrain)한 후, 분류층을 붙여 라벨을 이용하여 미세조정(fine-tune)하였다. 미세조정 결과는 표 2에 나타냈다. DyT는 자기 지도 학습에서도 LN과 동등한 성능을 보였다.

**표 2:** ImageNet-1K 자기 지도 학습 정확도.

![](/assets/images/posts/529/img_16.png)

### 확산 모델 (Diffusion models)

Diffusion Transformer(DiT, Peebles와 Xie, 2023)를 ImageNet-1K에서 크기(B, L, XL)에 따라 학습하였다. DiT에서는 LN의 아핀 파라미터가 클래스 조건부 처리를 위해 사용되므로, DyT 실험에서도 이를 유지한 채 정규화 연산만 tanh⁡(αx) 함수로 대체하였다. 평가 지표는 표준 ImageNet 기준 배치를 이용한 Fréchet Inception Distance(FID)이며, 결과는 표 3에 나타냈다. DyT는 LN 대비 유사하거나 개선된 FID를 보였다.

**표 3:** ImageNet 이미지 생성 품질 평가(FID, 낮을수록 좋음).

![](/assets/images/posts/529/img_17.png)

### 대형 언어 모델 (Large Language Models)

LLaMA 모델(7B, 13B, 34B, 70B, Touvron 외, 2023a,b; Dubey 외, 2024)을 사전 학습하여 RMSNorm(Zhang과 Sennrich, 2019)와 DyT를 비교하였다. 모델은 The Pile 데이터셋에서 200B 토큰을 사용하여 학습하였다. DyT를 적용한 LLaMA는 초반 임베딩 층 뒤에 학습 가능한 스칼라 파라미터를 추가하고, α의 초기값을 조정하였다(7장 참고). 사전 학습 후 손실 값과 lm-eval(Gao 외)의 15개 zero-shot 평가 작업 평균 성능을 표 4에 제시하였다. DyT는 모든 크기에서 RMSNorm과 비슷한 성능을 달성했다. 손실 곡선은 그림 6에 나타냈으며, 모든 크기에서 유사한 학습 경향을 보였다.

**표 4:** LLaMA 모델의 사전 학습 손실 및 15개 zero-shot 작업 성능.

![](/assets/images/posts/529/img_18.png)

![](/assets/images/posts/529/img_19.png)

![](/assets/images/posts/529/img_20.png)

![](/assets/images/posts/529/img_21.png)

![](/assets/images/posts/529/img_22.png)

**그림 6**: LLaMA 사전 학습 손실. DyT 및 RMSNorm 모델의 손실 곡선은 모델 크기 전반에 걸쳐 밀접하게 정렬되어 있습니다.

### 음성 도메인 자기 지도 학습 (Self-supervised learning in speech)

wav2vec 2.0 Transformer 모델을 LibriSpeech 데이터셋에서 사전 학습하였다. 표 5에 나타낸 결과 DyT는 LN과 동등한 성능을 보였다.

**표 5:** LibriSpeech 음성 사전 학습 검증 손실(validation loss).

![](/assets/images/posts/529/img_23.png)

### DNA 서열 모델링 (DNA sequence modeling)

장거리 DNA 서열 모델링에서 HyenaDNA(Nguyen 외, 2024), Caduceus(Schiff 외, 2024)를 사용하였다. DyT는 표 6에서 LN과 동등한 성능을 보였다.

**표 6:** DNA 분류 정확도(GenomicBenchmarks 평균).

![](/assets/images/posts/529/img_24.png)

---

이 논문에서 **RMSNorm과 DyT의 비교**가 등장한 이유는 논문이 주로 **Transformer 계열의 대형 언어 모델(LLM)** 에서 성능을 평가했기 때문입니다.

설명하신 것처럼 Layer Normalization(LN)은 Transformer 구조에서 가장 흔히 사용되는 정규화 층인데, 최근 몇몇 인기 있는 대형 언어 모델들—예를 들어 **LLaMA** 시리즈—은 RMSNorm(Root Mean Square Normalization)을 기본 정규화로 채택하고 있습니다. 따라서 이 논문은 LLaMA 모델과 같은 최신 대형 언어 모델에서도 DyT가 RMSNorm만큼 우수하게 작동하는지를 확인하기 위해 RMSNorm과 비교 실험을 진행한 것입니다.

---
