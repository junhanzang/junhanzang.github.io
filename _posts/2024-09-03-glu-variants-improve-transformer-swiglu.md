---
title: "GLU Variants Improve Transformer (SwiGLU)"
date: 2024-09-03 20:55:54
categories:
  - 인공지능
---

<https://arxiv.org/abs/2002.05202>

[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

요약

Gated Linear Units (GLU) [Dauphin et al., 2016]는 두 개의 선형 투영(Linear Projection)을 성분별로 곱하여 구성되며, 이 중 하나는 먼저 시그모이드 함수(Sigmoid Function)를 통과합니다. GLU의 변형들은 시그모이드 대신에 다양한 비선형 함수(혹은 선형 함수)를 사용하는 것이 가능합니다. 우리는 Transformer [Vaswani et al., 2017] 시퀀스-투-시퀀스 모델의 피드포워드 하위 계층에서 이러한 변형들을 테스트했고, 이 중 일부가 일반적으로 사용되는 ReLU나 GELU 활성화 함수에 비해 품질 개선을 가져오는 것을 발견했습니다.

## 1. 서론

Transformer [Vaswani et al., 2017] 시퀀스-투-시퀀스 모델은 다중 헤드 어텐션과 "위치별 피드포워드 네트워크" (FFN)라고 불리는 구조를 번갈아 가며 사용합니다. FFN은 시퀀스 내 특정 위치에서의 숨겨진 표현 벡터인 x를 입력으로 받아, 두 개의 학습된 선형 변환(행렬 W1과 W2, 그리고 편향 벡터 b1과 b2로 표현됨)을 통과시킵니다. 이 과정에서 두 선형 변환 사이에는 Rectified Linear Unit (ReLU) [Glorot et al., 2011] 활성화 함수가 적용됩니다.

FFN(x,W1,W2,b1,b2)=max(0,xW1+b1)W2+b2 (1)

T5 코드베이스 [Raffel et al., 2019]를 따라, 우리는 편향이 없는 버전을 사용합니다:

FFNReLU(x,W1,W2)=max(xW1,0)W2 (2)

이후의 연구에서는 ReLU를 Gaussian Error Linear Units (GELU), GELU(x)=xΦ(x) [Hendrycks and Gimpel, 2016], 및 Swish, Swish(x)=xσ(βx) [Ramachandran et al., 2017]와 같은 다른 비선형 활성화 함수로 대체할 것을 제안했습니다.

FFNGELU​(x,W1,W2)=GELU(xW1)W2

FFNSwish​(x,W1,W2)=Swish(xW1)W2 (3)

## 2. Gated Linear Units (GLU) 및 변형

[Dauphin et al., 2016]은 Gated Linear Units (GLU)를 소개했는데, 이는 입력의 두 개의 선형 변환을 성분별로 곱하여 하나는 시그모이드로 활성화한 신경망 층입니다. 이들은 또한 활성화를 생략한 "이선형" (bilinear) 층을 제안했으며, 이는 [Mnih and Hinton, 2007]에 기인한다고 언급했습니다.

GLU(x,W,V,b,c)=σ(xW+b)⋅(xV+c)

Bilinear(x,W,V,b,c)=(xW+b)⋅(xV+c)

우리는 다른 활성화 함수를 사용하는 GLU 변형도 정의할 수 있습니다:

ReGLU(x,W,V,b,c)=max(0,xW+b)⋅(xV+c)

GEGLU(x,W,V,b,c)=GELU(xW+b)⋅(xV+c)

SwiGLU(x,W,V,b,c,β)=Swishβ​(xW+b)⋅(xV+c)(5)

이 논문에서는, Transformer FFN 층에 대해 첫 번째 선형 변환과 활성화 함수 대신 GLU 또는 그 변형을 사용하는 추가 변형을 제안합니다. 이때, 편향 항은 생략합니다.

FFNGLU​(x,W,V,W2)=(σ(xW)⋅xV)W2

FFNBilinear​(x,W,V,W2)=(xW⋅xV)W2

FFNReGLU​(x,W,V,W2)=(max(0,xW)⋅xV)W2

FFNGEGLU​(x,W,V,W2)=(GELU(xW)⋅xV)W2

FFNSwiGLU​(x,W,V,W2)=(Swish1​(xW)⋅xV)W2(6)

이 모든 층은 원래의 FFN에 비해 두 개의 행렬이 아닌 세 개의 가중치 행렬을 가지고 있습니다. 따라서, 매개변수의 수와 계산량을 일정하게 유지하기 위해, 이 층들을 원래의 두 행렬 버전과 비교할 때, 숨겨진 유닛의 수 d\_ff​ (W와 V의 두 번째 차원과 W2의 첫 번째 차원)를 2/3으로 줄입니다.

![](/assets/images/posts/271/img.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

![](/assets/images/posts/271/img_1.png)

![](/assets/images/posts/271/img_2.png)

![](/assets/images/posts/271/img_3.png)

![](/assets/images/posts/271/img_4.png)

![](/assets/images/posts/271/img_5.png)

![](/assets/images/posts/271/img_6.png)

![](/assets/images/posts/271/img_7.png)

![](/assets/images/posts/271/img_8.png)

![](/assets/images/posts/271/img_9.png)

![](/assets/images/posts/271/img_10.png)

![](/assets/images/posts/271/img_11.png)

![](/assets/images/posts/271/img_12.png)

![](/assets/images/posts/271/img_13.png)

![](/assets/images/posts/271/img_14.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 3. Text-to-Text Transfer Transformer (T5) 실험

우리는 [Raffel et al., 2019]의 전이 학습 설정에서 설명한 FFN(Feed-Forward Network) 변형들을 테스트했습니다. 인코더-디코더 Transformer 모델 [Vaswani et al., 2017]을 사용하여 텍스트에서 누락된 부분을 예측하는 비지도 학습 목표로 훈련한 후, 다양한 언어 이해 작업에 대해 미세 조정(fine-tuning)을 진행했습니다.

![](/assets/images/posts/271/img_15.png)

표 1: [Raffel et al., 2019]의 세그먼트 채우기 작업에서 Transformer 모델의 검증 세트 로그-퍼플렉서티(log-perplexity). 모든 모델은 매개변수와 연산량이 동일하게 맞춰졌습니다.

![](/assets/images/posts/271/img_16.png)

## 3.2 사전 학습과 퍼플렉서티 결과

[Raffel et al., 2019]와 동일하게, 우리는 C4 데이터셋에서 스팬 채우기(span-filling) 목표로 524,288단계를 사전 학습했습니다. 각 학습 배치는 128개의 예시로 구성되며, 각 예시는 512개의 토큰 입력과 114개의 토큰 출력을 포함합니다. 출력에는 입력에서 삭제된 여러 스팬의 토큰이 포함됩니다. [Raffel et al., 2019]와 유사하게, 우리는 Adafactor 옵티마이저 [Shazeer and Stern, 2018]와 역제곱근(inverse-square-root) 학습률 스케줄을 사용했습니다. 또한, 훈련 단계의 마지막 10% 동안 학습률을 선형적으로 감소시켰습니다. [Raffel et al., 2019]과의 주요 차이점은 사전 학습 동안 드롭아웃을 사용하지 않았다는 점입니다. 우리는 이것이 더 우수한 결과를 만들어낸다고 판단했습니다. 우리는 C4의 홀드아웃된 샤드에서 학습 목표에 대한 로그-퍼플렉서티를 계산했으며, 이를 모델 품질의 좋은 지표로 간주했습니다. 각 모델 아키텍처에 대해, 우리는 짧은 기간(65,536단계) 동안 4개의 모델을 추가로 학습하여 실행 간 변동성을 측정했습니다. 결과는 표 1에 나와 있습니다. GEGLU와 SwiGLU 변형이 가장 좋은 퍼플렉서티 결과를 보여주었습니다.

## 3.3 미세 조정 (Fine-Tuning)

그 후, 우리는 완전히 학습된 각 모델을 Stanford Question-Answering Dataset (SQuAD) [Rajpurkar et al., 2016]과 GLUE [Wang et al., 2018], SuperGlue [Wang et al., 2019] 벤치마크의 모든 언어 이해 작업에서 예시 비율에 따라 혼합된 데이터셋을 사용해 한 번씩 미세 조정(fine-tuning)했습니다. 미세 조정은 학습률 10^-3로 131,072단계에 걸쳐 이루어집니다. 학습과 마찬가지로, 각 단계에서의 입력 시퀀스는 총 길이가 약 65,536 토큰에 달합니다. [Raffel et al., 2019]를 따르며, 우리는 층 출력, 피드포워드 숨겨진 층, 그리고 어텐션 가중치에 대해 0.1의 드롭아웃 비율을 사용했습니다. 임베딩 행렬은 미세 조정 동안 고정됩니다.

표 2, 3, 4에는 개발 세트에 대한 결과가 나와 있습니다. 각 작업에 대해, 미세 조정 중 기록된 체크포인트 중에서 가장 높은 점수를 보고했습니다. 결과는 다소 불규칙하지만, 새로운 GLU 변형이 대부분의 작업에서 가장 우수한 성능을 보였습니다. 비교를 위해, 각 표의 하단에는 [Raffel et al., 2019]의 결과를 나열했습니다. 해당 모델은 우리의 FFNReLU 모델과 동일합니다. 그들의 결과는 상당히 낮았는데, 이는 사전 학습 동안 드롭아웃을 사용했기 때문이라고 생각합니다. 또한, [Raffel et al., 2019]에서 측정된 실행 간 표준 편차도 나열되어 있습니다.

표 2: GLUE 언어 이해 벤치마크 [Wang et al., 2018] (개발 세트).

![](/assets/images/posts/271/img_17.png)

표 3: SuperGLUE 언어 이해 벤치마크 [Wang et al., 2019] (개발 세트).

![](/assets/images/posts/271/img_18.png)

표 4: SQuAD [Rajpurkar et al., 2016] v1.1 (개발 세트).

![](/assets/images/posts/271/img_19.png)

## 4. 결론

우리는 GLU 계열의 층을 확장하여 Transformer 모델에 그들의 사용을 제안했습니다. 전이 학습 설정에서, 새로운 변형들은 사전 학습에 사용된 디노이징 목표에 대해 더 나은 퍼플렉서티(perplexity)를 제공했으며, 많은 후속 언어 이해 작업에서도 더 나은 결과를 보이는 것처럼 보입니다. 이 아키텍처들은 구현이 간단하며, 명백한 계산적 단점이 없습니다. 이러한 아키텍처들이 왜 효과적인지에 대해서는 별다른 설명을 제공하지 않으며, 그 성공을 신의 은총 덕분이라고 생각합니다.

[2002.05202v1.pdf

0.10MB](./file/2002.05202v1.pdf)
