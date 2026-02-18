---
title: "Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models"
date: 2025-08-26 17:35:06
categories:
  - 인공지능
tags:
  - PolyNorm
---

<https://arxiv.org/abs/2411.03884>

[Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models](https://arxiv.org/abs/2411.03884)

**초록(Abstract)**  
Transformer는 강력한 적합(fitting) 능력 덕분에 다양한 분야에서 폭넓게 활용되고 있다. 이러한 성공은 부분적으로 Transformer가 지니고 있는 비선형성(inherent nonlinearity)에 기인한다. 따라서 원래 Transformer 아키텍처에서 사용된 ReLU 함수뿐만 아니라, 연구자들은 GeLU, SwishGLU와 같은 대체 모듈들을 탐구하여 비선형성을 강화하고 표현 능력(representational capacity)을 확장하려 하였다.

본 논문에서는 Transformer의 동역학(dynamics)을 최적화하기 위해 고안된 새로운 범주의 다항식 조합 활성화 함수(polynomial composition activations, **PolyCom**)를 제안한다. 이론적으로 우리는 PolyCom에 대한 포괄적인 수학적 분석을 제공하며, 다른 활성화 함수와 비교했을 때 PolyCom이 가지는 표현력과 효율성을 강조한다. 특히, PolyCom을 포함한 네트워크가 **최적 근사율(optimal approximation rate)**을 달성함을 보이며, 이는 Sobolev 공간에서 일반적인 매끄러운 함수(smooth functions)를 근사하는 데 PolyCom 네트워크가 최소한의 파라미터만 필요함을 의미한다.

실험적으로는 대규모 언어모델(large language models, **LLMs**)의 사전학습(pre-training) 구성에서, 밀집(dense) 및 희소(sparse) 아키텍처 모두를 대상으로 검증하였다. 기존의 활성화 함수를 PolyCom으로 대체함으로써, LLM이 데이터 내 고차 상호작용(higher-order interactions)을 포착할 수 있게 하여 정확도와 수렴 속도(convergence rates) 측면에서 성능을 향상시켰다. 광범위한 실험 결과는 제안 기법의 효과성을 입증하며, 다른 활성화 함수 대비 상당한 개선을 보여준다.

코드는 <https://github.com/BryceZhuo/PolyCom> 에서 확인할 수 있다.

![](/assets/images/posts/595/img.png)

**그림 1 설명**  
훈련 손실(training loss), 검증 퍼플렉서티(validation perplexity, **PPL**), 그리고 10억(1B) 파라미터 규모의 밀집 모델에서의 다운스트림 성능 비교. 여기서는 SwiGLU, GELU, ReLU, PolyReLU, PolyNorm 등 다양한 활성화 함수를 적용한 모델들을 비교하였다. 결과는 PolyReLU와 PolyNorm을 사용하는 모델이 더 낮은 훈련 손실과 검증 PPL을 보이며, 다운스트림 성능 또한 더 우수함을 나타낸다.

**1 서론(Introduction)**  
Transformer(Vaswani et al., 2017)는 심층학습(deep learning) 분야에 혁신을 가져왔으며, 자연어 처리(Radford et al., 2019), 컴퓨터 비전(Dosovitskiy et al., 2021), 그 외 다양한 영역(Dong et al., 2018; Arnab et al., 2021)에서 전례 없는 발전을 이끌어냈다. 주의(attention) 메커니즘을 핵심으로 하는 Transformer는 데이터 내 복잡한 관계를 포착하는 데 탁월하여 현대 기계학습 응용에서 필수적인 도구가 되었다. 그러나 이러한 폭넓은 성공에도 불구하고, 특히 **활성화 함수(activation function)**의 선택과 관련하여 여전히 개선의 여지가 존재한다. 활성화 함수는 신경망 내 각 뉴런의 출력을 결정하는 데 중요한 역할을 한다. 전통적으로는 **ReLU(Rectified Linear Unit, Nair & Hinton, 2010)**와 그 변형들(Hendrycks & Gimpel, 2016; So et al., 2021)이 계산 효율성과 구현의 용이성 덕분에 널리 사용되어 왔다. 이러한 활성화 함수들은 효과적이긴 하지만, 데이터 내 복잡한 고차(high-order) 관계를 모델링하는 데 본질적인 한계를 가진다. 특히 Transformer 아키텍처에서는 미묘하고 복잡한 의존성을 포착하는 능력이 필수적이기 때문에, 이러한 한계는 성능을 제한하는 요인이 될 수 있다.

본 논문에서는 Transformer 아키텍처의 성능을 향상시키기 위해 특별히 설계된 새로운 범주의 **다항식 조합 활성화 함수(Polynomial Composition Activation Functions, PolyCom)**를 소개한다. 기존의 활성화 함수들이 주로 선형적(linear) 또는 구간별 선형(piecewise linear) 함수인 것과 달리, PolyCom은 데이터 내 보다 복잡한 패턴을 모델링할 수 있도록 한다. 이러한 활성화 함수의 표현력 증가는 모델의 표현 능력(expressive capacity)을 한층 강화하여, 기존 방식으로는 간과될 수 있는 고차(high-order) 상호작용을 포착할 수 있게 한다. 또한 기존의 다항식(Hornik et al., 1989; Trefethen, 2019)들이 겪는 불충분한 근사 능력, 값의 폭발(exploding values), 진동적 거동(oscillatory behavior)과 같은 문제와 달리, PolyCom은 ReLU 및 전통적인 다항식보다 훨씬 강력한 표현 능력을 가지며, **Sobolev 공간(Sobolev space)** 내에서 최적 근사(optimal approximation)를 달성함을 보인다.

우리는 Transformer 모델에 **다항식 조합 활성화 함수(polynomial composition activations)**를 통합함으로써, 복잡한 데이터 해석을 요구하는 과제에서 성능 향상을 이끌어낼 수 있다고 주장한다. 이 가설을 검증하기 위해, 우리는 밀집(dense) 및 희소(sparse) 아키텍처를 모두 포함한 **대규모 언어 모델(large language models, LLMs)**의 사전학습(pre-training) 구성에서 포괄적인 실험을 수행하였다. 이 평가에서는 다양한 벤치마크를 활용하여, 다항식 조합 활성화를 적용한 Transformer와 기존 활성화 함수를 사용하는 Transformer의 성능을 비교하였다. 그 결과, 제안된 방법은 모델의 정확도를 향상시킬 뿐만 아니라 수렴 속도(convergence rate)도 가속화함을 확인하였다. 이는 다항식 조합 활성화 함수가 딥러닝 응용에서 실질적인 이점을 제공함을 시사한다.

이 논문의 주요 기여(contributions)는 다음과 같이 요약된다.

- 우리는 **다항식(polynomial)과 다른 유형의 함수를 조합(composition)**하여 만든 새로운 활성화 함수 **PolyCom**을 제안한다. 특히, PolyCom의 두 가지 구체적 예시인 **PolyReLU**와 **PolyNorm**을 소개하고, 이를 Transformer 아키텍처에 통합하는 방법을 상세히 설명한다.
- 이론적으로, PolyReLU 네트워크가 ReLU 네트워크를 근사(approximate)하는 데 필요한 학습 가능한 파라미터 개수에 대한 경계를 유도하고, 그 반대 경우도 제시한다. 추가로, 크기가 **O(ϵ^(-d/n))**인 PolyReLU 네트워크가 Sobolev 공간(Sobolev spaces) 내 임의의 함수를 오차 허용치 ϵ 내에서 근사할 수 있으며, 이때 **최적 근사율(optimal approximation rates)**을 달성함을 보인다.
- 실험적으로는, 10억(1B) 파라미터 규모의 밀집(dense) 모델과 10억 활성(1B active)·70억 총합(7B total) 파라미터 규모의 MoE(Mixture of Experts) 모델을 대상으로 새로운 활성화 함수의 효과성을 검증하였다. 두 모델 모두에서 PolyCom은 수렴 속도를 가속화할 뿐 아니라, **SwiGLU, GELU, ReLU** 등의 기존 활성화 함수들을 현저히 능가하는 성능을 보였다.

논문의 구성은 다음과 같다. **2장**에서는 PolyCom의 수학적 정식화와 Transformer 아키텍처 내 통합 방안을 제시한다. **3장**에서는 PolyCom의 향상된 표현력과 효율성을 강조하는 포괄적 이론 분석을 제공한다. **4장**에서는 대규모 언어모델(LLMs)을 대상으로 한 실험 결과를 상세히 설명한다. **5장**에서는 활성화 함수 및 Transformer 모델 내 적용과 관련된 선행 연구를 개괄한다. 마지막으로 **결론(Conclusion)**에서는 본 논문을 정리하고 향후 연구 방향을 제시한다.

**2 다항식 조합 활성화 함수 (Polynomial Composition Activation Function)**

이 장에서는 **다항식 조합 활성화 함수(PolyCom)**의 수학적 정식화와 Transformer 아키텍처에 통합되는 방법을 제시한다.

**PolyCom.** 다항식 활성화 함수(polynomial activation function)에 대한 연구는 Hornik et al. (1989)의 기초적 연구로 거슬러 올라간다. 해당 연구에서는 다항식 활성화를 사용하는 신경망은 연속 함수 공간 내에서 조밀(dense)하지 않음을 보였다. 또한, 실증적 증거에 따르면 순수한 다항식 활성화를 사용하는 심층 신경망은 성능이 저조한 경향을 보인다(Trefethen, 2019). 이러한 한계를 극복하기 위해 우리는 **다항식과 다른 함수를 조합한 새로운 형태의 PolyCom**을 제안한다. 구체적으로는 두 가지 조합 방식을 탐구한다.

![](/assets/images/posts/595/img_1.png)

여기서 r∈N은 PolyCom의 차수(order)를 나타내며, ρ는 ReLU, PReLU, Sigmoid, SiLU, 정규화(normalization) 등 임의의 함수가 될 수 있다.

두 방식의 핵심적인 차이는 **함수를 거친 뒤 거듭제곱(power)을 적용하는지, 아니면 거듭제곱을 적용한 뒤 함수를 적용하는지**에 있다. 그러나 ρ가 비선형 함수일 경우, 이론적으로 두 방식은 동등한 표현력을 가진다. 이는 다항식 항이 조합(composition)에 대해 대칭적이므로 Type I과 Type II 모두 유사한 함수 계열을 근사할 수 있기 때문이다. 즉, ρ와 다항식 거듭제곱의 순서를 바꾸어도 복잡한 비선형 함수를 근사하는 능력은 변하지 않는다.

실험적으로는 학습 가능한 계수 a\_i​를 포함한 **3차( r=3 ) PolyCom**을 사용한다. 초기화 시에는 i=1,2,…,r에 대해 a\_i = 1/r로 설정하고, a\_0 = 0으로 둔다.

Type I PolyCom의 경우, 단순성 때문에 **ReLU 함수**와의 조합을 구체적으로 고려하며, 이를 **PolyReLU**라 부른다. 차수 r의 PolyReLU는 다음과 같이 정의된다.

![](/assets/images/posts/595/img_2.png)

여기서

![](/assets/images/posts/595/img_3.png)

이다. 이 공식은 **ReLU**와 **Square ReLU** 모두를 확장한 형태로 볼 수 있다.

Type II PolyCom의 경우, 각 항의 크기가 일관되도록 거듭제곱을 정규화(normalize)하는 **PolyNorm**을 제안한다.

![](/assets/images/posts/595/img_4.png)

![](/assets/images/posts/595/img_5.png)

**그림 2 설명**  
ReLU/GELU, SwiGLU, PolyReLU, PolyNorm을 사용하는 Transformer MLP 블록의 블록 다이어그램.

- “FC”는 완전연결층(Fully Connected layer)을 의미한다.
- “x^i”는 입력 텐서 x의 i차 거듭제곱을 나타낸다.
- “a\_j​”는 학습 가능한 가중치 벡터 a의 j-번째 원소를 나타낸다.
- “N”은 정규화(normalization) 연산을 의미한다.

**Transformer에의 통합 (Integration into Transformer).**  
Transformer 아키텍처(Vaswani et al., 2017)는 **멀티헤드 어텐션(Multi-Head Attention, MHA)**과 **위치별 피드포워드 네트워크(position-wise Feed-Forward Networks, FNN)**라는 두 가지 모듈이 교대로 쌓여 구성된다. 이 중 **활성화 함수(activation function)**는 주로 FNN 층의 성능에 큰 영향을 미친다.

우리는 먼저 일반적인 FNN의 구조를 다음과 같이 정식화한다.

![](/assets/images/posts/595/img_6.png)

여기서 ρ는 ReLU, GeLU, PolyReLU, PolyNorm과 같은 활성화 함수를 나타낸다.

본 연구에서는 기존의 활성화 함수를 제안하는 **PolyCom 계열 함수(PolyReLU, PolyNorm 등)**로 대체하여 모델의 용량(capacity)과 성능을 향상시키며, 이는 그림 2에 시각적으로 설명되어 있다.

**3 이론적 분석 (Theoretical Analysis)**

2장에서 논의했듯이, **PolyReLU**와 **PolyNorm**은 동등한 표현력(expressivity)을 가진다. 분석을 단순화하기 위해, 본 장에서는 PolyReLU의 이론적 특성, 특히 그 **표현력**과 **효과성**에만 초점을 맞춘다. 추가적으로, GeLU와 SwiGLU와 같은 비선형 활성화 함수들은 원점 주변에서 테일러 다항식(Taylor polynomial)으로 국소적으로 근사될 수 있으므로, 우리는 주로 PolyReLU를 ReLU 및 다항식 활성화 함수와 비교한다. 혼동을 피하기 위해, ReLU 활성화를 사용하는 네트워크는 **ReLU 네트워크**, PolyReLU 활성화를 사용하는 네트워크는 **PolyReLU 네트워크**라고 부른다.

**3.1 PolyReLU에 의한 ReLU 네트워크 근사 (Approximating ReLU Networks by PolyReLU)**

이 절에서는 **PolyReLU 네트워크가 ReLU 네트워크를 근사(approximate)**하는 데 관한 이론적 결과를 제시한다. 다음의 보조정리(lemma)는 **ReLU, ReLU², 그리고 다항식 활성화(polynomial activation)**가 모두 PolyReLU 활성화의 특수한 경우임을 보여주며, 이를 통해 PolyReLU가 더 우수한 표현력을 지님을 강조한다. 이는 PolyReLU가 ReLU 및 다른 다항식 활성화 함수들에 비해 **더 적은 학습 가능한 파라미터(trainable parameters)**로도 더 강력한 근사 능력(approximation ability)을 가진다는 것을 의미한다.

![](/assets/images/posts/595/img_7.png)

![](/assets/images/posts/595/img_8.png)

**3.2 PolyReLU를 ReLU 네트워크로 근사하기 (Approximating PolyReLU with ReLU networks)**

이 절에서는 **PolyReLU 네트워크를 ReLU 네트워크로 근사**하는 데 관한 이론적 결과를 제시한다. 다음의 **Lemma 2**는 PolyReLU 활성화 함수가 주어진 오차 허용 범위(error tolerance) 내에서 ReLU 네트워크로 근사될 수 있음을 보여준다.

![](/assets/images/posts/595/img_9.png)

![](/assets/images/posts/595/img_10.png)

![](/assets/images/posts/595/img_11.png)

**3.3 일반적인 매끄러운 함수의 근사 (Approximation of General Smooth Function)**

Yarotsky (2017), Boullé et al. (2020)와 유사하게, 본 절에서는 **Sobolev 공간(Adams & Fournier, 2003)**의 맥락에서 PolyReLU 네트워크의 보편 근사 능력(universal approximation capabilities)을 탐구한다. 구체적으로, 우리는 PolyReLU 네트워크가 이 공간에서 **최적 근사율(optimal approximation rate)**을 달성함을 보인다. 이는 곧 PolyReLU 네트워크가 Sobolev 공간에서 일반적인 매끄러운 함수(smooth functions)를 근사하기 위해 다른 활성화 함수를 사용하는 네트워크보다 **더 적은 파라미터**만 필요하다는 것을 의미한다.

![](/assets/images/posts/595/img_12.png)

![](/assets/images/posts/595/img_13.png)

![](/assets/images/posts/595/img_14.png)

![](/assets/images/posts/595/img_15.png)

---

![](/assets/images/posts/595/img_16.png)

---
