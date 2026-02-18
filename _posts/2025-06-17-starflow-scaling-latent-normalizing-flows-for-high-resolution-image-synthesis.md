---
title: "STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis"
date: 2025-06-17 17:51:12
categories:
  - 인공지능
tags:
  - starflow
---

<https://arxiv.org/abs/2506.06276>

[STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis

We present STARFlow, a scalable generative model based on normalizing flows that achieves strong performance in high-resolution image synthesis. The core of STARFlow is Transformer Autoregressive Flow (TARFlow), which combines the expressive power of norma

arxiv.org](https://arxiv.org/abs/2506.06276)

![](/assets/images/posts/572/img.png)

**그림 1**: 다양한 종횡비로 생성된 고해상도 텍스트 조건 샘플들. 모두 3.8B 파라미터 규모의 STARFlow 모델에서 생성되었으며, 시각화의 편의를 위해 해상도가 조정되어 있음.

**초록(Abstract)**

우리는 고해상도 이미지 합성에서 강력한 성능을 보이는, 정규화 흐름(normalizing flows)에 기반한 확장 가능한 생성 모델인 **STARFlow**를 소개한다. STARFlow의 핵심 구성 요소는 **TARFlow(Transformer Autoregressive Flow)**로, 이는 정규화 흐름과 Autoregressive Transformer 아키텍처를 결합한 구조이며 최근 이미지 모델링 분야에서 인상적인 성과를 보인 바 있다.

이 연구에서는 먼저 TARFlow가 연속 확률 분포를 모델링하는 데 있어 **이론적으로 보편적(universal)**이라는 사실을 입증한다. 이 기반 위에 우리는 모델의 확장성을 크게 높여주는 일련의 아키텍처 및 알고리즘 혁신들을 도입하였다:

1. **딥-셸로우(Deep-Shallow) 구조**: 대부분의 모델 용량을 담당하는 깊은 Transformer 블록과, 계산량은 적지만 무시할 수 없는 기여를 하는 얕은 Transformer 블록 몇 개를 조합함.
2. **사전학습된 오토인코더의 잠재 공간(latent space)에서의 학습**: 픽셀을 직접 모델링하는 것보다 훨씬 효과적인 방식임을 입증함.
3. **샘플 품질을 크게 향상시키는 새로운 가이던스 알고리즘**.

무엇보다도, STARFlow는 **단일 end-to-end 정규화 흐름 모델**로 유지되며, 이는 **이산화(discretization) 없이 연속 공간에서 정확한 최대우도 학습**을 가능하게 한다.

STARFlow는 클래스 조건 및 텍스트 조건 이미지 생성 양쪽 모두에서 경쟁력 있는 결과를 달성하며, 생성된 샘플 품질은 최신 확산 모델(diffusion model)에 근접한다. 우리가 알기로, 이 연구는 이 규모와 해상도에서 정규화 흐름의 성공적인 적용을 처음으로 입증한 사례이다.

### 1. 서론

최근 몇 년간 고해상도 텍스트-이미지 생성 모델링 분야는 눈부신 발전을 이루었으며, 최첨단 기법들은 주로 두 가지 뚜렷한 범주로 나뉩니다. 한편으로는, 연속 공간(continuous space)에서 작동하는 **확산 모델(diffusion models)**(Ho et al., 2020; Rombach et al., 2022; Peebles & Xie, 2023; Esser et al., 2024)이 이미지 품질 측면에서 새로운 기준을 제시해왔습니다. 하지만 이러한 모델은 반복적인 디노이징(denoising) 과정을 필요로 하므로, 학습과 추론 모두에서 계산 비용이 매우 높다는 단점이 있습니다. 다른 한편으로는, **오토레그레시브(autoregressive) 방식의 이미지 생성 방법**(Yu et al., 2022; Sun et al., 2024; Tian et al., 2024)이 있으며, 이는 대형 언어 모델(LLMs, Brown et al., 2020; Dubey et al., 2024)의 성공에서 영감을 받아, 양자화(quantization)를 통해 이산 공간(discrete space)에서 이미지를 모델링함으로써 이러한 비효율성을 회피합니다. 그러나 양자화는 심각한 제약을 초래할 수 있으며, 결과적으로 이미지의 충실도(fidelity)를 저해할 수 있습니다. 최근에는 오토레그레시브 기법을 **연속 공간에 직접 적용하는 하이브리드 모델**(Li et al., 2024; Gu et al., 2024b; Fan et al., 2024)에 대한 탐색이 유망한 흐름으로 등장했습니다. 하지만 두 패러다임(확산과 오토레그레시브)의 본질적으로 다른 특성은 효과적인 통합에 있어 추가적인 복잡성을 유발합니다.

이 논문에서는 또 다른 모델링 접근 방식인 **정규화 흐름(Normalizing Flows, NFs)**(Rezende & Mohamed, 2015; Dinh et al., 2016)에 주목합니다. 이는 **우도 기반(likelihood-based)** 생성 모델 계열로, 최근 생성형 AI의 급류 속에서는 상대적으로 적은 관심을 받아왔습니다. 우리는 먼저 강력한 Transformer 아키텍처와 오토레그레시브 흐름(Autoregressive Flows, AFs)(Kingma et al., 2016; Papamakarios et al., 2017)을 결합한 최근 모델인 **TARFlow**(Zhai et al., 2024)를 분석하는 데서 출발합니다. TARFlow는 정규화 흐름(NF)이 유효한 모델링 프레임워크임을 보여주는 유망한 결과를 제시하지만, 이것이 확산 모델이나 이산 오토레그레시브 모델들과 비교해 확장 가능한 방법으로 작동할 수 있는지는 아직 불분명합니다. 이를 위해 우리는 **STARFlow**라는 생성 모델 계열을 제안합니다. 이 모델은 **정규화 흐름 기반 모델이 고해상도 및 대규모 이미지 모델링에도 성공적으로 일반화될 수 있음**을 최초로 입증합니다. 우리는 먼저 다중 블록 AF가 연속 분포를 모델링함에 있어 **보편성(universality)**을 지닌다는 이론적 통찰을 제공하며, 이를 기반으로 **새로운 Deep–Shallow 아키텍처**를 제안합니다. 실험 결과, 플로우 개수와 각 플로우에 할당된 Transformer의 깊이 및 너비와 같은 아키텍처 구성은 모델 성능에 결정적인 역할을 하는 것으로 나타났습니다. TARFlow에서는 모든 플로우에 균등한 깊이를 배분했지만, 우리는 대부분의 파라미터를 첫 번째 AF 블록(사전 분포에 가장 가까운)에 집중시키고, 이후 몇 개의 얕지만 의미 있는 블록을 추가하는 **비균등 구조(skewed architecture)**가 더 효과적임을 발견했습니다. 중요하게도, STARFlow는 **여전히 독립적인 end-to-end 정규화 흐름 모델**로서, 연속 공간에서 양자화 없이 **최대우도 학습**을 지원합니다. 우리는 픽셀 공간이 아닌, **사전학습된 오토인코더의 잠재 공간(latent space)**에서 AF를 학습하며, 이는 고해상도 입력을 보다 효과적으로 모델링할 수 있게 해줍니다. 이는 직관적이지만 중요한 관찰로, 실험에서도 직접 픽셀 공간에서 학습할 때보다 훨씬 우수한 성능을 보였습니다. TARFlow와 유사하게, **노이즈 주입(noise injection)**은 여전히 중요합니다. 우리는 디코더를 파인튜닝하며 **노이즈가 섞인 latent**에서 학습하고, 동시에 샘플링 파이프라인을 단순화했습니다. 또한, AF에 대한 **classifier-free guidance (CFG)** 알고리즘을 더 원리적인 방식으로 재검토하고, **새로운 가이던스 알고리즘**을 제안함으로써 특히 높은 guidance weight에서 **텍스트-이미지 생성의 품질을 크게 향상**시켰습니다.

이러한 혁신을 통해, 우리는 NF 기반 모델이 **대규모 고해상도 이미지 생성에 성공적으로 적용된 첫 사례**를 제시합니다. 우리의 접근 방식은 기존 확산 기반 또는 오토레그레시브 기반 방법에 대한 **확장 가능하고 효율적인 대안**을 제공하며, 클래스 조건 이미지 생성과 대규모 텍스트-이미지 합성 모두에서 **경쟁력 있는 성능**을 보입니다. 또한, STARFlow는 매우 유연한 프레임워크로, **이미지 인페인팅(inpainting)**이나 **지시 기반 이미지 편집(instruction-based editing)**과 같은 다양한 설정에도 **간단한 파인튜닝을 통해 적용이 가능함**을 실험을 통해 보여줍니다.

### 2. 사전 지식

#### 2.1 정규화 흐름(Normalizing Flows)

본 논문에서는 **정규화 흐름(Normalizing Flows, NFs)**(Rezende & Mohamed, 2015; Dinh et al., 2014, 2016)을 **변수 변환 공식(change of variable formula)**을 따르는 **우도 기반(likelihood-based)** 모델의 한 종류로 다룹니다.

![](/assets/images/posts/572/img_1.png)

![](/assets/images/posts/572/img_2.png)

![](/assets/images/posts/572/img_3.png)

---

말은 어렵게했지만 결국 diffusion대신에 autoregressive한 transformer 썼다.

### ? 배경 비교:

- **Diffusion 모델들**: 고품질 이미지를 생성하지만,
  - 수십~수백 번의 반복적 디노이징이 필요해서
  - **학습과 추론 비용이 큼**
- **기존 Autoregressive 모델들 (예: LLM 기반 이미지 생성)**:
  - 효율적이지만
  - \*\*이산 공간(quantized space)\*\*에서 작동 → 이미지 품질 저하 우려
- **TARFlow / STARFlow (제안 모델)**:
  - 연속 공간(continuous space)에서 **autoregressive Transformer + normalizing flow**를 사용
  - **Diffusion처럼 정밀하게**
  - **Autoregressive처럼 효율적으로**
---

### ? 핵심 전략:

- 이미지 공간을 직접 모델링하지 않고,  
  → **Autoencoder의 latent 공간**에서 학습
- 여러 개의 autoregressive flow 블록을 Transformer로 쌓아서  
  → **전체 flow를 end-to-end로 학습**
- sampling은 정규화 흐름이기 때문에  
  → **정확한 likelihood 계산 가능**
---

### ? 한 줄 요약:

> "Diffusion의 고품질 + Autoregressive의 효율성"을 정규화 흐름 기반으로 통합한 모델이 STARFlow입니다.
---

더 간단하게 요약하면:

> \*\*"Transformer 기반 autoregressive flow로 diffusion을 대체하려는 시도"\*\*죠.
---

### 3. STARFlow

본 절에서는 **STARFlow(Scalable Transformer Autoregressive Flow)**를 제안합니다. 이는 **정규화 흐름(NF)** 기반의 고해상도 이미지 생성의 한계를 확장하는 방법론입니다. 우리는 먼저 §3.1에서 **이론적 근거를 바탕으로 AF(Autoregressive Flow)**가 범용적인 모델링 기법으로서 얼마나 강력한 표현력을 지니는지를 입증합니다. 이 기반 위에서 기존 TARFlow를 다음과 같은 핵심 측면에서 개선한 STARFlow의 주요 접근법을 제안합니다:

1. **더 나은 아키텍처 구성** (§3.2)
2. **잠재 공간(latent space)에서의 학습 전략** (§3.3)
3. **새로운 가이던스 알고리즘** (§3.4)

학습 및 추론 파이프라인에 대한 전체적인 흐름은 그림 4(Fig. 4)에 제시되어 있습니다.

### 3.1 왜 TARFlow는 강력한 생성 모델인가?

경험적으로 TARFlow가 매우 경쟁력 있다는 것은 이미 확인되었지만(Zhai et al., 2024), 우리는 **모델링 관점**에서 질문을 던집니다:

> TARFlow는 충분히 표현력이 뛰어나서 확장할 가치가 있는가?

여기서 우리는 다음과 같은 주장을 합니다:

![](/assets/images/posts/572/img_4.png)

![](/assets/images/posts/572/img_5.png)

![](/assets/images/posts/572/img_6.png)

이 명제는 우리가 **대규모 데이터에서 AF를 확장하는 것이 정당화될 수 있음**을 보여줍니다.  
특히, **T = 2**라는 최소 설정에서는 완전한 보편성은 얻을 수 없지만,

> 자연 이미지처럼 차원이 높은 도메인에서는 이 제한은 **무시할 수 있을 정도로 미미**합니다.

![](/assets/images/posts/572/img_7.png)

**그림 2**: 2-블록 오토레그레시브 플로우 예시.
---

![](/assets/images/posts/572/img_8.png)

![](/assets/images/posts/572/img_9.png)

![](/assets/images/posts/572/img_10.png)

그러면 transformer가 아닌 bidirectional lstm이나 gru로서 해야되는거아니야 아니면 mamba쪽의 sb 계열 모델로 양방향 구현해야되고?

ChatGPT의 말:

아주 날카로운 지적이에요. 지금 하신 말은 본질적으로 다음과 같은 질문입니다:

> "**양방향 정보를 쓰고 싶다면**, 굳이 autoregressive transformer를 여러 블록 쌓기보다, **처음부터 bidirectional 구조(RNN/LSTM/GRU 혹은 structured state-space 모델들)로 구현하면 되는 거 아닌가?**"

그에 대한 답은 \*\*"그렇다, 하지만..."\*\*입니다. 아래에 구조적으로 설명드릴게요.
---

## ✅ 왜 Transformer 기반 autoregressive flow를 고집했을까?

### 1. **정규화 흐름의 기본 제약: invertible (가역성)**

- NF는 생성과 우도 추정을 동시에 하기 위해, **입력 ↔ latent 공간 간의 가역 함수**여야 함.
- 대부분의 **Bi-LSTM, Bi-GRU, Bi-Mamba** 등은 일반적으로 **가역(invertible)** 하지 않음.
  - 순방향은 있지만, 역방향 계산이 불가능하거나 명시적으로 정의되지 않음
  - 즉, **샘플링이 안 됨**

→ 그래서 flow 모델에서는 **가역 구조 설계**가 핵심입니다.
---

### 2. **Autoregressive Flow는 가역성과 우도 계산을 보장하면서도 양방향 구조를 흉내낼 수 있음**

- TARFlow처럼 여러 개의 autoregressive block을 \*\*다른 순서(ordering)\*\*로 쌓으면,
  - 정보가 양방향으로 전달되는 효과를 얻을 수 있고
  - 동시에 **우도(log-likelihood)** 계산도 가능하며
  - **역변환 (sampling)** 도 가능

→ 즉, 양방향성은 직접 쓰진 않지만, **우회적으로 구현**됨
---

## ? 그럼 Mamba 같은 Structured State Space 모델은 왜 안 썼을까?

- Mamba나 S4, Hyena 같은 **SSM(Structured State Space Models)** 계열은 본질적으로도 **시간축을 modeling**하는 데에 탁월함
- 특히 Mamba는 최근엔 양방향 순서 없이도 긴 context를 잘 잡는다고 입증됨

하지만 한계는:

1. **가역성 보장이 어려움**  
   대부분의 SSM 계열은 deterministic한 forward pass는 가능해도,  
   → **flow처럼 정확히 invertible 구조로 설계되어 있지 않음**
2. **우도 계산이 어려움**  
   flow의 장점은 학습 시 exact log-likelihood 계산이 가능한 건데,  
   SSM 모델들은 보통 이를 직접 계산하거나 보장하지 않음
---

## ? 결론: 왜 transformer autoregressive로 했는가?

- LSTM/GRU/Bi-Mamba는 양방향 정보처리에 적합하긴 하지만
  - 가역성이 없어서 정규화 흐름에 직접 쓰기 힘듦
- TARFlow 방식은:
  - 여러 방향으로 autoregressive block을 쌓는 것으로 **양방향 흐름을 우회적으로 구현**
  - 동시에 **정확한 우도 계산 + 샘플링**도 가능한 구조
---

## ? 그래서 결국 요약하면:

> "양방향 처리만 생각하면 Bi-GRU나 Mamba가 좋아 보일 수 있지만,  
> **flow 모델의 '가역성 + 우도 계산 + 샘플링'이라는 제약 조건을 만족하려면**,  
> TARFlow처럼 **Transformer 기반 autoregressive 구조를 스택으로 쌓는 것**이 가장 실용적인 해법"입니다.









실제 구조를 bidirectional한 lstm, gru, ssm 모델이라면 뭐가 문제일까?

![](/assets/images/posts/572/img_11.png)

![](/assets/images/posts/572/img_12.png)

![](/assets/images/posts/572/img_13.png)

![](/assets/images/posts/572/img_14.png)

![](/assets/images/posts/572/img_15.png)

![](/assets/images/posts/572/img_16.png)

![](/assets/images/posts/572/img_17.png)

![](/assets/images/posts/572/img_18.png)

![](/assets/images/posts/572/img_19.png)

![](/assets/images/posts/572/img_20.png)

![](/assets/images/posts/572/img_21.png)

![](/assets/images/posts/572/img_22.png)

![](/assets/images/posts/572/img_23.png)

![](/assets/images/posts/572/img_24.png)

![](/assets/images/posts/572/img_25.png)

그냥 latent score flow models이라고 불러야할판
---

![](/assets/images/posts/572/img_26.png)

**그림 3**: 위에서 아래로, 8개의 flow 블록을 가진 TARFlow 모델에서 처음 0개, 3개, 8개의 flow 블록에 대해 guidance를 적용한 결과. 그림에서 알 수 있듯, guidance는 상위 3개의 블록까지만 효과적임을 확인할 수 있다.

### 3.2 제안하는 아키텍처

§3.1의 유도 결과는, **현실적인 계산 자원 한도 내에서 확장 가능한 AF 아키텍처를 재설계**할 필요가 있음을 시사합니다.  
즉, 많은 수의 flow 블록을 추가할 필요는 없으며, 실제로는 **T = 2만으로도 충분한 경우가 많습니다**.

그러나 여전히 남는 의문은 다음과 같습니다:

> **"그 제한된 계산 자원을 각 블록에 어떻게 분배할 것인가?"**

우리는 먼저 기존 **TARFlow 아키텍처 구성 방식**을 살펴봅니다. TARFlow는 각 flow 블록에 **동일한 크기의 Transformer 레이어를 균등하게 할당**하도록 설계되어 있습니다. 그런데 우리가 TARFlow를 재현한 결과에서는, **가장 상위의 몇 개 AF 블록에 대부분의 계산량이 집중되는 경향**이 나타났습니다 (guidance 관점에서 측정, 그림 3 참고).  
이러한 현상을 통해 우리는 다음과 같은 가설을 세웁니다:

> **엔드-투-엔드 학습 과정에서 네트워크는 노이즈에 가장 가까운 레이어에 계산을 집중하는 경향을 보인다.**  
> 이 동작은 diffusion 모델과는 대조적입니다.

### Deep–Shallow 아키텍처

우리가 제안하는 아키텍처는, **기존의 오토레그레시브 언어 모델 (예: LLaMA (Dubey et al., 2024))**을 기반으로 한 **일반화된 deep–shallow 구조**로 직관적으로 이해할 수 있습니다.

![](/assets/images/posts/572/img_27.png)

이러한 **비대칭적인 설계(asymmetric design)**는,

- 깊은 블록이 **가우시안 기반의 언어 모델(Gaussian language model)** 역할을 하고,
- 얕은 블록들은 **학습된 이미지 토크나이저(image tokenizer)**처럼 작동하게 만듭니다.

### 조건부 STARFlow (Conditional STARFlow)

이 설계는 **조건부 생성**에도 자연스럽게 확장됩니다. 단순히 **flow 입력 앞에 조건 신호**(예: 클래스 레이블, 캡션)를 덧붙이면 됩니다.

흥미롭게도, 초기 실험 결과에 따르면:

- **깊은 블록에만 조건을 부여하고**,
- 얕은 블록들은 **이미지의 지역적인 정제(local refinement)**에만 집중하게 해도,  
  → **성능 손실이 전혀 발생하지 않았습니다.**

이 방식은 다음과 같은 장점을 제공합니다:

1. 전체 아키텍처를 단순화하고
2. **깊은 블록에 사전학습된 언어 모델**을 그대로 사용 가능하게 하며  
   → 구조 변경 없이도 초기화 가능

결과적으로, 우리가 제안한 이미지 생성기는 **어떤 LLM의 의미 공간(semantic space)**에도 바로 통합될 수 있으며, 별도의 텍스트 인코더 없이도 작동이 가능합니다.

![](/assets/images/posts/572/img_28.png)

![](/assets/images/posts/572/img_29.png)

![](/assets/images/posts/572/img_30.png)

![](/assets/images/posts/572/img_31.png)

![](/assets/images/posts/572/img_32.png)

**그림 4**: 텍스트-이미지 생성을 위한 제안 모델의 **오토레그레시브 추론 과정(좌측)**과 **병렬 학습 과정(우측)**을 나타낸 그림.  
초록색 위쪽 화살표는 **AF의 역방향 단계(inverse step)**, 보라색 아래쪽 화살표는 **정방향 단계(forward step)**를 나타내며, 이는 식 (2)를 참고하면 된다.

![](/assets/images/posts/572/img_33.png)

**그림 5**:  
(a) TARFlow(Zhai et al., 2024)에서의 기존 가이던스 방식  
(b) ImageNet 256×256 기준, 본 논문에서 제안한 가이던스 방식

### 3.4 오토레그레시브 플로우를 위한 Classifier-Free Guidance 재고

**Classifier-Free Guidance (CFG)**는 원래 **diffusion 모델**을 위해 제안된 방식이며 (Ho & Salimans, 2021), 오늘날의 생성 모델링에서 **핵심 기술 중 하나로 자리잡았습니다.** 이 기법은 오토레그레시브(AR) 모델을 포함한 다양한 아키텍처에서도 광범위하게 효과적인 것으로 입증되었습니다 (Yu et al., 2022). 높은 수준에서 보았을 때, **CFG는 조건부 예측과 비조건부 예측 간의 차이를 증폭시켜**, 보다 **모드 중심적인(mode-seeking)** 생성 결과를 유도합니다.

![](/assets/images/posts/572/img_34.png)

![](/assets/images/posts/572/img_35.png)

![](/assets/images/posts/572/img_36.png)

![](/assets/images/posts/572/img_37.png)

![](/assets/images/posts/572/img_38.png)

**그림 6**: (a) 이미지 인페인팅, (b) 인터랙티브 편집

### 3.5 응용 사례 (Applications)

**STARFlow는 다양한 조건 하에서 고품질 이미지를 생성할 수 있을 뿐 아니라, 자연스럽게 다양한 다운스트림 응용에도 확장 가능한 범용 생성 모델입니다.** 여기서는 두 가지 예시를 소개합니다: **이미지 인페인팅**과 **인터랙티브 편집**입니다.

#### ?️ 학습 없이 수행하는 인페인팅 (Training-Free Inpainting)

1. 먼저 마스크 처리된 이미지를 잠재 공간으로 매핑하고,
2. 마스크된 영역은 **가우시안 노이즈**로 대체합니다.
3. 이후 역방향 샘플링(reverse sampling)을 수행하여,
   - 마스크되지 않은 영역은 **정답(ground truth)** 픽셀로 복원합니다.
4. 이 과정을 반복 수행하여 최종 인페인팅된 이미지를 생성합니다.

#### ✏️ 인터랙티브 생성 및 편집 (Interactive Generation and Editing)

우리는 STARFlow를 **이미지 편집 데이터셋**에 대해 파인튜닝하여 (그림 6b 참고), **생성과 편집을 하나의 조건부 AF 모델로 함께 처리할 수 있도록** 했습니다.

또한 STARFlow의 **가역성(invertibility)** 덕분에  
→ 이미지를 직접 잠재 공간으로 인코딩할 수 있어,  
→ **인터랙티브한 편집 사용 사례**에도 적합합니다.

![](/assets/images/posts/572/img_39.png)

**그림 7**: STARFlow가 생성한 ImageNet 256×256 및 512×512 해상도의 무작위 샘플들 (ω = 3.0)

### 4. 실험 (Experiments)

#### 4.1 실험 설정 (Experimental Settings)

**데이터셋(Dataset)**  
STARFlow는 **클래스 조건(class-conditioned)** 및 **텍스트-이미지 생성(text-to-image generation)** 두 가지 과제를 대상으로 실험을 진행했습니다.

- 클래스 조건 생성의 경우, **ImageNet-1K**(Deng et al., 2009)에서 **256×256** 및 **512×512** 해상도로 실험을 수행했습니다.
- 텍스트-이미지 생성에서는 두 가지 설정을 사용했습니다:
  1. **제한된 설정(constrained setting)**: **CC12M**(Changpinyo et al., 2021) 데이터셋을 사용하며, 각 이미지에 대해 Gu et al. (2024a)의 방식에 따라 **합성 캡션(synthetic caption)**을 부여합니다.
  2. **확장된 설정(scaled setting)**: CC12M을 포함한 사내 구축 데이터셋을 사용하여 **총 약 7억 개의 텍스트-이미지 쌍**으로 학습합니다.

**평가 지표(Evaluation)**  
기존 연구들과 동일하게, 생성된 이미지의 사실성과 다양성을 정량화하기 위해 **Fréchet Inception Distance (FID)**(Heusel et al., 2017)를 보고합니다.

- 텍스트-이미지 생성의 경우, 모델의 **제로샷(zero-shot)** 성능을 평가하기 위해 **MSCOCO 2017**(Lin et al., 2014) 검증 세트를 사용합니다.
- 또한, 부록 C에는 **GenEval**(Ghosh et al., 2023) 등 **추가적인 평가 지표**도 보고합니다.

**모델 및 학습 세부사항 (Model and Training Details)**  
모든 모델은 **Dubey et al. (2024)**의 설정을 따르며, **RoPE positional encoding**(Su et al., 2024)을 사용합니다.

- 기본적으로, 아키텍처는 다음과 같이 설정됩니다:
  - 클래스 조건 모델: d(N)=18(6), 모델 차원: 2048 → **총 14억(1.4B) 파라미터**
  - 텍스트-이미지 모델: d(N)=24(6), 모델 차원: 3096 → **총 38억(3.8B) 파라미터**
  - (※ §3.2에서 설명한 deep–shallow 구조를 따름)
- STARFlow는 **압축된 잠재 공간**에서 작동하므로, **패치 크기 p=1**로 모든 모델을 학습할 수 있습니다.
- 텍스트 인코더로는 기본적으로 **T5-XL**(Raffel et al., 2020)을 사용합니다.

또한, STARFlow의 **범용성**을 강조하기 위해,

- 텍스트 인코더 없이
- **사전학습된 LLM(Gemma2, Team et al., 2024)**으로 **deep block을 초기화한 변형 모델**도 학습합니다.
- 모든 모델은 해상도 256×256에서 **4억 장의 이미지**로 **글로벌 배치 사이즈 512**로 사전학습됩니다.
- 고해상도 파인튜닝은 **입력 길이 증가**를 통해 수행됩니다.

텍스트-이미지 모델의 경우 **혼합 해상도 학습(mixed-resolution training)**을 사용하여 가변 입력 길이를 지원합니다:

- 이미지를 9개의 해상도 그룹(shape bucket)으로 사전 분류한 뒤,
- 시퀀스로 평탄화하여 통합된 방식으로 처리합니다.

※ 자세한 설정은 **부록 B(Appendix B)**를 참고하세요.

### 4.2 결과 (Results)

#### ✅ 기존 기법과의 비교 (Comparison with Baselines)

우리는 **클래스 조건 ImageNet-256**에서 STARFlow의 성능을 벤치마크하고, **이산(discrete)** 및 **연속(continuous)** 영역 모두에서의 **diffusion 모델과 autoregressive 모델**과 비교를 수행했습니다 (표 1).

공정한 비교를 위해,

- **TARFlow(Zhai et al., 2024)**를 **픽셀 공간(pixel space)**에서
- 유사한 파라미터 수 및 원래 아키텍처(8개의 flow, 각 flow에 8층, 너비 1280)로 학습하였습니다.

또한, STARFlow와 동일한 구조이되 입력만 픽셀로 바꾸고 패치 크기를 선형 확장한 **deep–shallow 구조의 변형 버전**도 함께 실험하였습니다.

결과적으로:

- **NF 모델들 중에서는 deep–shallow 구조가 항상 표준 구조보다 뛰어났고**,
- **입력을 latent space로 전환하면 성능이 추가로 향상**되었습니다.
- 전체적으로 STARFlow는 **기존 베이스라인들과 비교해 경쟁력 있는 성능**을 보여주었습니다 (표 1 및 표 2 참조).

특히, **ImageNet 256×256에서의 FID는 디코더 성능 한계에 거의 도달**한 상태입니다 (자세한 내용은 부록 B 참고).

또한, **COCO에서의 제로샷 평가(Table 3)**에서는 STARFlow가 텍스트 조건 생성에서 **강력한 성능**을 보이며, **NF도 확장 가능하고 경쟁력 있는 생성 프레임워크가 될 수 있음을 시사합니다.**

#### ? 정성적 결과 (Qualitative Results)

- **그림 7과 그림 8**은 각각 **클래스 조건 및 텍스트 조건 생성 샘플**을 보여줍니다. STARFlow는 다양한 종횡비(aspect ratio)를 지원하며, **최신 diffusion 및 autoregressive 방식과 유사한 수준의 고해상도 이미지**를 생성합니다.
- **그림 9**는 이미지 편집 사례를 보여주며, 모델이 학습된 prior를 활용하여 **입력 이미지와 간단한 설명만으로 다양한 편집 명령을 수행**할 수 있음을 나타냅니다.
- 추가적인 **정성적 샘플 및 인터랙티브 편집 결과**는 부록 G에 수록되어 있으며, 이는 STARFlow가 생성할 수 있는 **다양성과 정확성의 폭을 뒷받침**합니다.

#### ? Diffusion 및 Autoregressive 모델과의 비교

우리는 STARFlow와 diffusion, autoregressive(AR) 모델을 추가 비교하며 **학습 다이나믹(training dynamics)**을 분석했습니다.

- **그림 10a**는 거의 동일한 아키텍처를 사용했을 때의 **FID 변화 추이**를 보여줍니다.
- 4,096개의 샘플로 평가할 때는 STARFlow와 다른 베이스라인 간의 FID 차이가 크지 않지만,
- **50,000개의 샘플로 평가하면** STARFlow가 **모든 학습 체크포인트에서 가장 낮은 FID**를 달성했습니다.

→ 이는 STARFlow가 **더 다양한 샘플을 생성**함을 시사하며,  
→ **소규모 평가셋으로는 이러한 다양성이 완전히 포착되지 않을 수 있음**을 보여줍니다.

![](/assets/images/posts/572/img_40.png)

**그림 8**: STARFlow가 생성한 다양한 종횡비의 텍스트-이미지 샘플 (ω = 4.0). 시각화를 위해 해상도는 비율에 맞게 조정됨.

![](/assets/images/posts/572/img_41.png)

**그림 9**: STARFlow를 활용한 이미지 편집 예시. 입력 이미지와 간단한 텍스트 설명만으로, 모델은 학습된 prior를 바탕으로 다양한 편집 작업을 자연스럽게 수행함.

![](/assets/images/posts/572/img_42.png)

**표 1**: 클래스 조건 ImageNet 256×256 (FID-50K)

![](/assets/images/posts/572/img_43.png)

**표 2**: 클래스 조건 ImageNet 512×512 (FID-50K)

![](/assets/images/posts/572/img_44.png)

**표 3**: COCO에서의 제로샷 텍스트-이미지 생성 (FID-30K)

![](/assets/images/posts/572/img_45.png)

**그림 10**: 포괄적인 제거 실험(ablation study) 결과

**추론 속도 비교 (Fig. 10a)**  
그림 10a는 **Diffusion, AR(Autoregressive), STARFlow 모델 간의 추론 처리량**을 **단일 H100 GPU** 기준으로 비교한 결과도 포함하고 있습니다.

- **Diffusion 모델**은 **정제 단계 수에 따라 추론 시간이 선형적으로 증가**합니다. 최고 성능(FID)을 달성하려면 **약 250단계**가 필요하며, 따라서 **세 모델 중 가장 느립니다**.
- 반면, **AR 및 STARFlow**에서는 각 단계가 **가벼운 순방향 연산**으로 구성되어 있고, **토큰당 비용이 낮기 때문에**, 배치 크기가 커질수록 처리량이 증가합니다. 특히 배치 크기가 **32 이상**일 경우, STARFlow는 **deep 블록에만 guidance를 적용하고**, **토큰별 multinomial sampling 루프를 제거함으로써** AR 기반 모델보다 **더 우수한 추론 성능 확장성**을 보입니다.

**CFG 전략 비교 (Fig. 10b)**  
그림 10b는 **CFG(classifier-free guidance) 전략을 비교**한 결과입니다.

- **Zhai et al. (2024)**에서 사용한 원래 전략은 **“급격한 하락-급상승(dip-and-spike)” 패턴**을 보이며, 특정 guidance weight 근처에서만 최적의 FID를 기록합니다. 그 구간을 벗어나면 성능이 급격히 악화됩니다.
- **“annealing trick”**을 적용하더라도, 두 스케일(해상도) 모두에서 여전히 성능 저하가 뚜렷하게 나타납니다.
- 반면, **본 논문에서 제안한 CFG는**
  - 별도의 트릭 없이도 기존 전략보다 **더 나은 최적 성능**을 보이며,
  - **훨씬 넓은 guidance weight 범위에서도 안정적인 품질을 유지**합니다.

→ 결과적으로, 텍스트 조건 생성에서 **튜닝 유연성**이 훨씬 향상됩니다.

**확장성 분석 (Fig. 10c, 10d)**  
모델의 확장성을 평가하기 위해, **deep 블록의 깊이를 달리하면서 학습 중 성능 변화를 분석**하였습니다.

- 그림 10c: **Negative Log-Likelihood (NLL)**
- 그림 10d: **FID (4096개 샘플 기준)**

→ 두 지표 모두, **모델이 깊을수록 더 빠르게 수렴하며, 최종 성능도 향상**됨을 보여줍니다.  
이는 모델의 표현력이 증가했음을 시사합니다.

**모델 설계에 대한 제거 실험 (Fig. 10e, 10f)**  
**명제 1(Prop. 1)**의 이론적 통찰을 실증적으로 검증하기 위해, **deep 블록의 층 수 T**가 모델 표현력에 어떤 영향을 미치는지를 분석했습니다.

- T<2일 경우 성능이 급격히 하락하지만,
- T≥2에서는 성능이 유사하게 유지되며, 이는 명제 1과 일치하는 결과입니다.

또한 그림 10e, 10f에서는 **deep 블록의 수와 깊이를 각각 제거 실험(ablation)** 했으며,  
→ **수량보다는 블록의 깊이**가 더 중요하다는 점을 확인했습니다.  
→ 이는 아키텍처 설계 시 **실용적인 지침**을 제공합니다.

### 5. 관련 연구 (Related Work)

#### ? 연속 정규화 흐름(Continuous Normalizing Flows), 플로우 매칭(Flow Matching), 디퓨전 모델

**정규화 흐름(Normalizing Flows, NFs)**는 **연속 시간 모델(continuous-time)**로 확장될 수 있으며, 이러한 확장 형태가 바로 **연속 정규화 흐름(CNFs, Chen et al., 2018)**입니다. CNF는 변환 과정을 **상미분방정식(ODE)**으로 모델링하며, 명시적인 가역 매핑을 요구하지 않으면서도 **야코비안 계산을 trace 연산으로 단순화**합니다 (Grathwohl et al., 2018). 다만, 이 경우 **확률적 추정(stochastic estimator)**이 필요하며 (Hutchinson, 1989), 노이즈로 인해 안정성 문제가 발생할 수 있습니다.

**Flow Matching**(Lipman et al., 2023)은 CNF에서 영감을 받아 개발된 방식으로, **Tweedie의 보조정리(Tweedie’s Lemma, Efron, 2011)**에 기반한 **벡터 필드(vector field)**를 사용하여, 사전 분포와 실제 데이터 사이의 **샘플 단위 보간(interpolation)**을 학습합니다.

- CNF와 NF는 **가역적인 변환을 통한 정확한 우도 최적화**를 추구하는 반면,
- Flow Matching은 **변분 학습 목적(variational training objective)**을 공유하면서 **디퓨전 모델과 더 유사한 방향**으로 정렬되어 있습니다.

#### ? 오토레그레시브 모델(Autoregressive Models)

**이산 오토레그레시브(discrete autoregressive)** 모델, 특히 **대형 언어 모델(Large Language Models)** (Brown et al., 2020; Dubey et al., 2024; Guo et al., 2025)은 오늘날 생성 AI의 중심을 차지하고 있으며, **다음 토큰 예측(next-token prediction)**을 확장하는 방식으로 발전해왔습니다.

- **Scaling laws**(Kaplan et al., 2020)는, 데이터와 파라미터 양을 늘릴수록 **성능 향상이 예측 가능하게 나타난다**는 것을 보여주었습니다.
- 이러한 모델들은 현재 **멀티모달 이해 및 생성 시스템의 핵심 구성 요소**로 사용되고 있습니다 (Liang et al., 2024; Sun et al., 2024; Tian et al., 2024; Li et al., 2025).

최근에는 **양자화(quantization)로 인한 정보 손실을 극복하기 위해**, 오토레그레시브 모델링을 **연속 공간으로 확장**하려는 연구도 활발히 진행되고 있습니다.

- 예: **가우시안 혼합(Mixture of Gaussians)** 기반 방식 (Tschannen et al., 2024a,b),
- 또는 **디퓨전 디코딩(diffusion decoding)**을 활용하는 방식 (Li et al., 2024; Gu et al., 2024b; Fan et al., 2024)

이 외에도, **오토레그레시브와 디퓨전 모델의 패러다임을 통합하려는 하이브리드 접근 방식**들도 등장하고 있습니다 (Gu et al., 2024a; Zhou et al., 2024; OpenAI, 2024).

### 6. 결론 및 한계 (Conclusion and Limitation)

우리는 본 논문에서 **STARFlow**를 소개하였다. 이는 고해상도 이미지 및 대규모 텍스트-이미지 생성에까지 확장 가능한 **최초의 잠재 공간 기반 정규화 흐름(latent-based normalizing flow)** 모델이다. 실험 결과는, **정규화 흐름(normalizing flows)**이 확장 가능한 생성 모델링 방법이며, 강력한 디퓨전 및 오토레그레시브 모델들과 **비교 가능한 성능**을 달성할 수 있음을 보여준다.

하지만 본 연구에는 몇 가지 **한계점**도 존재한다. 예를 들어, 구현의 단순성을 위해 **사전 학습된 오토인코더(pretrained autoencoders)**에 전적으로 의존했지만, 이로 인해 **잠재 공간(latent space)**과 **NF(normalizing flow)** 모델을 **공동으로 설계하는 가능성**은 탐구되지 못했다.

또한, 우리는 이번 연구에서 **고품질 모델 학습**에 중점을 두었으나, 그 대가로 **추론 속도는 최적화되지 않은 상태**에 머물렀다.

평가 측면에서도, 본 연구는 **클래스 조건(class-conditional)** 및 **텍스트 조건(text-conditional)** 이미지 생성을 **표준 벤치마크 데이터셋**에 한정하여 실험을 수행하였다. 이 접근 방식이 **비디오나 3D 장면(video, 3D scenes)**과 같은 **다른 모달리티(modality)**나  
보다 **다양하고 실제적인(real-world)** 데이터 분포에 대해 **얼마나 일반화될 수 있는지는 향후 연구 과제**로 남아 있다.

**감사의 말 (Acknowledgements)**  
Ying Shen, Yizhe Zhang, Navdeep Jaitly, Alaa El-Nouby, Preetum Nakkiran과의 유익한 논의에 감사드립니다. 또한 본 연구가 가능하도록 리더십을 제공한 **Samy Bengio**에게도 깊은 감사를 드립니다.
