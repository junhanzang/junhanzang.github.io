---
title: "DoPE: Denoising Rotary Position Embedding"
date: 2025-11-30 21:44:58
categories:
  - 인공지능
---

<https://arxiv.org/abs/2511.09146>

[DoPE: Denoising Rotary Position Embedding](https://arxiv.org/abs/2511.09146)

**초록**  
Transformer 모델의 Rotary Position Embedding(RoPE)은 구조적으로 길이 외삽(length extrapolation)에 한계를 가지고 있다. 본 연구에서는 위치 인코딩이 적용된 어텐션 맵을 잡음이 포함된 feature map으로 재해석하고, feature map에서 이상치 주파수 대역(outlier frequency bands)을 검출하기 위해 절단된 행렬 엔트로피(truncated matrix entropy)에 기반한 **훈련 불필요(training-free) 방식**의 **Denoising Positional Encoding(DoPE)**을 제안한다. 또한 feature map의 잡음 특성을 활용하여, 추가 파라미터 없이(parameter-free) Gaussian 분포로 재파라미터화함으로써 강인한 길이 외삽 성능을 달성한다.

우리의 방법은 **어텐션 싱크(attention sink) 현상의 근본 원인**과 그것이 **절단된 행렬 엔트로피와 연결되는 메커니즘**을 이론적으로 규명한다. Needle-in-a-haystack 실험과 many-shot in-context learning 태스크에서 DoPE는 최대 **64K 토큰**에 이르는 확장된 컨텍스트 환경에서도 **검색 정확도와 추론 안정성을 크게 향상**시키는 것으로 나타났다. 결과적으로, 제안된 위치 임베딩 디노이징 전략은 attention sink를 효과적으로 완화하고 균형 잡힌 어텐션 패턴을 복원하여, **길이 일반화(length generalization)를 향상시키는 단순하면서도 강력한 해결책**임을 입증한다.

### 1 서론

위치 인코딩(position encoding)은 대규모 언어 모델(LLM)의 핵심 구성 요소로, 토큰 간 상호작용에 중요한 영향을 미친다. 어텐션 스코어는 쿼리 벡터와 키 벡터의 내적(dot product)으로 계산된다.

![](/assets/images/posts/604/img.png)

일반적으로 위치 인코딩은 시퀀스의 순서를 반영하기 위해 쿼리와 키 벡터에 더해진다. 다양한 기법들 중 **Rotary Position Embedding (RoPE)** (Su et al., 2024)은 쿼리와 키를 회전 변환해 상대적 위치 정보를 곧바로 내적 연산에 반영할 수 있다는 장점으로 널리 사용된다. 이때 RoPE는 다음과 같이 적용된다.

![](/assets/images/posts/604/img_1.png)

그러나 DAPE(Zheng et al., 2024)가 위치 인코딩을 어텐션 스코어 위 MLP로 대체하고, NoPE(Wang et al., 2024)가 어떠한 위치 인코딩도 사용하지 않는 방식을 제시한 것처럼, 기존 RoPE 방식이 Transformer(Vaswani et al., 2017) 성능을 제약할 수 있다는 연구들이 등장하고 있다.

본 연구에서는 위치 인코딩을 **절단된 행렬 엔트로피(truncated matrix entropy)**(Xiong et al., 2024)를 통해 **잡음이 포함된 feature map**으로 개념화한다. 우리는 feature의 잡음 수준을 측정하고, **추가 파라미터가 없는(parameter-free) Gaussian 분포 기반 재파라미터화**를 통해 길이 외삽(length extrapolation)을 수행한다. 구체적으로 본 논문의 주요 기여는 다음과 같다.

• **절단된 행렬 엔트로피를 도입해 noisy head를 식별하고, 위치 인코딩을 파라미터 없이 Gaussian 분포로 모델링하여 길이 외삽 성능을 달성했다.**  
• **이론적·실험적으로, 어텐션 스코어 계산에 활용되는 위치 인코딩의 저주파 정렬(low-frequency alignment)이 attention sink 및 retrieval head 같은 구조적 희소 현상의 근본 원인임을 규명했다.**  
• **강력한 외삽 성능을 가진 어텐션 head는 공통적으로 low-rank 구조를 보이며, 이러한 head의 위치 인코딩은 유지하고 high-rank head의 위치 인코딩만 제거하면 학습 없이 최대 10포인트의 성능 향상을 얻을 수 있음을 발견했다.**

### 2 배경

본 절에서는 먼저 Vaswani et al.(2017)의 Transformer 아키텍처를 개괄하며, 특히 멀티-헤드 어텐션 메커니즘에 초점을 맞춘다. 이후 Rotary Position Embedding(RoPE)이 어텐션 패턴을 형성하는 데 어떤 역할을 수행하는지 분석한다.

### 2.1 멀티-헤드 자기어텐션(Multi-Head Self-Attention)

여기서는 디코더 전용 Transformer로 구현된 인과 언어 모델(causal language model)을 고려한다. 토큰 표현

![](/assets/images/posts/604/img_2.png)

### 2.2 Rotary Position Embedding

대부분의 LLM은 기본적인 위치 인코딩 메커니즘으로 **Rotary Position Embedding(RoPE)**(Su et al., 2024)을 채택하며, 이는 현대 아키텍처에서 사실상의 표준(de facto standard)이 되었다. RoPE는 각 쿼리·키 벡터를 일련의 이차원 평면 상에서 회전시키는 방식으로 토큰의 위치를 인코딩한다. 이러한 수식화는 단순한 내적 구조(dot-product structure)를 유지하면서도 어텐션 스코어가 **상대적 위치 차이(relative positional offsets)**에 의존하도록 만들어 준다.

![](/assets/images/posts/604/img_3.png)

![](/assets/images/posts/604/img_4.png)

### 3 관련 연구

**RoPE 기반 길이 외삽(Length Extrapolation With RoPE)**  
RoPE는 매우 널리 채택되어 왔다(Su et al., 2024). LLaMA2(Touvron et al., 2023), LLaMA3(Dubey et al., 2024), Qwen3(Yang et al., 2025)와 같은 모델들은 토큰 순서를 **위치에 따라 각도가 달라지는 벡터 회전**을 통해 인코딩한다. 그뿐 아니라 RoPE 및 그 파생 기법들은 Qwen(Bai et al., 2023), Qwen2(Team, 2024a), Qwen2.5(Team, 2024b), Qwen2.5-VL(Bai et al., 2025), Qwen3(Yang et al., 2025), Mistral(Jiang et al., 2023), Gemma(Team et al., 2024a), Gemma-2(Team et al., 2024b), Gemma-3(Team et al., 2025) 등 최근 공개된 대규모 모델 전반에서 폭넓게 사용되고 있다.

그러나 입력 길이가 학습 시 사용된 길이를 여러 배 초과하게 되면(Peng et al., 2023; Chen et al., 2023; Ding et al., 2024), 모델 성능이 심각하게 저하된다. 이러한 성능 저하는 RoPE만의 문제가 아니며, ALiBi(Press et al., 2021)나 Kerple(Chi et al., 2022) 같은 다른 위치 인코딩에서도 유사한 현상이 관측된다. 이를 해결하기 위해 다양한 접근법이 제안되어 왔다. 예를 들어 FIRE(Li et al., 2023)는 학습 가능한 위치 인코딩을 도입하여 긴 컨텍스트 성능 저하를 완화하는데, MLP를 사용해 적절한 위치 표현을 생성하는 방식이다. 반면 NTK 기반 방법들(Peng et al., 2023)은 **주파수 스펙트럼을 수정**해 컨텍스트 길이를 확장하고 긴 시퀀스에서의 안정성을 높이는 방식으로 길이 외삽을 개선한다.

**위치 인코딩 없이 길이 외삽(Length Extrapolation Without Positional Encoding)**  
위치 인코딩은 시퀀스 정보를 부여하고 모델의 표현 능력을 향상시킨다(Shaw et al., 2018; Yun et al., 2019; Luo et al., 2022). 그러나 여러 연구(Zuo et al., 2024; Haviv et al., 2022; Köcher et al., 2025)는 **인과 디코더 기반 Transformer**가 명시적인 위치 인코딩이 없어도 토큰 순서를 암묵적으로 포착할 수 있음을 보여준다. 더 나아가 NoPE(Kazemnejad et al., 2023)는 **causal mask 자체가 위치 관계를 본질적으로 암호화한다**는 사실을 입증했다.

최근에는 DAPE(Zheng et al., 2024) 및 DAPE v2(Zheng et al., 2024)와 같은 **데이터 의존적(data-dependent) 위치 인코딩 방식**이 등장했다. 이 접근법들은 위치 인코딩을 어텐션의 **입력 의존 feature map**으로 간주하여 길이 외삽 성능을 향상한다. 그러나 이러한 기법들 역시 **위치 인코딩을 구성하기 위한 학습 가능한 파라미터 행렬에 의존한다**는 한계를 가진다.

![](/assets/images/posts/604/img_5.png)

Figure 1:Visualization of DoPE

### 4 Denoising Positional Encoding

#### 4.1 이상치 특성(Outlier Features)

![](/assets/images/posts/604/img_6.png)

![](/assets/images/posts/604/img_7.png)

![](/assets/images/posts/604/img_8.png)

![](/assets/images/posts/604/img_9.png)

![](/assets/images/posts/604/img_10.png)

![](/assets/images/posts/604/img_11.png)

![](/assets/images/posts/604/img_12.png)

![](/assets/images/posts/604/img_13.png)

![](/assets/images/posts/604/img_14.png)

![](/assets/images/posts/604/img_15.png)

![](/assets/images/posts/604/img_16.png)

![](/assets/images/posts/604/img_17.png)

![](/assets/images/posts/604/img_18.png)

![](/assets/images/posts/604/img_19.png)

![](/assets/images/posts/604/img_20.png)

---

## ? 문제 상황: RoPE가 특정 헤드에서 “스파이크”를 만들어 냄

RoPE가 잘못 동작하는 헤드는 어텐션 맵이 이렇게 됨:

```
정상 어텐션
[ . . . . . ]
[ . . . . . ]
[ . . . . . ]
[ . . . . . ]
[ . . . . . ]

스파이크/밴드가 생긴 어텐션
[ . . X . . ]
[ . . X . . ]
[ . . X . . ]
[ . . X . . ]
[ . . X . . ]
```

세로줄/가로줄이 “밝게” 나타나면서  
→ 모델이 특정 위치에 정착해버림  
→ 긴 문맥에서 Attention Sink 발생

## ? DoPE-by-Gaussian이 하는 일

문제 있는 헤드를 **찾아서**, 해당 헤드의 RoPE 위치 인코딩을 **가우시안 랜덤 노이즈로 교체**함.

아스키로 표현하면:

### 원래 RoPE로 회전된 Q/K (문제 헤드)

```
스파이크 방향 (RoPE)
→ → → → →
```

Gaussian 노이즈로 교체

```
등방성(모든 방향 고르게 분포)
↗ ↓ ↙ ↑ ← ↘ →
```

즉, 방향이 한쪽으로 **쏠려 있는 벡터**를  
→ **여러 방향이 섞인 랜덤 벡터**로 바꿔 균형 잡힘

? 계산식의 의미를 직관적으로 바꾸면

```
m_h = 1  → 정상 헤드, 그대로 둠
m_h = 0  → 스파이크 헤드, RoPE를 Gaussian으로 바꾸기
```

구현 형태를 단순한 아스키로 나타내면:

```
if m_h == 1:
    K = RoPE(K)     # 그대로 사용
    Q = RoPE(Q)
else:
    K = Gaussian()  # 치환
    Q = Gaussian()
```

![](/assets/images/posts/604/img_21.png)

직관적으로 그리면:

```
[스파이크]      [DoPE-by-Gaussian 후]
|||||          /\ ↖ ↓ → ↗ ↑ ←
한 방향         여러 방향 → 결집 깨짐
```

## ? 한 줄 요약

> DoPE-by-Gaussian은 스파이크가 생긴 헤드의 RoPE 방향성을 무시하고, **모든 방향으로 고르게 퍼진 랜덤 벡터로 대체하여** 주파수/스펙트럼 균형을 되돌리는 방식이다.

---
