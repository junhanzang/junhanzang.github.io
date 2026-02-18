---
title: "Fast and Simplex: 2-Simplicial Attention in Triton"
date: 2025-07-18 15:58:04
categories:
  - 인공지능
tags:
  - Attention
---

<https://arxiv.org/abs/2507.02754?_bhlid=b823264a61f867fcfc11342f5464010c50351360>

[Fast and Simplex: 2-Simplicial Attention in Triton

Recent work has shown that training loss scales as a power law with both model size and the number of tokens, and that achieving compute-optimal models requires scaling model size and token count together. However, these scaling laws assume an infinite sup

arxiv.org](https://arxiv.org/abs/2507.02754?_bhlid=b823264a61f867fcfc11342f5464010c50351360)

**초록 (Abstract)**

최근 연구에 따르면, 훈련 손실(training loss)은 모델 크기와 토큰 수가 증가함에 따라 거듭제곱 법칙(power law)을 따르며, 계산(compute)에 최적인 모델을 달성하려면 모델 크기와 토큰 수를 함께 확장해야 한다는 사실이 밝혀졌다. 그러나 이러한 스케일링 법칙은 무한한 양의 데이터가 존재한다는 가정 하에 성립하며, 주로 연산량이 제한된(compute-bound) 상황에 적용된다. 현대의 대형 언어 모델들이 점점 더 대규모 인터넷 기반 데이터셋에 의존함에 따라, 이들이 연산 제한 상황에 있다고 가정하는 것은 점차 타당성을 잃고 있다. 이와 같은 변화는 **토큰 효율성(token efficiency)**을 우선시하는 아키텍처의 필요성을 부각시킨다.

본 연구에서는, 표준 점곱(dot-product) 어텐션을 삼중 선형 함수(trilinear function)로 일반화한 아키텍처인 **2-simplicial Transformer**의 사용을 탐구한다. 이 아키텍처는 효율적인 Triton 커널 구현을 통해 실현되었다. 우리는 2-simplicial Transformer가 표준 Transformer보다 더 높은 토큰 효율성을 달성함을 보인다. 즉, 동일한 토큰 예산 하에서 유사한 크기의 모델이 수학, 코딩, 추론, 논리와 같은 과제에서 기존 점곱 기반 모델보다 더 뛰어난 성능을 보인다.

또한 우리는, 2-simplicial 어텐션이 지식과 추론 과제에 대한 스케일링 법칙의 **지수(exponent)**를 변화시킴으로써, 그 성능 이득을 정량적으로 입증한다.

## 1. 서론

Transformer 아키텍처(Vaswani et al., 2017)를 기반으로 한 대형 언어 모델(LLM)은 GPT-3(Brown et al., 2020), GPT-4(Achiam et al., 2023), Gemini(Team et al., 2023), Llama(Touvron et al., 2023) 등 최첨단 인공지능 시스템의 핵심 기반이 되었다. 이러한 모델의 놀라운 발전은 **신경망 스케일링 법칙(neural scaling laws)**(Hestness et al., 2017; Kaplan et al., 2020; Hoffmann et al., 2022)에 의해 이끌어졌는데, 이 법칙은 **훈련 손실(training loss)**과 **모델 파라미터 수**, **훈련 데이터량** 사이의 거듭제곱 법칙(power-law) 관계를 실증적으로 보여준다.

이러한 연구에서 도출된 핵심 통찰은, 단순히 모델 크기만 늘리는 것이 아니라 **모델의 파라미터 수와 훈련 데이터 양을 함께 확장해야 최적의 성능**을 낼 수 있다는 점이다. 특히 Hoffmann et al. (2022)은 **계산 자원에 최적인 모델(compute-optimal model)**을 위해서는 균형 잡힌 확장 전략이 필요하다고 강조한다. 그들의 연구에 따르면, **파라미터 수가 700억 개인 Chinchilla 모델**이 **2800억 파라미터를 가진 Gopher 모델**보다 뛰어난 성능을 보였는데, 이는 Chinchilla가 훨씬 더 많은 데이터로 학습되었기 때문이다. 이 결과는 **대형 언어 모델의 성능 향상을 위해 모델 크기와 함께 데이터 규모를 확장하는 것이 얼마나 중요한지**를 잘 보여준다.

그러나 인공지능(AI)이 계속 발전함에 따라, 이제는 **충분히 고품질인 토큰(token)**을 확보하는 것이 점점 더 어려운 과제가 되고 있다. 이러한 전환점에 가까워지면서, 우리는 **기존 Transformer보다 더 효율적으로 스케일링 가능한 새로운 아키텍처**를 탐색할 필요성이 커지고 있다. 하지만 대부분의 아키텍처 개선이나 옵티마이저의 발전은 단지 손실 곡선의 위치(offset)를 이동시킬 뿐, **스케일링 법칙의 지수(exponent)** 자체를 변화시키지는 못한다(Everett, 2025). Kaplan et al. (2020), Shen et al. (2024)는 대부분의 아키텍처 변경이 지수에 영향을 주지 못함을 보였으며, Hestness et al. (2017)은 옵티마이저에 대해서도 유사한 결론을 도출했다. 현재까지 **스케일링 지수에 긍정적인 변화를 준 유일한 영역은 데이터**인데, Sorscher et al. (2022), Bahri et al. (2024), Brandfonbrener et al. (2024)는 **데이터 분포를 바꾸는 것**이 지수에 영향을 줄 수 있음을 보여주었다.

이러한 맥락에서, 우리는 **Transformer의 점곱(dot-product) 어텐션을 삼중선형 형태(trilinear form)**로 일반화한 Clift et al. (2019)의 오래된 연구를 다시 조명한다. 이 구조는 **2-simplicial Transformer**로 불린다. 우리는 또한 RoPE(Su et al., 2024)를 **삼중선형 함수로 일반화**하는 방식을 탐색하고, **회전 불변(rotation-invariant)**을 가지면서도 **2-simplicial attention만큼 표현력이 뛰어난** 삼중선형 형태를 제시한다.

나아가 우리는 2-simplicial Transformer가 **제한된 토큰 예산 하에서 Transformer보다 더 우수한 확장성을 가진다**는 것을 보인다. 동일한 토큰 수를 사용했을 때, 유사한 크기의 2-simplicial Transformer가 수학, 코딩, 추론 과제에서 기존 Transformer보다 더 나은 성능을 보여준다. 뿐만 아니라, 실험 결과에 따르면 2-simplicial Transformer는 **모델 파라미터 수에 대한 스케일링 지수 역시 더 유리한 방향으로 변화**함을 확인할 수 있었다. 이는 기존의 Chinchilla 스케일링(Hoffmann et al., 2022)과 달리, **2-simplicial Transformer는 파라미터 수를 늘릴 때 토큰 수를 더 느린 속도로 증가시켜도 되는 가능성**을 시사한다.

결론적으로, 우리 연구는 **토큰 수가 제한된 환경**에서 2-simplicial Transformer가 기존의 점곱 기반 Transformer보다 **자연어의 비감축 정보(entropy)에 더 근접하게 도달할 수 있는 효율적인 접근 방식**임을 시사한다.

## 2. 관련 연구

Vaswani et al. (2017)의 기념비적인 연구 이후, **어텐션 메커니즘의 일반화**를 시도한 다양한 연구들이 제안되었다. 가장 초기의 흐름 중 하나는 **시퀀스 길이에 따른 어텐션의 이차 복잡도(quadratic complexity)**를 줄이려는 시도였다. 특히 Parmar et al. (2018)은 이미지 생성 문맥에서 **로컬 어텐션(local attention)**을 제안하였고, 이후 여러 연구들(Zaheer et al., 2020; Roy et al., 2021)은 이를 언어 모델링에 다른 방법들과 결합해 적용하였다.

또 다른 접근으로는 **소프트맥스(softmax) 어텐션을 완전히 제거**하는 방식이 있다. 예를 들어, Katharopoulos et al. (2020)은 소프트맥스를 정규화 없는 지수함수(exponential)로 대체하고, 행렬 곱셈의 결합 법칙을 이용해 **선형 시간 복잡도(linear time)**를 갖는 Transformer를 구현하였다. 또 다른 선형 시간 어텐션 방식으로는 **Mamba**(Gu & Dao, 2023)와 같은 **state space 모델**이 있다. 그러나 이러한 선형 어텐션 기법들은 **실제 성능이 Transformer보다 떨어지는 경우가 많아** 널리 채택되지는 않았다. Allen (2025)에 따르면, Mamba가 실전에서 좋은 성과를 낸 주요 요인은 **Conv1D 연산자(conv1d operator)**의 활용 덕분이며, Transformer 아키텍처의 대안으로 제안된 So et al. (2021), Roy et al. (2022)의 연구도 유사한 방향성을 보인다.

반대편 스펙트럼에서는, 어텐션을 **이차(quadratic)에서 고차(higher-order)**로 확장하려는 시도들이 있다. 이 분야에서 가장 먼저 등장한 것으로 알려진 연구는 Clift et al. (2019)의 **2-simplicial attention**이다. 이들은 이 구조가 **심층 강화학습 환경에서 논리적 문제 해결에 적합한 근사 방식**임을 보였다. 유사한 일반화로 Bergen et al. (2021)은 **Edge Transformer**를 제안했으며, 이들은 **삼각형 어텐션(triangular attention)**을 도입하였다. 또한 AlphaFold 논문(Jumper et al., 2021)에서도 단백질의 2D 기하 구조로부터 유도된 **삼각형 자기어텐션(triangle self-attention)**이 사용되었다.

고차 상호작용은 추천 시스템 환경에서 Wang et al. (2021)에 의해 탐색되었으며, 최근 Sanford et al. (2023)은 **n-레이어 2-simplicial Transformer로 해결 가능한 문제의 집합이 dot-product Transformer로 해결 가능한 집합보다 더 크다**는 것을 보였다. 특히 이들은 **Match3**라는 문제 클래스를 정의하고, dot-product 어텐션은 이 과제를 해결하기 위해 시퀀스 길이에 비례해 **지수적으로 많은 레이어**가 필요함을 보였다. 이어지는 연구로 Kozachinskiy et al. (2025)은 **2-simplicial attention의 확장 가능한 근사 기법**을 제안하고, **Strassen 어텐션**과 dot-product 어텐션 간의 **VC 차원(Vapnik, 1968)에 기반한 하한(lower bound)** 관계를 증명하였다. 이는 더 복잡한 추론이 요구되는 태스크에서의 이론적 우위를 시사한다.

이와 관련된 또 다른 연구로는 Dehghani et al. (2018)이 제안한 **루핑(Looping) Transformer layer**이 있다. 이는 **Universal Transformer**로 구현되었으며, 이후 Yang et al. (2023), Saunshi et al. (2025) 등에서도 유사 개념을 다룬 바 있다. **고차 어텐션과 루핑 구조**는 공통적으로 **단위 파라미터당 더 표현력 있는 함수**를 계산하려는 목적을 갖는다. 이러한 연구들은 루핑 Transformer가 논리적 추론 과제에서 더 나은 성능을 보인다는 점을 보여준다. 다만 루핑 구조의 주요 한계는 **대규모 모델로의 확장 시 학습의 어려움**이다. 구체적으로, **k번 루핑하면 모델의 깊이가 k배 증가**하여 **심층 신경망에서 흔히 발생하는 학습 난점들**을 더욱 심화시킬 수 있다. 따라서 **대형 루핑 Transformer의 학습 가능성**은 여전히 불확실하며, 이를 해결하기 위한 추가 연구가 필요하다.

### 표기법 (Notation)

- **벡터**는 **소문자 볼드체**, **행렬 및 텐서**는 **대문자**, **스칼라**는 일반 소문자로 표기한다.
- 두 벡터 **?, ?** 간의 점곱(dot product)은 ⟨?, ?⟩로, 삼중선형(dot product of three vectors)은  
  ⟨?, ?, ?⟩ = ∑₍ᵢ₌₁₎ᵈ ⟨?ᵢ, ?ᵢ, ?ᵢ⟩ 로 표기한다.
- 행렬 곱셈은 @ 기호로 나타내며, 예: (A B) @ C.
- 배열 슬라이싱은 ?[l:l+m] = (a\_l, ..., a\_{l+m−1})로 나타내며, 인덱스는 0부터 시작한다.
- 텐서 연산 중 일부는 Numpy의 아인슈타인 표기법(Einstein summation notation)을 따른다(Harris et al., 2020).
- **FLOPs**는 부동소수점 연산 수를 의미한다.
- 배열의 열 단위 결합은 [?, ?, ?]로 표기한다.
- 정방행렬의 행렬식(determinant)은 det으로 표기한다.

## 3. 신경망 스케일링 법칙 개요

이 절에서는 **Kaplan et al. (2020)**에서 처음 소개된 **신경망 스케일링 법칙(neural scaling laws)**에 대해 간략히 개요를 제시한다. 우리는 **Hoffmann et al. (2022)**에서 제안한 접근 방식을 따르며, 여기서는 **손실 함수 L(N,D)**가 **모델 파라미터 수 N** 및 **토큰 수 D**에 대해 **거듭제곱 법칙(power law)**에 따라 감소한다고 가정한다:

![](/assets/images/posts/582/img.png)

- 첫 번째 항 E는 **비감축 손실(irreducible loss)**을 의미하며, 이는 자연어 텍스트의 **엔트로피(entropy)**에 해당한다.
- 두 번째 항은 **N** 개의 파라미터를 가진 모델이 이러한 이상적인 생성 과정을 얼마나 따라가지 못하는지를 나타낸다.
- 세 번째 항은 모델이 **유한한 양의 데이터**만을 학습하고 **충분히 수렴(convergence)**되지 않았음을 반영한다.

이론적으로는, N→∞, D→∞일 때, 대형 언어 모델은 텍스트 분포의 **비감축 손실 E**에 점점 수렴해야 한다.

### 최적의 파라미터 및 데이터 크기

특정 **계산 자원 예산 C** 하에서, 총 부동소수점 연산량(FLOPs)을 다음과 같이 표현할 수 있다:

![](/assets/images/posts/582/img_1.png)

이 조건 하에서, **최적의 파라미터 수**와 **최적의 데이터 크기**는 다음과 같은 관계를 따른다:

![](/assets/images/posts/582/img_2.png)

Hoffmann et al. (2022)의 연구는 여러 실험을 통해 손실 함수를 파라메트릭 함수로 근사하여, 지수 a와 b를 추정하였다. 다양한 방법론을 통해 일관되게 다음과 같은 값이 도출되었다:

![](/assets/images/posts/582/img_3.png)

이로부터 Hoffmann et al. (2022)의 핵심 주장—즉, **모델의 크기(model size)에 비례하여 토큰 수도 함께 확장해야 한다**—가 도출된다.

### 품질 좋은 토큰의 한계와 법칙의 구조적 불변성

그러나 앞선 **1절**에서 논의했듯이, 현재는 **충분히 고품질인 토큰의 확보 자체가 병목**이 되고 있다. 이로 인해, 우리는 기존 Transformer 구조 외에 **대안적 아키텍처와 학습 알고리즘**을 탐색할 필요가 커지고 있다.

한편, 최근 여러 연구들은 기존 문헌에서 제안된 대부분의 **모델링 및 최적화 기법**이 실제로는 **손실 함수의 오프셋 E**만 이동시킬 뿐, **거듭제곱 법칙의 지수 (α,β)** 자체는 근본적으로 바꾸지 못한다는 점을 보여주었다. 이와 관련해 Everett (2025)의 심층적인 논의를 참고할 수 있다.

---

대부분은 알겠지만 이는 실험적으로 탄생한 공식이다.
---

## 4. 2-simplicial Transformer

![](/assets/images/posts/582/img_4.png)

**그림 1**: dot-product attention과 2-simplicial attention의 기하적 구조 비교.

**2-simplicial Transformer**는 Clift et al. (2019)에서 처음 제안된 구조로, 기존의 **dot-product attention**을 **쌍선형(bilinear)** 형태에서 **삼선형(trilinear)** 형태로 확장하였다. 이는 기하학적으로 **1-simplex(두 노드 간의 선)**에서 **2-simplex(세 노드가 이루는 삼각형)**으로 일반화한 것과 같다.

![](/assets/images/posts/582/img_5.png)

![](/assets/images/posts/582/img_6.png)

![](/assets/images/posts/582/img_7.png)

![](/assets/images/posts/582/img_8.png)

2-simplicial attention의 forward 연산 과정은 **알고리즘 1**에 의사코드(pseudo-code)로 나타나 있으며, 위의 식 (5)는 RoPE(Su et al., 2024)와 같은 **위치 인코딩(position encoding)**을 포함하지 않는다. 이에 대한 논의는 다음 절에서 다룬다.
---

그러니까 qkv가 원래 qxk 하고 이 값을 v로하니까 이를 단선으로 생각하는데, 2-simplicial은 qkv를 한번에 한다라고 생각하면 되나?

-> ㄴㄴ

2-simplicial Transformer에서 k가 2개가 되는구나. 그러면 이거는 다음걸 가져오는거야 아니면 모든것에 대해서

-> 모든것에 대해서

그러니까 결국은 k의 갯수를 늘린거네? 이를 통해서 백터라이제이션으로 어떻게보면 그 표현형을 늘린거고.

-> yes

그러면 뭐하러? 표현이 더 고차원적인건 알겠는데, 이게 실제 효용성이 그렇게 커보이진 않는데? 어처피 dot프로덕트할거면 차원은 동일하니까 어처피 layer가 늘어는거랑 큰차이 없어보여

-> 현재 학계에서 고민하는것!

그리고 k가 두개면 backporb할때 더 많은 연산이 들어갈텐데, 지금은 transformer용 gpu kernel구조에 맞춰져있는데, 정말 빠른지도 애매할거고
---

### 알고리즘 1: 2-simplicial attention의 forward 패스 의사코드

```
1: 프로시저 2-simplicial_attention(Q, K, V, K′, V′)
2:     
     # 3차원 attention logits 계산 (삼중 내적)
     logits ← einsum("btnh, bsnh, brnh → bntsr", Q, K, K′)
     
3:     
     # 소프트맥스 적용 (마지막 두 차원에 대해), optional: causal mask 포함
     attention ← softmax(logits + causal_mask, axis=[-1, -2])
     
4:     
     # attention weights를 value와 Hadamard product로 가중합
     output ← einsum("bntsr, bsnh, brnh → btnh", attention, V, V′)
     
5:     return output
6: 종료
```

## 5. 행렬식 기반 삼선형(Trilinear) 형태

![](/assets/images/posts/582/img_9.png)

![](/assets/images/posts/582/img_10.png)

![](/assets/images/posts/582/img_11.png)

![](/assets/images/posts/582/img_12.png)

![](/assets/images/posts/582/img_13.png)

![](/assets/images/posts/582/img_14.png)
---

 넓이(parallelogram) → **부피(parallelotope)**로 확장하고, 이 부피를 회전 불변 기하량으로 삼아 RoPE처럼 위치 인코딩이 적용될 수 있게끔 만든 고차 attention 구조
---

![](/assets/images/posts/582/img_15.png)

![](/assets/images/posts/582/img_16.png)

![](/assets/images/posts/582/img_17.png)

![](/assets/images/posts/582/img_18.png)

![](/assets/images/posts/582/img_19.png)

![](/assets/images/posts/582/img_20.png)
---

![](/assets/images/posts/582/img_21.png)

![](/assets/images/posts/582/img_22.png)

![](/assets/images/posts/582/img_23.png)

![](/assets/images/posts/582/img_24.png)

![](/assets/images/posts/582/img_25.png)

![](/assets/images/posts/582/img_26.png)
---

## **7. 커널 최적화 (Kernel Optimization)**

우리는 2-simplicial attention을 위해 설계된 일련의 커널 최적화 기법을 제안합니다. 이 최적화는 Flash Attention (Dao et al., 2022)의 **online softmax** 기반 방식을 토대로 구축되었습니다.

**삼선형(trilinear) 연산**을 위해, 한 입력을 **원소별 곱(elementwise multiplication)**을 통해 병합한 뒤, 이 곱셈 결과에 대해 **2D 타일 기반 행렬곱(matmul)**을 수행합니다. 이는 Figure 2에 시각화되어 있으며, 다음과 같은 연산 구조를 만듭니다:

- CUDA Core에서는  
  Q @ K′ (원소곱)
- Tensor Core에서는  
  (Q @ K′) @ K 및  
  P @ (V @ V′)

이렇게 연산을 분리해 서로 다른 연산 유닛(CUDA Core vs Tensor Core)을 동시에 활용할 수 있도록 설계했습니다.

우리는 이 방식을 **Triton**으로 구현하였고, **520 TFLOPS** 성능을 달성하였으며, 이는 가장 빠른 **FlashAttention v3 Triton 구현체**와 맞먹는 성능입니다. 더 미세한 성능 튜닝을 위해 CUTLASS와 같은 하위 수준 언어를 사용할 수도 있지만, 이미 **긴 시퀀스에 대해 CUTLASS FAv3와 경쟁력 있는 성능**을 달성했습니다 (Figure 3 참조).

![](/assets/images/posts/582/img_27.png)

### **Figure 2 요약:**

- 왼쪽: 슬라이딩 윈도우 기반 2-simplicial attention 시각화. 각 Qᵢ는 [w₁, w₂] 모양의 영역 안의 K, K′를 attend함.
- 오른쪽: 2-simplicial einsum (Q @ K @ K′)을 다음 두 단계로 나눠 타일링
  - Q @ K′는 CUDA Core
  - 그 결과에 @ K는 Tensor Core

![](/assets/images/posts/582/img_28.png)

![](/assets/images/posts/582/img_29.png)

그림 3: FAv3와 2-simplical 주의의 FLOPs 및 지연 시간

### **역전파(Backward) 연산**

역전파에서는 다음과 같은 파생 gradient 연산들이 포함됩니다:

```
(10) dV[j]     = Σ_{i,k} A[i,j,k] ⋅ dO[i] ⋅ V′[k]
(11) dV′[k]    = Σ_{i,j} A[i,j,k] ⋅ dO[i] ⋅ V[j]
(12) dP[i,j,k] = Σ_d dO[i,d] ⋅ V[j,d] ⋅ V′[k,d′]
(13) dS        = softmax_grad_jk(dP)
(14) dK[j]     = Σ_{i,k} Q[i] ⋅ dS[i,j,k] ⋅ K′[k]
(15) dK′[k]    = Σ_{i,j} Q[i] ⋅ dS[i,j,k] ⋅ K[j]
(16) dQ[i]     = Σ_{j,k} dS[i,j,k] ⋅ K[j] ⋅ K′[k]
```

이러한 연산은 **3차원 인덱싱을 가진 텐서**에 대해 계산되므로, **원자 연산(atomic op)**으로 인한 비용이 커질 수 있습니다. 이를 해결하기 위해 다음과 같은 전략을 사용합니다:

### ✅ **두 개의 별도 커널로 분할**

- 커널 1: dK, dV 계산
- 커널 2: dK′, dV′, dQ 계산

이렇게 분할하면 원자 연산 대신 일부 결과(O, dS)를 재계산하는 오버헤드를 감수하고도 성능이 더 좋아집니다.이는 Triton의 coarse-grained 파이프라인 제약에 기인합니다.

### ✅ **작은 w₂에 대한 두 단계 접근법 (Algorithm 2)**

w₂가 작을 때는 다음과 같은 두 단계 접근을 활용하여 atomic 없이 dQ를 계산합니다:

#### ? 알고리즘 2 요약:

1. 시퀀스를 [w₂, dim] 크기의 타일로 분할
2. stage = 0 → 짝수 타일 계산: dQ, dK, dK′, dV, dV′ 저장
3. stage = 1 → 홀수 타일 계산: dQ 누적, dK, dK′, dV, dV′ 더함
4. 마지막에 저장

이렇게 함으로써 atomic 연산을 피하면서 병렬화를 유지할 수 있습니다.

## **8. 실험 및 결과 (Experiments & Results)**

우리는 활성 파라미터가 **10억 개에서 35억 개**, 총 파라미터가 **570억 개에서 1760억 개**에 이르는 다양한 **MoE 모델**(Jordan & Jacobs, 1994; Shazeer et al., 2017)을 학습했습니다. 모든 모델은 **interleaved sliding-window 2-simplicial attention**을 사용하며, **4개 층마다 한 번씩 2-simplicial attention 레이어**를 삽입했습니다. 이렇게 배치한 이유는 **파이프라인 병렬화**(Huang et al., 2019; Narayanan et al., 2019)를 사용할 때, 가장 연산량이 큰 **2-simplicial attention과 global attention이 서로 분산되도록** 하기 위함입니다. 이 두 연산은 FLOPs 측면에서도 비슷한 수준의 연산량을 갖습니다.

### 학습 설정

- 옵티마이저: **AdamW** (Loshchilov et al., 2017)
- 최대 학습률: **4 × 10⁻³**
- weight decay: **0.0125**
- 워밍업 스텝: **4000**
- 학습률 스케줄: **Cosine decay** (최종적으로 최대 학습률의 1%까지 감소)

### 평가 기준

모델은 사전학습(pre-training)에서 수학, 추론, 코딩 능력을 평가하는 다음의 벤치마크에서 **Negative Log-Likelihood(NLL)**로 성능을 평가하였습니다:

- **GSM8k** (Cobbe et al., 2021)
- **MMLU** (Hendrycks et al., 2020)
- **MMLU-pro** (Wang et al., 2024)
- **MBPP** (Austin et al., 2021)

![](/assets/images/posts/582/img_30.png)

### **Scaling 분석**

모델 규모가 **1B → 3.5B**로 커질수록 **NLL 감소량(Δ)이 더 커지는 현상**이 관측됩니다. 즉, **2B 미만의 작은 모델에서는 2-simplicial attention이 이득을 주지 않지만**, 일정 이상으로 모델이 커지면 **성능 향상 효과가 나타납니다**.

Section 3의 scaling 법칙을 다시 정리하면 다음과 같습니다:

#### 이론식:

- 손실 함수는 다음과 같이 표현됩니다:  
  **L(N, D) = E + A·N^(-α) + B·D^(-β)**
- 토큰 수가 동일하므로 D 항은 무시하고:  
  **L(N) = E′ + A·N^(-α)**
- 로그를 취하면:  
  **log L(N) ≈ log E′′ + log A − α·log N**  
  → **-log L(N) = α·log N + β**  
  (여기서 β는 −log E′′ − log A)

따라서, Table 2의 loss 값들을 기반으로 각 모델별 scaling 계수 **α, β**를 추정할 수 있습니다:

![](/assets/images/posts/582/img_31.png)

![](/assets/images/posts/582/img_32.png)

## **9. 논의 (Discussion)**

2-simplicial attention은 **스케일링 법칙(scaling law)**에서 **지수 항(α)**을 향상시키지만, 이 기법이 **토큰 효율(token efficiency)**이 더욱 중요한 **특정 학습 조건**에서 더 유용할 수 있다는 점에는 주의가 필요합니다.

우리가 구현한 **Triton 커널**은 프로토타이핑에는 효율적이지만, 아직 **실제 생산 환경에 투입하기에는 부족한 상태**입니다. 따라서, 앞으로는 **특정 하드웨어 가속기(accelerator)**에 최적화된 **2-simplicial attention 구현을 공동 설계(co-design)**하는 연구가 더 필요합니다.

## **10. 결론 (Conclusion)**

본 연구에서는 Clift et al. (2019)의 **2-simplicial attention**을 Transformer의 **dot product attention**(Vaswani et al., 2017)과 유사한 크기의 모델에 적용하여, **수학, 추론, 코딩 문제에서 negative log-likelihood를 향상시켰음**을 보여주었습니다 (표 2 참고).

또한, 표 3을 통해 **Scaling 법칙의 지수 항(α)**이 개선되었음을 명확하게 수치로 확인했습니다. 특히, **Transformer 대비 reasoning과 coding 태스크에서 α 값이 더 크며**, 이는 **토큰 수가 제한된 상황에서 더 좋은 스케일링 특성**을 유도합니다. 그리고 **MMLU-pro** 및 **GSM8k**처럼 **더 어렵고 덜 포화된 벤치마크**일수록 α 값의 증가율이 더 높다는 점도 관찰되었습니다.

우리는 앞으로 **2-simplicial Transformer의 스케일 확장**이 **추론 중심 태스크의 성능을 대폭 향상시킬 수 있는 가능성**을 열어줄 것으로 기대합니다. 나아가, **전용 커널 및 최적화된 구현체 개발**이 이 아키텍처의 잠재력을 완전히 실현하는 핵심이라고 믿습니다.

## **11. 감사의 글 (Acknowledgments)**

저자들은 다음 분들의 소중한 지원과 피드백에 깊이 감사드립니다:  
**Chuanhao Zhuge, Tony Liu, Ying Zhang, Ajit Mathews, Afroz Mohiuddin, Vinay Rao, Dhruv Choudhary**.
