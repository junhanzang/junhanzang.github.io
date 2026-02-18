---
title: "Attention Is Not What You Need: Grassmann Flows as an Attention-Free Alternative for Sequence Modeling"
date: 2026-02-18 00:00:00
tags:
  - Transformer
  - Attention
  - Grassmann Manifold
  - Sequence Modeling
  - Deep Learning
  - 논문 리뷰
---

> **논문 링크**: [arXiv 2512.19428](https://arxiv.org/abs/2512.19428)
> **저자**: Zhang Chong

---

## Abstract

Self-attention은 현대 시퀀스 모델의 사실상(de facto) 표준 연산이 되었다. 강력한 자연어 성능을 위해서는 dense 또는 approximate attention 메커니즘을 통해 모든 토큰 쌍을 attend해야 한다고 암묵적으로 가정되곤 한다. 본 논문에서는 이 가정에 의문을 제기한다.

저자는 self-attention을 **텐서 리프팅(tensor lifting)**의 한 특수 사례로 재해석한다: 은닉 벡터가 쌍별 상호작용의 고차원 공간으로 매핑되고, 이 리프팅된 텐서에 대한 제약을 통해 경사 하강법으로 학습이 진행된다. 이 리프팅은 매우 표현력이 높지만 수학적으로 추적하기 어렵다; 여러 레이어와 헤드를 거치면서, 소수의 명시적 불변량으로 모델의 행동을 설명하는 것이 거의 불가능해진다. 이러한 관점에서, 대규모 Transformer 모델의 "해석 불가능성(uninterpretability)"의 핵심 원인은 단순히 크기가 아니라, 핵심 연산이 분석적으로 불투명한(analytically opaque) 고차원 텐서 리프트라는 사실에 있다.

대안적 설계로서, 저자는 **Grassmann 흐름(Grassmann flows)** 기반의 attention-free 시퀀스 모델을 제안한다. $L \times L$ attention 행렬을 구성하는 대신, (i) 토큰 상태를 저차원 공간으로 축소하고, (ii) 로컬 토큰 쌍을 Grassmann 다양체 $\mathrm{Gr}(2,r)$ 위의 2차원 부분공간으로 해석하며, (iii) Plücker 좌표를 통해 유한 차원 사영 공간에 임베딩하고, (iv) 게이트 믹싱 블록으로 기하학적 특성을 은닉 상태에 다시 융합한다. 정보는 명시적 쌍별 가중치가 아닌, 레이어와 다중 스케일 로컬 윈도우에 걸친 저랭크 부분공간의 제어된 변형을 통해 시퀀스를 흐른다.

이 Causal Grassmann 아키텍처를 언어 모델링(Wikitext-2)과 자연어 추론(SNLI)에서 평가한다. Wikitext-2에서, 13~18M 파라미터의 순수 Grassmann 기반 언어 모델이 크기 매칭된 Transformer 베이스라인 대비 10~15% 이내의 validation perplexity를 달성한다. SNLI에서, DistilBERT 위에 Grassmann 기반 분류 헤드가 Transformer 헤드를 약간 능가한다(best validation accuracy 0.8550 vs. 0.8545; test accuracy 0.8538 vs. 0.8511). 복잡도 분석은 본 믹싱 메커니즘이 고정 랭크에서 시퀀스 길이에 대해 선형으로 스케일링됨을 보여주며, 이는 full self-attention의 이차 비용과 대비된다.

저자의 목표는 attention이 구식이라고 주장하는 것이 아니라, 그것을 **탈중심화(de-center)**하는 것이다. 근본적으로 필요한 것은 attention 자체가 아니라 은닉 표현을 위한 충분히 표현적인 기하학적 진화 메커니즘이라고 주장한다. Grassmann 흐름은 명시적 attention 가중치 없이도 경쟁적 성능을 얻을 수 있음을 보여주며, 동시에 모델의 핵심을 유한 차원 다양체 위의 흐름이라는 — 명시적 불변량과 기하학적 분석에 더 적합한 — 수학적 설정으로 옮긴다.

---

## 1. Introduction

Transformer[1, 2, 4, 5]는 self-attention을 중심 연산으로 만들어 시퀀스 모델링을 재편했다. 은닉 상태 $H \in \mathbb{R}^{L \times d}$의 시퀀스가 주어지면, self-attention은 쿼리, 키, 밸류를 구성하고:

$$Q = HW_Q, \quad K = HW_K, \quad V = HW_V$$

$L \times L$ 쌍별 호환성 행렬 $QK^\top$을 계산하여 softmax로 정규화한 가중치를 형성하고, 이를 사용하여 밸류를 혼합한다. 이 연산은 표현적이고, 병렬화 가능하며, 너무나 ubiquitous해져서 종종 필수불가결한 것으로 취급된다.

본 논문에서는 다른 입장을 취한다. Attention을 더 싸게, 더 희소하게, 더 확장 가능하게 만드는 방법을 묻는 대신, 더 근본적인 질문을 던진다:

> **명시적 self-attention, 즉 $L \times L$ 가중치 텐서가 정말 강력한 시퀀스 모델링과 추론에 필요한 근본적 요소인가?**

저자의 답은 **"아니오"**이다. Attention은 표현의 기하학적 리프팅을 구현하는 한 가지 특정 방법이지, 유일한 방법이 아니다.

### 1.1 Attention as Tensor Lifting

개념적으로, self-attention은 텐서 리프팅의 한 형태를 수행한다. 위치 $t$에서의 은닉 벡터 $h_t \in \mathbb{R}^d$로부터, 토큰 간 쌍별 관계를 인코딩하는 공간 — 헤드당 $L \times L$ 호환성 텐서와 정규화된 가중치 — 으로 이동한다. 이 리프팅은 몇 가지 핵심 특성을 갖는다:

- **극도로 세밀하다(extremely fine-grained)**: 모든 위치 쌍 $(i,j)$가, 모든 헤드에서, 모든 레이어에서 별도의 학습된 가중치를 받는다.
- **고차원이다(high-dimensional)**: 모델의 유효 상태는 토큰 임베딩뿐만 아니라 레이어에 걸쳐 진화하는 attention 텐서의 구름도 포함한다.
- **압축하기 어렵다(hard to compress)**: 모든 레이어와 헤드에 걸친 attention 행동을 요약하는 명백한 소수의 전역 불변량이 없다.

기하학적 관점에서, self-attention은 시퀀스를 토큰 표현의 다양체에서 쌍별 상호작용의 훨씬 더 큰 공간으로 리프팅하고, 그곳에서 연산을 수행한 후 다시 투영한다. Transformer의 성공은 이 리프팅이 강력하다는 것을 시사하지만; 동시에 왜 해석하기 어려운지에 대한 이유도 시사한다:

> **Claim.** 대규모 모델의 "해석 불가능성"의 주요 원인은 attention이 수행하는 텐서 리프팅이 수학적으로 추적 불가능(non-traceable)하다는 점이다: 각 레이어와 헤드에서 도입되는 자유도가 너무 커서, 모델의 전역적 효과를 설명할 수 있는 작고 명시적인 불변량 집합이 존재하지 않는다.

다시 말해, 모델의 핵심은 간결한 분석적 설명에 저항하는 고차원 텐서 공간에 살고 있다. 개별 attention 맵을 시각화하는 것은 가능하지만, 이를 일관된 전역적 그림으로 집계하는 것은 불가능하다.

### 1.2 Reasoning as Geometry on a Semantic Manifold

대안적 출발점은 추론을 의미 다양체 위의 기하학으로 생각하는 것이다. 높은 수준에서, 다음의 관점을 취한다:

1. 언어 모델의 은닉 상태는 고차원 **의미 다양체(semantic manifold)** 위의 점으로 볼 수 있다. 각 순전파는 이 다양체 위의 경로를 추적한다.
2. **Attention**은 특정 종류의 **텐서 리프팅**을 구현한다: 벡터 $h_t$를 취하고, 다른 위치와의 내적을 통해 그 정보를 쌍별 상호작용의 공간으로 리프팅하여 더 풍부한 로컬 기하학을 발견한다.
3. Transformer의 **효과성**은 후속 신경망 레이어가 이 리프팅된 기하학을 어떻게 **정렬하고 제약하는지**에 달려 있다. 네트워크는 attention의 야생적 자유도를 유용한 행동을 지원하는 흐름으로 제한하는 것을 학습한다.
4. 이 그림에서 **추론(reasoning)**은 의미 다양체의 내재적 기하학적 구조를 반복적으로 샘플링하고 정제하는 과정이다: 표현을 구조화된 방식으로 변형하는 연산자를 레이어 위에 레이어를 적용한다.

이 관점 내에서, 핵심 문제는 텐서 공간으로 리프팅할지 여부가 아니라, 표현의 기하학적 진화 규칙을 어떻게 설계하는가이다. Self-attention은 그러한 규칙 중 하나이지만, 불투명하다: 리프팅된 텐서가 너무 풍부해서 그 장거리 행동이 수학적 용어로 추적하기 극히 어려워진다.

이는 다음의 더 날카로운 철학적 주장을 시사한다:

> **Claim.** 대규모 Transformer의 비해석성은 단순히 깊이나 파라미터 수 때문이 아니다; 그것은 핵심 연산으로 텐서 리프팅을 선택한 것에 근본적 원인이 있다. $L \times L$ attention 텐서로 리프팅하는 순간, 소수의 불변량에 의한 간단하고 명시적이며 전역적인 설명의 가능성을 이미 잃게 된다.

더 수학적으로 구조화된 추론의 관점을 원한다면, 이 불투명한 텐서 리프팅을 더 제어된 기하학적 객체로 대체해야 할 수 있다.

### 1.3 Grassmann Flows as a Controlled Alternative

본 논문은 대안적 설계를 탐구한다: attention 텐서로 리프팅하는 대신, **Grassmann 다양체**로 리프팅한다. 구성은 직관적이다:

- 학습된 선형 맵 $W_{\text{red}}$를 적용하여 $h_t \in \mathbb{R}^d$를 $z_t \in \mathbb{R}^r$ ($r \ll d$)로 축소한다.
- 로컬 윈도우(예: 쌍 $(t, t+\Delta)$)에 대해, $\{z_t, z_{t+\Delta}\}$가 $\mathbb{R}^r$에서 span하는 부분공간을 고려하고, 이를 Grassmann 다양체 $\mathrm{Gr}(2,r)$ 위의 한 점으로 취급한다.
- **Plücker 임베딩**을 사용하여, 각 2차원 부분공간을 $\mathbb{R}^{\binom{r}{2}}$의 좌표 벡터로 매핑하며, 이는 알려진 대수적 관계를 따른다.
- 이 Plücker 좌표는 **기하학적 특성(geometric features)**이 되어 $\mathbb{R}^d$로 다시 투영되고, 학습된 게이팅을 통해 원본 은닉 상태와 융합된다.

Attention과의 핵심 차이는 모델이 명시적이고 유한 차원 구조를 가진 다양체 위에서 작동하도록 제약된다는 것이다:

- 자유도는 $r$과 윈도우 크기의 선택에 의해 제어된다; $L \times L$ 텐서가 없다.
- 로컬 기하학의 표현은 명시적 대수적 제약(Grassmann + Plücker 관계)을 가진 공간에 산다[11, 12, 13, 14].
- 원칙적으로 이 Grassmann 특성의 레이어에 걸친 진화를 다양체 위의 유한 차원 **흐름(flow)**으로 연구할 수 있다.

이 결과 아키텍처를 **Causal Grassmann 모델** 또는 **Grassmann 흐름**이라 부른다. 중요한 점은 이것이 완전히 **attention-free**라는 것이다: attention 행렬을 구성하거나 softmax 정규화 텐서 가중치를 계산하는 단계가 없다.

### 1.4 Contributions

본 논문의 기여는 다음과 같다:

1. **Self-attention에 대한 개념적 비판.** Attention을 텐서 리프팅 메커니즘으로 프레이밍하고, Transformer 비해석성의 주요 원인이 이 리프팅이 수학적으로 추적 불가능하다는 것 — 모델의 핵심 계산이 소수의 불변량이 없는 극도로 고차원적 공간에 살고 있다는 것 — 이라 주장한다.
2. **Grassmann 흐름 기반의 attention-free 아키텍처.** (i) 토큰 상태를 축소하고, (ii) 로컬 쌍을 Plücker 좌표를 통해 $\mathrm{Gr}(2,r)$ 위의 점으로 인코딩하며, (iii) 이 기하학적 특성을 명시적 attention 가중치 없이 표현에 다시 융합하는 Causal Grassmann 믹싱 레이어를 제안한다.
3. **Wikitext-2와 SNLI에서의 실증적 증거.** 13~18M 파라미터의 순수 Grassmann 언어 모델이 Wikitext-2에서 Transformer 베이스라인과 경쟁적이며, Grassmann 기반 NLI 헤드가 SNLI에서 Transformer 헤드를 약간 능가함을 보인다.
4. **복잡도 및 해석 가능성 분석.** Grassmann 믹싱의 점근적 복잡도를 분석하고, 고정 랭크와 윈도우 크기에서 시퀀스 길이에 대해 선형으로 스케일링됨을 보인다. 또한 Grassmann 다양체 위에서 작동하면 모델의 행동에 대한 전역 불변량을 정의하는 것이 더 현실적이라 주장한다.

목표는 attention을 모든 곳에서 대체하는 것이 아니라, 설계 공간의 다른 영역을 열어주는 것이다: 핵심 시퀀스 상호작용이 불투명한 텐서 리프팅이 아닌 명시적 기하학적 흐름과 다양체에 의해 지배되는 아키텍처.

---

## 2. Attention as Geometric Lifting and Grassmann Background

이 섹션에서는 attention에 대한 기하학적 관점을 더 정밀하게 만들고, 모델의 기반이 되는 Grassmann 구조를 소개한다.

### 2.1 Self-Attention as Geometric Lifting

은닉 상태 $H \in \mathbb{R}^{L \times d}$의 시퀀스가 주어지면, 표준 multi-head self-attention 레이어는 각 헤드 $h$에 대해 다음을 계산한다:

$$Q_h = HW_Q^{(h)}, \quad K_h = HW_K^{(h)}, \quad V_h = HW_V^{(h)}$$

$W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d \times d_h}$이며 일반적으로 $d_h = d/H_{\text{heads}}$이다.

각 헤드에 대해 다음을 계산한다:

$$A_h = \operatorname{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{L \times L}$$

그리고 헤드 출력을 얻는다:

$$O_h = A_h V_h \in \mathbb{R}^{L \times d_h}$$

출력 $O_h$는 연결(concatenate)되어 $\mathbb{R}^d$로 선형 투영된다.

이 과정은 다음과 같이 해석할 수 있다:

- 선형 맵 $W_Q^{(h)}, W_K^{(h)}$는 시퀀스를 내적이 위치 간 호환성을 인코딩하는 표현으로 임베딩한다.
- 행렬 $Q_h K_h^\top$은 $H$에서 쌍별 상호작용의 공간으로의 **리프트(lift)**이다: 각 토큰은 벡터만으로가 아니라 다른 모든 토큰과의 유사성으로도 표현된다.
- Softmax와 $V_h$에 의한 곱은 이 리프팅된 구조에 대한 특정 기하학적 연산을 구현하며, 이 호환성에 따라 정보를 재분배한다.

기하학적으로, 모델은 단순히 은닉 상태의 다양체를 따라 이동하는 것이 아니라, 시퀀스 길이에 대해 이차적으로 증가하는 차원의 쌍별 관계의 구름도 수정한다. 여러 헤드와 레이어에 걸쳐, 이 구름은 극도로 복잡해진다. 개별 attention 맵을 시각화할 수 있지만, 모델의 전역적 행동은 이러한 리프트의 합성에 의해 지배되어 간결한 불변량으로 요약하기 어렵다.

### 2.2 Grassmann Manifolds and Plücker Coordinates

**Grassmann 다양체** $\mathrm{Gr}(k,r)$은 $\mathbb{R}^r$의 모든 $k$차원 선형 부분공간의 집합이다. 이것은 차원 $k(r-k)$의 매끄러운 다양체이다. 본 논문에서는 $k=2$에 초점을 맞추므로, $\mathrm{Gr}(2,r)$은 $\mathbb{R}^r$의 모든 2차원 부분공간을 차원 $2(r-2)$로 매개변수화한다.

Grassmann 다양체 위의 점을 표현하는 여러 표준 방법이 있다. 저자는 각 $k$차원 부분공간을 사영 공간의 한 점으로 매핑하는 **Plücker 임베딩**을 사용한다:

- $k$차원 부분공간 $U \subset \mathbb{R}^r$의 기저 $(u_1, \dots, u_k)$가 주어지면, $k$번째 외적 거듭제곱(exterior power) $\Lambda^k \mathbb{R}^r$에서 외적(exterior product) $u_1 \wedge \cdots \wedge u_k$를 형성한다.
- 좌표로, $u_1 \wedge \cdots \wedge u_k$는 행렬 $[u_1 \dots u_k]$의 $k \times k$ 소행렬(minors)인 원소를 가진 $\mathbb{R}^{\binom{r}{k}}$의 벡터로 표현할 수 있다.
- $k=2$의 경우, 이것은 특히 간단하다: $u, v \in \mathbb{R}^r$이 2차원 부분공간을 span하면, $u \wedge v$는 모든 쌍별 행렬식으로 주어진다:

$$p_{ij} = u_i v_j - u_j v_i, \quad 1 \leq i < j \leq r$$

이것은 벡터 $p \in \mathbb{R}^{\binom{r}{2}}$를 형성한다.

이 임베딩 하에서 $\mathrm{Gr}(2,r)$의 상(image)은 $\mathbb{R}^{\binom{r}{2}}$ 전체가 아니라; 이차 Plücker 관계에 의해 정의된 대수적 다양체(algebraic variety)이다. 그럼에도, 본 논문의 목적을 위해 $p$를 $u$와 $v$가 span하는 부분공간을 인코딩하는 정규화된 특성 벡터로 간주할 수 있다. 같은 부분공간을 span하는 다른 기저는 비례하는 Plücker 벡터를 산출하며, 이는 임베딩의 사영적(projective) 성질을 반영한다.

이 표현을 사용하여 선형 차원 축소 후 토큰 벡터 쌍의 로컬 기하학을 인코딩한다.

### 2.3 Why Grassmann for Sequence Modeling?

왜 Grassmann 다양체를 믹싱 규칙의 backbone으로 선택하는가? 다른 다양체나 더 ad hoc한 특성 변환 대신? 몇 가지 이유가 있다.

**로컬 선형 구조(Local linear structure).** 매끄러운 다양체 위에서, 로컬 기하학은 접선 공간과 그 부분공간으로 포착될 수 있다. Grassmann 다양체는 선형 부분공간의 집합을 자연스럽게 매개변수화하므로, 더 복잡한 구조의 로컬 선형 근사를 표현하기에 적합하다. 축소된 은닉 상태의 쌍을 취하고 그 span을 형성할 때, 우리는 효과적으로 의미 다양체에서의 로컬 방향과 평면을 인코딩하고 있다.

**유한 차원 대수적 구조(Finite-dimensional algebraic structure).** Grassmann 다양체는 유한 차원이며 Plücker 임베딩을 통해 사영 공간 내에 위치한다. 이는 기하학적 정보를 알려진 대수적 제약을 따르는 고정 차원 특성 벡터로 인코딩할 수 있음을 의미한다. 신경망은 이 특성에 대해 연산할 수 있으면서, 기반 객체는 명확한 기하학적 의미를 가진 부분공간으로 남는다.

**근사 정리와의 호환성(Compatibility with approximation theorems).** 실해석학의 관점에서, 의미 공간을 다양체 $M$으로, 모델을 연산자 $\Phi: M \to M$으로 이상화할 수 있다. 고전적 보편 근사 정리는 충분한 용량이 주어지면 신경망이 그러한 연산자를 근사할 수 있다고 말하지만, 특정 기하학을 규정하지는 않는다. 우리의 믹싱 규칙이 로컬 이웃을 먼저 $\mathrm{Gr}(2,r)$로 인코딩한 다음 그 다양체 위에서 작동하도록 제약함으로써, 모델의 동역학이 제어된 자유도를 가진 구조화된 다양체를 통해 팩터링되도록 요구하는 것이다. 보편 근사는 여전히 적용되지만, 근사가 이제 구조를 분석할 수 있는 공간 위에서 전개된다.

종합하면, 이 특성들은 비구조화 텐서 리프팅을 더 제어되고, 해석 가능한, 기하학적 원시 연산으로 대체하고자 할 때 Grassmann 다양체를 자연스러운 선택으로 만든다.

---

## 3. Methods: A Causal Grassmann Transformer without Attention

아키텍처를 상세히 설명한다. 설계는 Transformer 인코더의 넓은 윤곽을 따르지만, 각 self-attention 블록을 다음을 수행하는 Causal Grassmann 믹싱 블록으로 대체한다:

1. 은닉 상태를 저차원 공간으로 축소하고,
2. 로컬 쌍을 구성하여 $\mathbb{R}^{\binom{r}{2}}$의 Plücker 벡터로 인코딩하며,
3. 이 기하학적 특성을 게이팅과 피드포워드 네트워크를 통해 원본 은닉 상태에 다시 혼합한다.

### 3.1 Token and Positional Embeddings

어휘 크기 $V$에 대한 표준 next-token LM 설정에서, 토큰 시퀀스 $(x_1, \dots, x_L)$을 학습된 임베딩 행렬 $E \in \mathbb{R}^{V \times d}$와 위치 임베딩 $P \in \mathbb{R}^{L_{\max} \times d}$를 사용하여 $\mathbb{R}^d$에 임베딩한다:

$$h_t^{(0)} = E(x_t) + P_t, \quad t = 1, \dots, L$$

실험 전반에 걸쳐 $d = 256$을 사용한다. 결과 시퀀스 $H^{(0)} = (h_1^{(0)}, \dots, h_L^{(0)})$는 $N$개의 쌓인 Causal Grassmann Transformer 레이어를 통과한다.

### 3.2 Causal Grassmann Mixing Layer

각 레이어는 $H \in \mathbb{R}^{L \times d}$를 입력으로 받아 업데이트된 시퀀스 $\tilde{H} \in \mathbb{R}^{L \times d}$를 출력한다. 레이어 내의 핵심 연산은 다음과 같다:

#### Step 1: Linear Reduction (선형 축소)

먼저 각 은닉 상태를 저차원 벡터로 축소한다:

$$z_t = W_{\text{red}} h_t + b_{\text{red}}, \quad W_{\text{red}} \in \mathbb{R}^{r \times d},\ b_{\text{red}} \in \mathbb{R}^r$$

일반적으로 $r \ll d$; 실험에서 $r = 32$를 사용한다. 이것은 $Z = (z_1, \dots, z_L) \in \mathbb{R}^{L \times r}$을 생성한다.

#### Step 2: Multi-Scale Local Pairing (다중 스케일 로컬 페어링)

윈도우 크기(오프셋) 집합 $\mathcal{W} = \{\Delta_1, \dots, \Delta_m\}$을 정의한다, 예를 들어:

$$\mathcal{W} = \{1, 2, 4, 8, 12, 16\}$$

또는 더 깊은 모델의 경우 $(1,1,2,2,4,4,8,8,12,12,16,16)$ 같은 다층 스케줄. 각 위치 $t$와 오프셋 $\Delta \in \mathcal{W}$에 대해 $t + \Delta \leq L$이면 쌍 $(z_t, z_{t+\Delta})$를 형성한다.

주어진 $t$에 대해, 이것은 최대 $m$개의 쌍을 생성한다:

$$(z_t, z_{t+\Delta_1}),\ (z_t, z_{t+\Delta_2}),\ \dots$$

이것들을 다중 스케일의 로컬 이웃으로 취급한다. 페어링은 **인과적(causal)**이다: $t$를 엄밀히 오른쪽(미래) 위치와만 쌍을 이루어 left-to-right 언어 모델링과 일관성을 유지한다.

#### Step 3: Grassmann / Plücker Encoding

각 쌍 $(z_t, z_{t+\Delta})$에 대해, $\mathbb{R}^r$에서 이 벡터들이 span하는 2차원 부분공간을 고려한다. Plücker 벡터 $p_t^{(\Delta)} \in \mathbb{R}^{\binom{r}{2}}$를 형성한다:

$$p_{ij}^{(\Delta)}(t) = z_{t,i} \cdot z_{t+\Delta,j} - z_{t,j} \cdot z_{t+\Delta,i}, \quad 1 \leq i < j \leq r$$

수치 안정성을 위해 선택적 정규화를 적용한다:

$$\hat{p}_t^{(\Delta)} = \frac{p_t^{(\Delta)}}{\max(\|p_t^{(\Delta)}\|_2, \varepsilon)}$$

이것은 각 $t$와 $\Delta$에 대한 Plücker 특성 집합을 산출한다.

#### Step 4: Projection Back to Model Space (모델 공간으로 역투영)

학습된 선형 맵을 통해 Grassmann 특성을 모델 차원으로 역투영한다:

$$g_t^{(\Delta)} = W_{\text{plü}} \hat{p}_t^{(\Delta)} + b_{\text{plü}}, \quad W_{\text{plü}} \in \mathbb{R}^{d \times \binom{r}{2}}$$

오프셋에 걸쳐 합산 또는 평균으로 집계한다:

$$g_t = \frac{1}{|\mathcal{W}_t|} \sum_{\Delta \in \mathcal{W}_t} g_t^{(\Delta)}$$

여기서 $\mathcal{W}_t = \{\Delta \in \mathcal{W} : t + \Delta \leq L\}$은 위치 $t$에서의 유효 오프셋 집합이다.

벡터 $g_t \in \mathbb{R}^d$는 위치 $t$ 주변의 다중 스케일 로컬 Grassmann 기하학을 포착한다.

#### Step 5: Gated Fusion (게이트 융합)

원본 은닉 상태와 Grassmann 특성을 연결하여 게이트를 계산한다:

$$u_t = [h_t; g_t] \in \mathbb{R}^{2d}$$

$$\alpha_t = \sigma(W_{\text{gate}} u_t + b_{\text{gate}}), \quad W_{\text{gate}} \in \mathbb{R}^{d \times 2d}$$

혼합 표현은:

$$\tilde{h}_t^{\text{mix}} = \alpha_t \odot h_t + (1 - \alpha_t) \odot g_t$$

이후 layer normalization과 dropout이 따른다:

$$\hat{h}_t = \text{LayerNorm}(\tilde{h}_t^{\text{mix}}); \quad \hat{h}_t = \text{Dropout}(\hat{h}_t)$$

#### Step 6: Feed-Forward Block

표준 Transformer와 같이, position-wise 피드포워드 네트워크를 적용한다:

$$\phi_t = W_2 \sigma(W_1 \hat{h}_t + b_1) + b_2$$

$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $d_{\text{ff}} = 4d$이며, 비선형성 $\sigma$로 GELU를 사용한다. 또 다른 잔차 연결과 layer normalization이 레이어를 완성한다:

$$h_t' = \text{LayerNorm}(\hat{h}_t + \phi_t)$$

$N$개의 이런 레이어를 쌓으면 전체 Causal Grassmann Transformer가 된다.

### 3.3 Comparison to Self-Attention

시퀀스 길이 $L$과 은닉 차원 $d$에서, 표준 multi-head self-attention 레이어의 시간 복잡도는:

$$\mathcal{O}(Ld^2 + L^2 d_{\text{head}})$$

$d_{\text{head}}$는 헤드당 차원이다. 첫 번째 항은 $Q, K, V$ 계산에서, 두 번째 항은 행렬 곱 $QK^\top$ (크기 $L^2$)과 후속하는 $L \times L$ attention 행렬과 $V$의 곱에서 발생한다.

Grassmann 믹싱 레이어의 주요 비용은:

- **선형 축소**: $HW_{\text{red}}^\top$은 $\mathcal{O}(Ldr)$이 소요된다.
- **Plücker 계산**: 각 위치와 오프셋에 대해 $p_t^{(\Delta)}$를 형성하는 데 $\mathcal{O}(r^2)$이 소요된다. $m = |\mathcal{W}|$ 오프셋으로, $\mathcal{O}(Lmr^2)$이 기여한다.
- **모델 공간으로의 투영**: $W_{\text{plü}} \hat{p}_t^{(\Delta)}$는 쌍당 $\mathcal{O}(d\binom{r}{2})$이 소요되어, $\mathcal{O}(Lmdr^2)$를 준다.
- **게이팅과 피드포워드**: 둘 다 표준 Transformer와 같이 $\mathcal{O}(Ld^2)$이다.

$r$과 $m$을 고정 하이퍼파라미터로 취급하면(실험에서 그러하며), $r \ll d$이고 $r^2$이 적당하므로, Plücker와 투영 비용은 $\mathcal{O}(Ld^2)$ 항에 흡수될 수 있다. 결정적으로, **$L^2$ 항이 없다**: 고정 $r$과 $m$에서 복잡도는 $L$에 대해 선형이다:

$$\text{Causal Grassmann: } \mathcal{O}(Ld^2) \quad \text{vs.} \quad \text{Self-attention: } \mathcal{O}(L^2 d_{\text{head}} + Ld^2)$$

실제로, 현재 구현은 적당한 $L$에서 고도로 최적화된 GPU attention 커널보다 스텝당 느리다 — Plücker 좌표 계산과 reshape 처리의 오버헤드 때문이다. 그러나, $L$에 대해 점근적으로, 그리고 추가 엔지니어링을 통해, Grassmann 레이어는 원칙적으로 더 확장 가능할 수 있다.

---

## 4. Experimental Setup

제안된 Causal Grassmann 아키텍처를 두 가지 표준 NLP 벤치마크에서 평가한다: 언어 모델링을 위한 Wikitext-2와 자연어 추론을 위한 SNLI[6, 9].

### 4.1 Wikitext-2 Language Modeling

**데이터와 토크나이제이션.** Wikitext-2-raw 데이터셋을 사용한다. 시퀀스는 고정 길이 $L$ (블록 크기)의 연속 텍스트 청크로 형성된다. 주요 실험에서 $L = 128$과 $L = 256$을 고려한다. BERT 스타일 토크나이저에 맞춰 크기 $V \approx 30{,}522$의 WordPiece 유사 어휘를 사용한다.

**모델.** 다음을 비교한다:

- **TransformerLM**: $N$개 레이어, 모델 차원 $d = 256$, 피드포워드 차원 $d_{\text{ff}} = 1024$, 4헤드 multi-head self-attention을 가진 표준 decoder-only Transformer.
- **GrassmannLM**: 동일한 backbone (임베딩, 레이어 수, $d$, $d_{\text{ff}}$)이지만, 각 self-attention 블록을 위에서 설명한 Causal Grassmann 믹싱 블록으로 대체.

두 가지 레이어 깊이를 탐구한다:

- **Shallow**: $N = 6$ 레이어; GrassmannLM은 ~13.0M 파라미터, TransformerLM은 ~12.6M.
- **Deeper**: $N = 12$ 레이어; GrassmannLM은 ~18.2M 파라미터, TransformerLM은 ~17.3M.

GrassmannLM에서 축소 차원 $r = 32$를 설정하고, 6-레이어 모델에는 다중 스케일 윈도우 $\mathcal{W} = \{1, 2, 4, 8, 12, 16\}$을, 12-레이어 모델에는 깊이에 걸쳐 반복 패턴 $(1,1,2,2,4,4,8,8,12,12,16,16)$을 사용한다.

**학습.** 두 모델을 동일한 옵티마이저와 학습률 스케줄로 공유 스크립트에서 학습하며, 믹싱 블록의 선택만 다르다. 모든 모델을 30 에포크 학습하고, 학습 중 최고 validation perplexity를 보고한다. 배치 크기는 $L = 128$일 때 32, $L = 256$일 때 16이다.

### 4.2 SNLI Natural Language Inference

**데이터.** 함의(entailment), 모순(contradiction), 또는 중립(neutral)으로 레이블된 문장 쌍으로 구성된 SNLI 데이터셋을 사용한다. 표준 train/validation/test 분할을 따른다.

**Backbone.** 공정한 비교를 위해, DistilBERT-base-uncased backbone을 특성 추출기로 고정한다. Backbone은 문장당 최대 시퀀스 길이 48 토큰(토크나이제이션 및 절단 후)까지 문맥화된 토큰 임베딩을 생성한다. 이후 풀링을 적용하여 문장 수준 표현을 얻는다.

**분류 헤드.** DistilBERT backbone 위에 다음을 비교한다:

- **Transformer 헤드**: 풀링된 특성에 대한 self-attention과 3-way 분류를 위한 최종 선형 레이어를 가진 2-레이어 Transformer 스타일 분류기.
- **Grassmann–Plücker 헤드**: 제안된 Grassmann 기반 헤드(GrassmannPluckerNLIModel)로, 투영된 특성에 대해 다중 스케일 윈도우를 통한 Grassmann 믹싱 모듈을 적용한 다음 피드포워드 분류기를 사용한다.

Grassmann 헤드의 하이퍼파라미터: 축소 차원 $d_{\text{proj}} = 64$, 토큰 시퀀스에 대한 윈도우 크기 8과 stride 8, $d_{\text{model}} = 256$, 2 믹싱 레이어, 4 믹싱 헤드(쌍 그룹핑용), $d_{\text{ff}} = 512$, dropout 0.1. 두 헤드 모두 비교 가능한 파라미터 수를 가지며, backbone의 동일한 초기화에서 20 에포크 학습한다.

**메트릭.** Validation과 test 세트에서의 분류 정확도를 학습 손실 곡선과 함께 보고한다.

---

## 5. Results

### 5.1 Wikitext-2 Language Modeling

Table 1과 2는 주요 언어 모델링 결과를 요약한다. 파라미터 수와 30 에포크에 걸친 최고 validation perplexity를 보고한다.

**Table 1**: Wikitext-2: 블록 크기 128의 6-레이어 모델과 두 가지 다중 스케일 윈도우 스케줄. GrassmannLM은 TransformerLM보다 validation perplexity에서 약 10~15% 뒤처지지만 동일한 전반적 영역에 남아있다.

| Model | Layers | Params (M) | Val PPL |
|---|---|---|---|
| TransformerLM (block size 128) | 6 | 12.59 | 248.4 |
| GrassmannLM (block size 128) | 6 | 13.00 | 275.7 |
| TransformerLM (block size 128) | 6 | 12.59 | 253.6 |
| GrassmannLM (block size 128) | 6 | 13.00 | 282.3 |

블록 크기 128과 다중 스케일 윈도우 $\mathcal{W} = \{1,2,4,8,12,16\}$의 6-레이어 모델에서, GrassmannLM은 매칭된 학습 조건 하에서 TransformerLM의 241.0~253.6에 비해 약 275.7의 최고 validation perplexity를 달성한다. 약간 다른 윈도우 스케줄(예: $\{1,2,4,8,8,8\}$)에서도 유사한 격차를 보인다: GrassmannLM 282.3 vs. TransformerLM 248.4.

**Table 2**: Wikitext-2: 블록 크기 256의 12-레이어 모델과 깊이에 걸친 반복 다중 스케일 윈도우 $(1,1,2,2,4,4,8,8,12,12,16,16)$. GrassmannLM은 다시 TransformerLM의 대략 10% 이내이다.

| Model | Layers | Params (M) | Val PPL |
|---|---|---|---|
| TransformerLM (block size 256) | 12 | 17.32 | 235.2 |
| GrassmannLM (block size 256) | 12 | 18.16 | 261.1 |

더 깊은 12-레이어 모델에서, GrassmannLM은 261.1의 최고 validation perplexity에 도달하고, TransformerLM은 235.2에 도달한다. 상대적 격차가 6-레이어 설정보다 작아, 추가 깊이가 Grassmann 모델의 더 국소적인 믹싱을 보상하는 데 도움이 됨을 시사한다.

전체적으로, 이 설정들에 걸쳐:

- GrassmannLM은 attention을 사용하지 않음에도, 크기 매칭된 TransformerLM의 validation perplexity 대비 **일관되게 10~15% 이내**이다.
- 깊이가 증가함에 따라 **격차가 좁아지는 것**으로 보이며, 이는 반복적 로컬 Grassmann 믹싱이 더 풍부한 상호작용을 근사할 수 있다는 관점과 일관된다.
- 파라미터 수는 비교 가능하게 유지된다: GrassmannLM은 Plücker 투영과 게이팅 레이어로 인해 약간 더 많은 파라미터를 갖지만, 차이는 ~3~5% 수준이다.

이 결과들은 최첨단 언어 모델과 경쟁하려는 것이 아니라, Grassmann 흐름을 통한 "attention-free" 시퀀스 모델링이 중간 규모에서 실행 가능함을 보여주기 위한 것이다.

### 5.2 SNLI Natural Language Inference

Table 3은 SNLI 결과를 요약한다. 두 모델이 동일한 DistilBERT backbone을 공유하며 분류 헤드만 다르다는 것을 상기하라.

**Table 3**: DistilBERT backbone을 사용한 SNLI 분류 정확도. Grassmann–Plücker 헤드가 validation과 test 세트 모두에서 Transformer 헤드를 약간 능가한다.

| Head Type | Val Accuracy | Test Accuracy |
|---|---|---|
| Transformer head | 0.8545 | 0.8511 |
| Grassmann–Plücker head | **0.8550** | **0.8538** |

Grassmann 헤드는 best validation accuracy 0.8550과 test accuracy 0.8538을 달성하여, 0.8545 validation과 0.8511 test accuracy에 도달하는 Transformer 헤드를 약간 능가한다. 학습 곡선은 유사한 수렴 속도를 보여주며, Grassmann 헤드는 학습 후반에 약간 더 낮은 validation loss를 보인다.

마진은 작지만, 이 결과는 개념적으로 중요하다:

- 다운스트림 추론 과제에서, 분류 헤드에 명시적 기하학적 구조를 주입하면 backbone이 고정되어 있을 때에도 Transformer 헤드와 동등하거나 약간 초과할 수 있음을 보여준다.
- Grassmann 메커니즘이 단순한 이론적 호기심이 아니라 실용적 설정에서 성능에 긍정적으로 기여할 수 있음을 나타낸다.

### 5.3 Complexity and Empirical Runtime

앞서 논의한 대로, Causal Grassmann 레이어의 점근적 복잡도는 고정 축소 차원 $r$과 윈도우 수 $m$에서 시퀀스 길이 $L$에 대해 선형인 반면, self-attention은 $L \times L$ attention 행렬로 인해 $L$에 대해 이차적으로 스케일링된다.

그러나 현재 구현에서, 순수 Grassmann 모델의 경험적 스텝당 런타임은 시퀀스 길이 256까지에서 Transformer 베이스라인보다 느리다. 이는 예상된 것이다:

- GPU 라이브러리는 dense 행렬 곱과 attention 메커니즘에 고도로 최적화된 커널을 제공한다.
- 우리의 Plücker 계산은 아직 저수준 커널 퓨전이나 커스텀 CUDA 구현을 활용하지 않는 명시적 element-wise 연산과 reshape을 포함한다.

따라서 여기 보고된 실험은 아키텍처와 그 복잡도 프로파일에 대한 **개념 증명(proof of concept)**으로 해석되어야 하며, 최적화된 엔지니어링 솔루션이 아니다. Grassmann 연산을 퓨즈하고 $\mathrm{Gr}(2,r)$의 구조를 활용하는 전용 구현이 실제로 잠재적인 선형 스케일링 이점을 완전히 실현하기 위해 필요할 것이다.

---

## 6. Discussion

### 6.1 What Does Grassmann Mixing Actually Buy Us?

실험은 순수 기하학적, 국소성 기반 믹싱 규칙이 명시적 self-attention에 의존하지 않고도 비자명한 언어 모델링과 자연어 추론을 지원할 수 있음을 보여준다. 비교적 작은 모델과 적당한 컨텍스트 길이에서, 제안된 Causal Grassmann 아키텍처는:

- Wikitext-2에서 크기 매칭된 Transformer와 경쟁적이며,
- DistilBERT 기반 NLI 모델로 사용될 때 SNLI에서 Transformer 분류 헤드를 약간 능가한다.

엔지니어링 관점에서, 이 규모에서 역량 있는 시퀀스 모델링에 attention이 엄밀히 필요하지 않음을 보여준다. 개념적 관점에서, 더 미묘한 주장을 지지한다:

> **Claim.** 모델이 기하학적으로 충분히 풍부한 로컬 진화 규칙을 갖추기만 하면, 명시적 attention 가중치 없이도 의미론적 추론이 나타날 수 있다.

Self-attention은 학습된 $L \times L$ 가중치 행렬을 통해 각 토큰이 다른 모든 토큰을 보게 한다. Grassmann 믹싱은 이와 대조적으로 로컬 부분공간 업데이트의 시퀀스를 구성한다: 정보는 다중 스케일 윈도우를 통해 저랭크 부분공간을 회전하고 구부림으로써 흐른다. 두 메커니즘 모두 레이어에 걸쳐 고차 기하학적 구조를 축적하지만, 다른 원시 연산으로:

- **Self-attention**은 텐서 리프팅과 전역 쌍별 상호작용을 사용한다;
- **Grassmann mixing**은 저랭크 부분공간과 로컬 윈도우를 다양체 위의 제어된 흐름으로 사용한다.

현재 규모에서, Grassmann 모델은 언어 모델링에서 Transformer를 능가하지 못한다; 약간 뒤처져 있다. 이는 설계의 단순성과 광범위한 하이퍼파라미터 튜닝의 부재를 감안하면 놀랍지 않다. 그럼에도, SNLI 결과는 backbone이 고정되고 헤드에 집중할 때 명시적 기하학을 추가하면 측정 가능한 이득을 얻을 수 있음을 보여준다. 이는 기하학적 관점이 철학적으로 매력적일 뿐만 아니라 실용적으로도 유용함을 시사한다.

### 6.2 Interpretability: From Tensor Lifting to Finite-Dimensional Flows

서론에서 Transformer 비해석성의 핵심 이유가 텐서 리프팅으로서의 attention의 성질이라 주장했다. 각 레이어는 표현을 쌍별 상호작용의 고차원 공간으로 리프팅한다; 전체 모델은 이러한 리프트의 합성이다. 각 개별 attention 맵은 볼 수 있지만, 전역적 행동은 소수의 불변량으로 요약하기 어렵다.

이와 대조적으로, Grassmann 아키텍처는 관련 자유도를 유한 차원의 수학적으로 엄밀한 다양체로 의도적으로 압축한다:

- 축소된 상태 $z_t \in \mathbb{R}^r$은 저차원 공간에서의 로컬 방향을 포착한다.
- 쌍 $(z_t, z_{t+\Delta})$은 $\mathrm{Gr}(2,r)$ 위의 점을 정의한다; 이 점들은 고정 차원 $\binom{r}{2}$의 Plücker 벡터로 인코딩된다.
- 믹싱 과정은 $L \times L$ 텐서의 임의적 조작이 아닌, 이 저랭크 부분공간의 로컬 변형으로 제약된다.

이것은 더 희망적인 해석 가능성 이야기를 시사한다. 학습 후, Plücker 벡터나 다른 Grassmann 기술자를 후보 설명 불변량으로 취급할 수 있다:

- 이들은 수가 유한하고 명시적 대수적 관계를 따른다.
- 레이어에 걸쳐 비교 가능하다.
- 미분 기하학과 대수 기하학의 도구로 연구할 수 있다.

이것이 해석 가능성을 사소하게 만드는 것은 아니다. 그러나 전역 불변량을 정의하고 계산할 현실적 전망이 있는 영역으로 모델의 핵심을 옮긴다. 진화하는 attention 텐서 컬렉션을 요약하려 하는 대신, 다양체 $\mathrm{Gr}(2,r)$ 위의 진화하는 궤적을 요약하려 시도할 수 있다.

### 6.3 Why Grassmann? A Link to Approximation Theorems

근사 이론의 관점에서, 시퀀스 모델을 의미 다양체 $M \subset \mathbb{R}^d$ 위의 연산자 $\Phi$를 근사하는 것으로 이상화할 수 있다. 보편 근사 정리는 가벼운 조건 하에서 신경망이 그러한 연산자를 임의로 잘 근사할 수 있음을 보장한다.

그러나 그 정리들은 아키텍처의 기하학적 구조에 대해 불가지론적이다. 비구조화 텐서에서 작동하는 모델과 구조화된 다양체에서 작동하는 모델을 구별하지 않는다. Grassmann 다양체의 선택은 추가적인, 기하학 인식 편향을 부여하는 것으로 볼 수 있다:

- 먼저 $M$의 로컬 이웃을 선형 축소와 외적(wedge product)을 통해 $\mathrm{Gr}(2,r)$의 부분공간으로 인코딩한다.
- 그런 다음 MLP와 게이팅을 사용하여 Grassmann 다양체 위의 유도된 변환을 근사한다.
- 마지막으로 원래 표현 공간으로 다시 매핑한다.

이 의미에서, 근본적 근사 능력을 변경하는 것이 아니다 — 네트워크는 원칙적으로 여전히 보편적이다 — 하지만 그 능력을 실현하는 방식을 제약한다. 모든 비국소적 상호작용은 명시적 구조를 가진 유한 차원 다양체를 통해 팩터링되어야 한다. 이것은 정확히 attention이 강제하지 않는 것이다: attention은 고차원 텐서 공간의 매우 자유로운 탐색을 허용한다.

### 6.4 Global and Long-Range Invariants as the Next Step

현재 Causal Grassmann 설계는 로컬 윈도우만 사용한다. 장거리 의존성은 깊이와 다중 스케일 윈도우를 통해 암묵적으로 모델링된다. 이것은 여기서 연구된 과제에는 충분하지만, 이 연구를 동기 부여한 직관과 일치하는 자연스러운 다음 단계를 시사한다:

> 시퀀스 수준 Grassmann 흐름의 명시적 전역 또는 장거리 불변량을 구성하고 이를 특성으로 피드백한다.

예를 들어, 다음을 계산할 수 있다:

- 시퀀스에 걸친 부분공간의 전반적 궤적을 요약하는 **"평균 Grassmann 방향"**.
- 주성분 방향이나 곡률 유사 양(curvature-like quantities)과 같은 Plücker 좌표의 **시퀀스 수준 통계**.
- 특정 부분공간이 깊이에 걸쳐 얼마나 안정적인지를 측정하는 **교차 레이어 불변량**.

이 불변량들은 각 레이어에 보조 입력이나 게이트로 주입되어, 아키텍처를 로컬 흐름이 전역 제약에 의해 안내되는 시스템으로 전환할 수 있다. 이것은 정보 기하학에서 로컬 메트릭(예: Fisher 정보)과 전역 곡률이 함께 추론을 형성하는 로컬과 전역 구조 간의 상호작용을 반향할 것이다.

본 논문에서는 핵심 아이디어를 명확하게 유지하기 위해 의도적으로 최소 설계 — $k = 2$, 명시적 전역 불변량 없음 — 로 제한했다. 그러나 "전역 불변량 + 로컬 Grassmann 흐름"을 기하학 인식 추론에 대한 향후 연구의 유망한 방향으로 본다.

---

## 7. Related Work

### 7.1 Efficient and Long-Context Transformers

Self-attention의 이차 비용을 줄이려는 대규모 연구가 있으며, 선형화/커널화 attention, 희소/로컬 attention 패턴, 메모리 증강/검색 기반 아키텍처를 포함한다. 이러한 접근법들은 일반적으로:

- $QK^\top$ 계산을 근사하거나 희소화하거나,
- attention을 로컬 윈도우나 구조화된 패턴으로 제한하거나,
- 컨텍스트의 일부를 외부 메모리나 캐시로 오프로드한다.

그러나 이들 모두 동일한 핵심 연산을 유지한다: 모델은 여전히 $L \times L$ 쌍별 가중치 행렬을 계산(또는 근사)한다. 본 연구는 직교적이다: attention을 완전히 제거하고 Grassmann 흐름 기반의 기하학적 믹싱 규칙이 그 역할을 채울 수 있는지 탐구한다.

### 7.2 State-Space Models and Structured Sequence Models

상태 공간 모델과 관련 아키텍처는 시퀀스를 선형 동역학계에 의해 지배되는 신호로 해석하며, 종종 비선형 리드아웃과 결합된다. 이 모델들은 시퀀스 길이에 대한 선형 복잡도로 장기 컨텍스트 모델링에 뛰어나며, 제어 이론 및 신호 처리와 강한 연관이 있다.

Grassmann 믹싱은 상태 공간 모델과 시간에 따라 진화하는 구조화된 잠재 상태를 유지한다는 아이디어를 공유하지만, 강조점이 다르다:

- **SSM**은 잠재 상태의 시간적 진화에 초점을 맞춘다; 그들의 기하학은 종종 암묵적이다.
- **Grassmann 믹싱**은 표현 공간에서의 기하학적 진화에 초점을 맞추며, 시간은 인과적 윈도우를 통해 진입한다.

두 관점은 상호보완적이며, SSM 스타일의 시간적 동역학을 은닉 표현에 대한 Grassmann 제약과 결합하는 하이브리드 아키텍처가 향후 연구의 흥미로운 방향이다.

### 7.3 Geometric and Manifold-Based Representation Learning

쌍곡면, 구면 및 기타 리만 다양체를 포함한 비유클리드 공간에서의 학습에 대한 관심이 증가하고 있다. 이러한 접근법들은 일반적으로 데이터를 거리가 기반 구조(예: 계층, 주기성)를 더 잘 포착하는 곡면 다양체에 임베딩한다.

Grassmann 다양체는 부분공간 클러스터링, 저랭크 근사, 메트릭 학습과 같은 고전적 기계학습 맥락에서 부분공간의 집합을 표현하는 데 나타났다. 시퀀스 모델에서 주요 믹싱 메커니즘으로의 사용은 덜 탐구되었다. 본 기여는 Grassmann–Plücker 파이프라인을 Transformer 유사 블록에 직접 통합하여, 부분공간 기하학을 시퀀스 상호작용 메커니즘의 핵심 부분으로 전환하는 것이다.

### 7.4 Interpretability and Attention Analysis

Attention 맵은 Transformer에서 해석 가능성의 대리로 자주 사용된다: 어떤 토큰이 어떤 다른 토큰에 attend하는지 시각화한다. 그러나 attention 가중치가 인과적 중요성과 일치한다는 보장이 없으며, 레이어와 헤드에 걸쳐 집계하면 고도로 복잡한 패턴이 된다.

본 연구는 그 자체로 새로운 해석 가능성 방법을 도입하지 않지만, 분석의 대상을 변경한다. Attention 텐서를 이해하려 하는 대신, Grassmann 특성의 진화를 분석할 것을 제안한다. 이것들은 기하학적 또는 대수적 분석에 더 자연스럽게 적합할 수 있는 유한 차원의 구조화된 객체이다.

---

## 8. Conclusion and Future Work

간단하지만 근본적인 질문을 재방문했다: Transformer에서 일반적으로 구현되는 명시적 self-attention이 정말 강력한 시퀀스 모델링과 추론에 필요한가?

Attention을 텐서 리프팅의 한 형태로 재해석함으로써, 그 힘이 수학적 추적 가능성의 비용으로 온다고 주장했다: 모델의 핵심은 명시적 불변량으로 전역적 행동을 요약하기 어려운 고차원 텐서 공간에 산다. 그런 다음 시퀀스 상호작용이 $L \times L$ attention 행렬이 아닌 Grassmann 다양체 위의 흐름에 의해 지배되는 대안을 제안했다.

결과 Causal Grassmann 아키텍처는:

- 13~18M 파라미터에서 Wikitext-2의 Transformer 베이스라인과 경쟁적이면서 **완전히 attention-free**이며,
- 고정 DistilBERT backbone에 플러그인할 때 SNLI에서 Transformer 기반 분류 헤드를 **약간 능가**하고,
- 고정 축소 차원과 윈도우 크기에서 시퀀스 길이에 대해 **선형인** 점근적 복잡도를 갖는다.

이 실증적 결과를 넘어, 주요 기여는 개념적이다: Grassmann 흐름은 핵심 연산이 비구조화 텐서 공간이 아닌 명시적 구조를 가진 유한 차원 다양체 위에 사는 시퀀스 모델을 어떻게 설계할 수 있는지의 구체적 사례를 제공한다. 이것은 신경망에서의 추론에 대한 더 기하학적인 이해로의 문을 연다.

향후 연구의 많은 방향이 있다:

- **전역 및 장거리 불변량.** Grassmann 흐름의 시퀀스 수준 불변량 — 예: 평균 부분공간, 곡률 유사 측도, 또는 교차 레이어 안정성 통계 — 을 개발하고 로컬 믹싱에 대한 특성 또는 제약으로 주입한다.
- **더 풍부한 Grassmann 구조.** $k = 2$ 부분공간을 넘어, 더 고차원 부분공간을 탐구하고, 레이어에 걸쳐 $\mathrm{Gr}(k,r)$ 위의 매끄러운 궤적을 장려하는 정규화기를 연구한다.
- **하이브리드 아키텍처.** Grassmann 믹싱을 상태 공간 모델, 커널화 attention, 또는 합성곱 모듈과 결합하여 로컬, 전역, 시간적 정보의 균형을 더 잘 맞춘다.
- **해석 가능성 연구.** Plücker 좌표, 모델 행동, 인간이 이해 가능한 패턴 간의 상관관계를 체계적으로 조사하며, 원시 attention 맵보다 더 안정적인 불변량을 정의하는 것을 목표로 한다.
- **스케일링과 엔지니어링.** 퓨즈된 Grassmann 커널과 최적화된 GPU 연산자를 구현하여 이론적 선형 스케일링을 실현하고, 더 큰 규모와 더 도전적인 추론 벤치마크에서 아키텍처를 테스트한다.

요약하면, 결과는 강력한 시퀀스 모델링에 근본적으로 필요한 것이 attention 자체가 아니라, 표현이 자신이 거주하는 다양체 위에서 이동하는 원칙적인 방법이라는 것을 시사한다. Grassmann 흐름은 이 아이디어의 하나의 구체적 실현을 제공하며, 신경 아키텍처 설계에서 attention에 대한 기하학적 대안의 추가 탐구를 장려하기를 바란다.

---

## References

[1] A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
[2] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.
[3] A. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI, 2019.
[4] T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS, 2020.
[5] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.
[6] S. Merity et al., "Pointer Sentinel Mixture Models," arXiv:1609.07843, 2016.
[9] S. R. Bowman et al., "A Large Annotated Corpus for Learning Natural Language Inference," EMNLP, 2015.
[11] R. Hartshorne, *Algebraic Geometry*, Springer, 1977.
[12] J. M. Lee, *Introduction to Riemannian Manifolds*, 2nd ed., Springer, 2018.
[13] A. Edelman et al., "The Geometry of Algorithms with Orthogonality Constraints," SIAM J. Matrix Anal. Appl., 1998.
[14] P.-A. Absil et al., *Optimization Algorithms on Matrix Manifolds*, Princeton University Press, 2008.
