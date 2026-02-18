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

## Abstract (초록)

Self-attention은 현대 시퀀스 모델의 사실상 표준 연산이 되었다. 강력한 자연어 성능을 위해서는 dense 또는 approximate attention 메커니즘을 통해 모든 토큰 쌍을 attend해야 한다고 암묵적으로 가정되곤 한다. 본 논문에서는 이 가정에 의문을 제기한다.

저자는 self-attention을 **텐서 리프팅(tensor lifting)**의 한 특수 사례로 재해석한다: 은닉 벡터를 쌍별 상호작용의 고차원 공간으로 매핑하고, 이 리프팅된 텐서에 대한 제약을 통해 학습이 진행된다. 이 리프팅은 매우 표현력이 높지만 수학적으로 추적하기 어렵다; 여러 레이어와 헤드를 거치면서, 소수의 명시적 불변량으로 모델의 행동을 설명하는 것이 거의 불가능해진다. 이러한 관점에서, 대규모 Transformer 모델의 **"해석 불가능성(uninterpretability)"**의 핵심 원인은 단순히 크기가 아니라, 핵심 연산이 분석적으로 불투명한 고차원 텐서 리프트라는 사실에 있다.

대안적 설계로서, 저자는 **Grassmann 흐름(Grassmann flows)** 기반의 attention-free 시퀀스 모델을 제안한다. L×L attention 행렬을 구성하는 대신:

1. 토큰 상태를 저차원 공간으로 축소
2. 로컬 토큰 쌍을 Grassmann 다양체 Gr(2,r) 위의 2차원 부분공간으로 해석
3. Plücker 좌표를 통해 유한 차원 사영 공간에 임베딩
4. 게이트 믹싱 블록으로 기하학적 특성을 은닉 상태에 융합

정보는 명시적 쌍별 가중치가 아닌, 레이어와 다중 스케일 로컬 윈도우에 걸친 **저랭크 부분공간의 제어된 변형**을 통해 시퀀스를 흐른다.

**주요 결과:**
- **Wikitext-2**: 13~18M 파라미터의 순수 Grassmann LM이 크기 매칭된 Transformer 대비 10~15% 이내의 validation perplexity 달성
- **SNLI**: DistilBERT 위에 Grassmann 분류 헤드가 Transformer 헤드를 약간 능가 (validation accuracy 0.8550 vs. 0.8545; test accuracy 0.8538 vs. 0.8511)
- **복잡도**: 고정 랭크에서 시퀀스 길이에 대해 **선형 스케일링** (full self-attention의 이차 비용 대비)

---

## 1. Introduction (서론)

Transformer는 self-attention을 중심 연산으로 만들어 시퀀스 모델링을 재편했다. 은닉 상태 H ∈ ℝ<sup>L×d</sup>가 주어지면, self-attention은 쿼리, 키, 밸류를 구성하고:

$$Q = HW_Q, \quad K = HW_K, \quad V = HW_V$$

L×L 쌍별 호환성 행렬 QK<sup>⊤</sup>을 계산하여, softmax로 정규화한 가중치로 값을 혼합한다.

본 논문에서는 다른 입장을 취한다. attention을 더 싸게, 더 희소하게, 더 확장 가능하게 만드는 것이 아니라, 더 근본적인 질문을 던진다:

> **명시적 self-attention, 즉 L×L 가중치 텐서가 정말 강력한 시퀀스 모델링과 추론에 필요한 근본적 요소인가?**

저자의 답은 **"아니오"**이다. Attention은 표현의 기하학적 리프팅을 구현하는 한 가지 특정 방법이지, 유일한 방법이 아니다.

### 1.1 Attention as Tensor Lifting (텐서 리프팅으로서의 Attention)

개념적으로, self-attention은 텐서 리프팅의 한 형태를 수행한다. 위치 t에서의 은닉 벡터 h<sub>t</sub> ∈ ℝ<sup>d</sup>로부터, 토큰 간 쌍별 관계를 인코딩하는 공간으로 이동한다 — 헤드당 L×L 호환성 텐서와 정규화된 가중치.

기하학적 관점에서, self-attention은 시퀀스를 토큰 표현의 다양체에서 쌍별 상호작용의 훨씬 더 큰 공간으로 리프팅하고, 그곳에서 연산을 수행한 후 다시 투영한다.

> **주장(Claim)**: 대규모 모델의 "해석 불가능성"의 주요 원인은 attention이 수행하는 텐서 리프팅이 수학적으로 추적 불가능하다는 점이다: 각 레이어와 헤드에서 도입되는 자유도가 너무 커서 모델의 전역적 효과를 설명할 수 있는 작고 명시적인 불변량 집합이 존재하지 않는다.

### 1.2 Reasoning as Geometry on a Semantic Manifold (의미 다양체 위의 기하학으로서의 추론)

대안적 출발점은 추론을 **의미 다양체 위의 기하학**으로 생각하는 것이다. 핵심 문제는 텐서 공간으로 리프팅할지 여부가 아니라, 표현의 기하학적 진화 규칙을 어떻게 설계하는가이다.

> **주장(Claim)**: 대규모 Transformer의 비해석성은 단순히 깊이나 파라미터 수 때문이 아니라, 핵심 연산으로 텐서 리프팅을 선택한 것에 근본적 원인이 있다. L×L attention 텐서로 리프팅하는 순간, 소수의 불변량으로 간단하고 명시적이며 전역적인 설명이 가능할 가능성을 이미 잃게 된다.

### 1.3 Grassmann Flows as a Controlled Alternative (제어된 대안으로서의 Grassmann 흐름)

본 논문은 대안적 설계를 탐구한다: attention 텐서로 리프팅하는 대신 **Grassmann 다양체**로 리프팅한다.

Attention과의 핵심 차이는 모델이 명시적이고 유한 차원 구조를 가진 다양체 위에서 작동하도록 제약된다는 것이다. 이 아키텍처를 **Causal Grassmann 모델** 또는 **Grassmann 흐름**이라 부른다. 중요한 점은 이것이 완전히 attention-free라는 것이다: attention 행렬을 구성하거나 softmax 정규화 텐서 가중치를 계산하는 단계가 없다.

### 1.4 Contributions (기여)

본 논문의 기여:
- Self-attention을 텐서 리프팅으로 재해석하고, 이것이 해석 불가능성의 핵심 원인임을 주장
- Grassmann 다양체와 Plücker 좌표 기반의 attention-free 시퀀스 모델 제안
- Wikitext-2와 SNLI에서 실증적 평가
- 시퀀스 길이에 대한 선형 복잡도 분석

---

## 2. Attention as Geometric Lifting and Grassmann Background

### 2.1 Self-Attention as Geometric Lifting (기하학적 리프팅으로서의 Self-Attention)

은닉 상태 H ∈ ℝ<sup>L×d</sup>가 주어지면, 표준 multi-head self-attention 레이어는 각 헤드 h에 대해 다음을 계산한다:

W<sub>Q</sub><sup>(h)</sup>, W<sub>K</sub><sup>(h)</sup>, W<sub>V</sub><sup>(h)</sup> ∈ ℝ<sup>d×d<sub>h</sub></sup>, 일반적으로 d<sub>h</sub> = d/H<sub>heads</sub>.

각 헤드에 대해 attention 행렬을 계산한다:

$$A_h = \text{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{L \times L}$$

헤드 출력을 얻는다:

$$O_h = A_h V_h \in \mathbb{R}^{L \times d_h}$$

출력 O<sub>h</sub>는 연결되어 ℝ<sup>d</sup>로 선형 투영된다.

기하학적으로, 모델은 단순히 은닉 상태의 다양체를 따라 이동하는 것이 아니라, 시퀀스 길이에 대해 이차적으로 증가하는 차원의 쌍별 관계의 구름도 수정한다. 여러 헤드와 레이어에 걸쳐, 이 구름은 극도로 복잡해진다.

### 2.2 Grassmann Manifolds and Plücker Coordinates

**Grassmann 다양체** Gr(k,r)은 ℝ<sup>r</sup>의 모든 k차원 선형 부분공간의 집합이다. 이것은 차원 k(r−k)의 매끄러운 다양체이다. 본 논문에서는 k=2에 초점을 맞추므로, Gr(2,r)은 ℝ<sup>r</sup>의 모든 2차원 부분공간을 매개변수화한다.

**Plücker 임베딩**을 사용하며, 각 k차원 부분공간을 사영 공간의 한 점으로 매핑한다:

$$p_{ij} = u_i v_j - u_j v_i, \quad 1 \leq i < j \leq r$$

Plücker 벡터 p는 u와 v가 span하는 부분공간을 인코딩하는 정규화된 특성 벡터로 볼 수 있다.

### 2.3 Why Grassmann for Sequence Modeling?

**로컬 선형 구조**: 매끄러운 다양체 위에서 로컬 기하학은 접선 공간과 그 부분공간으로 포착된다. Grassmann 다양체는 선형 부분공간의 집합을 자연스럽게 매개변수화하므로, 더 복잡한 구조의 로컬 선형 근사를 표현하기에 적합하다.

**유한 차원 대수적 구조**: Grassmann 다양체는 유한 차원이며 Plücker 임베딩을 통해 사영 공간 내에 위치한다. 따라서 기하학적 정보를 알려진 대수적 제약을 따르는 고정 차원 특성 벡터로 인코딩할 수 있다.

**근사 정리와의 호환성**: 신경망의 보편 근사 정리는 여전히 적용되지만, 근사가 이제 구조를 분석할 수 있는 공간 위에서 전개된다.

---

## 3. Methods: A Causal Grassmann Transformer without Attention

### 3.1 Token and Positional Embeddings

어휘 크기 V에 대한 표준 next-token LM 설정에서, 토큰 시퀀스 (x<sub>1</sub>, ..., x<sub>L</sub>)를 학습된 임베딩 행렬 E ∈ ℝ<sup>V×d</sup>와 위치 임베딩 P ∈ ℝ<sup>L<sub>max</sub>×d</sup>를 사용하여 ℝ<sup>d</sup>에 임베딩한다:

$$h_t^{(0)} = E(x_t) + P_t, \quad t = 1, \dots, L$$

실험에서 d=256을 사용한다. 결과 시퀀스는 N개의 Causal Grassmann Transformer 레이어를 통과한다.

### 3.2 Causal Grassmann Mixing Layer

각 레이어는 H ∈ ℝ<sup>L×d</sup>를 입력으로 받아 업데이트된 시퀀스 H̃ ∈ ℝ<sup>L×d</sup>를 출력한다.

#### Step 1: Linear Reduction (선형 축소)

각 은닉 상태를 저차원 벡터로 축소한다:

$$z_t = W_{\text{red}} h_t + b_{\text{red}}, \quad W_{\text{red}} \in \mathbb{R}^{r \times d},\ b_{\text{red}} \in \mathbb{R}^r$$

실험에서 r=32 (r ≪ d). Z = (z<sub>1</sub>, ..., z<sub>L</sub>) ∈ ℝ<sup>L×r</sup>을 생성한다.

#### Step 2: Multi-Scale Local Pairing (다중 스케일 로컬 페어링)

윈도우 크기(오프셋) 집합을 정의한다:

$$\mathcal{W} = \{1, 2, 4, 8, 12, 16\}$$

또는 더 깊은 모델의 경우 (1,1,2,2,4,4,8,8,12,12,16,16) 같은 다층 스케줄. 각 위치 t와 오프셋 Δ에 대해 t+Δ ≤ L이면 쌍 (z<sub>t</sub>, z<sub>t+Δ</sub>)를 형성한다.

페어링은 **인과적(causal)**이다: t는 오른쪽(미래) 위치와만 쌍을 이루어 left-to-right 언어 모델링과 일관성을 유지한다.

#### Step 3: Grassmann / Plücker Encoding

각 쌍 (z<sub>t</sub>, z<sub>t+Δ</sub>)에 대해, ℝ<sup>r</sup>에서 이 벡터들이 span하는 2차원 부분공간을 고려한다. Plücker 벡터 p<sub>t</sub><sup>(Δ)</sup> ∈ ℝ<sup>C(r,2)</sup>를 형성한다:

$$p_{ij}^{(\Delta)}(t) = z_{t,i} z_{t+\Delta,j} - z_{t,j} z_{t+\Delta,i}, \quad 1 \leq i < j \leq r$$

수치 안정성을 위해 선택적 정규화를 적용한다:

$$\hat{p}_t^{(\Delta)} = \frac{p_t^{(\Delta)}}{\max(\|p_t^{(\Delta)}\|_2, \varepsilon)}$$

#### Step 4: Projection Back to Model Space (모델 공간으로 역투영)

학습된 선형 맵으로 Grassmann 특성을 모델 차원으로 역투영한다:

$$g_t^{(\Delta)} = W_{\text{plü}} \hat{p}_t^{(\Delta)} + b_{\text{plü}}, \quad W_{\text{plü}} \in \mathbb{R}^{d \times \binom{r}{2}}$$

오프셋에 걸쳐 평균으로 집계한다:

$$g_t = \frac{1}{|\mathcal{W}_t|} \sum_{\Delta \in \mathcal{W}_t} g_t^{(\Delta)}$$

벡터 g<sub>t</sub> ∈ ℝ<sup>d</sup>는 위치 t 주변의 다중 스케일 로컬 Grassmann 기하학을 포착한다.

#### Step 5: Gated Fusion (게이트 융합)

원본 은닉 상태와 Grassmann 특성을 연결하여 게이트를 계산한다:

$$u_t = [h_t; g_t] \in \mathbb{R}^{2d}$$

$$\alpha_t = \sigma(W_{\text{gate}} u_t + b_{\text{gate}}), \quad W_{\text{gate}} \in \mathbb{R}^{d \times 2d}$$

혼합 표현:

$$\tilde{h}_t^{\text{mix}} = \alpha_t \odot h_t + (1 - \alpha_t) \odot g_t$$

이후 Layer Normalization과 Dropout을 적용한다.

#### Step 6: Feed-Forward Block

표준 Transformer와 동일하게 position-wise FFN을 적용한다:

$$\phi_t = W_2 \sigma(W_1 \hat{h}_t + b_1) + b_2$$

$$h_t' = \text{LayerNorm}(\hat{h}_t + \phi_t)$$

W<sub>1</sub> ∈ ℝ<sup>d<sub>ff</sub>×d</sup>, W<sub>2</sub> ∈ ℝ<sup>d×d<sub>ff</sub></sup>, d<sub>ff</sub>=4d, 비선형성은 GELU를 사용한다.

N개의 이런 레이어를 쌓으면 전체 Causal Grassmann Transformer가 된다.

### 3.3 Comparison to Self-Attention (Self-Attention과의 비교)

시퀀스 길이 L, 은닉 차원 d에서:

- **Self-attention**: O(Ld² + L²d<sub>head</sub>) — L²항은 QK<sup>⊤</sup> 행렬 곱과 attention × V 곱에서 발생
- **Causal Grassmann**: O(Ld²) — **L²항이 없음**, 고정 r과 m에서 L에 대해 선형

$$\text{Causal Grassmann: } \mathcal{O}(Ld^2) \quad \text{vs.} \quad \text{Self-attention: } \mathcal{O}(L^2 d_{\text{head}} + Ld^2)$$

---

## 4. Experimental Setup (실험 설정)

### 4.1 Wikitext-2 Language Modeling

- **데이터**: Wikitext-2-raw, 블록 크기 L=128 및 L=256
- **토크나이저**: BERT 스타일 WordPiece (V ≈ 30,522)
- **비교 모델**: TransformerLM vs GrassmannLM
- **깊이**: 6레이어 및 12레이어
- **GrassmannLM 설정**: r=32, 다중 스케일 윈도우 𝒲 = {1,2,4,8,12,16}
- **학습**: 30 에포크, 동일 옵티마이저/스케줄, 배치 크기 32 (L=128) / 16 (L=256)

### 4.2 SNLI Natural Language Inference

- **백본**: DistilBERT-base-uncased (고정 특성 추출기)
- **최대 시퀀스 길이**: 문장당 48 토큰
- **분류 헤드 비교**: Transformer head vs Grassmann–Plücker head
- **Grassmann 헤드 하이퍼파라미터**: d<sub>proj</sub>=64, 윈도우 크기 8, d<sub>model</sub>=256, 2 믹싱 레이어, 4 믹싱 헤드, d<sub>ff</sub>=512, dropout 0.1
- **학습**: 20 에포크

---

## 5. Results (결과)

### 5.1 Wikitext-2 Language Modeling

**Table 1: 6-layer 모델 (block size 128)**

| Model | Layers | Params (M) | Val PPL |
|---|---|---|---|
| TransformerLM | 6 | 12.59 | 248.4 |
| GrassmannLM | 6 | 13.00 | 275.7 |
| TransformerLM | 6 | 12.59 | 253.6 |
| GrassmannLM | 6 | 13.00 | 282.3 |

**Table 2: 12-layer 모델 (block size 256)**

| Model | Layers | Params (M) | Val PPL |
|---|---|---|---|
| TransformerLM | 12 | 17.32 | 235.2 |
| GrassmannLM | 12 | 18.16 | 261.1 |

12-layer 모델에서 상대적 격차가 줄어들어, 추가 깊이가 Grassmann 모델의 더 국소적인 믹싱을 보상하는 데 도움이 됨을 시사한다.

### 5.2 SNLI Natural Language Inference

**Table 3: SNLI 분류 정확도**

| Head Type | Val Accuracy | Test Accuracy |
|---|---|---|
| Transformer head | 0.8545 | 0.8511 |
| Grassmann–Plücker head | **0.8550** | **0.8538** |

Grassmann 헤드가 validation과 test 모두에서 Transformer 헤드를 약간 능가한다. 마진은 작지만 개념적으로 중요하다: **명시적 attention 가중치 없이도 경쟁적 정확도를 달성할 수 있음**을 보여준다.

### 5.3 Complexity and Empirical Runtime (복잡도와 실제 런타임)

- **이론적**: Grassmann 레이어는 시퀀스 길이에 대해 선형, self-attention은 이차
- **실제**: 현재 구현은 L=256까지에서 Transformer보다 느림 — Plücker 좌표 계산과 reshape 오버헤드 때문
- Grassmann 연산을 퓨즈하는 전용 구현이 필요

---

## 6. Discussion (논의)

### 6.1 What Does Grassmann Mixing Actually Buy Us?

> **주장(Claim)**: 모델이 기하학적으로 충분히 풍부한 로컬 진화 규칙을 갖추기만 하면, 명시적 attention 가중치 없이도 의미론적 추론이 나타날 수 있다.

Self-attention과 Grassmann mixing의 근본적 차이:
- **Self-attention**: L×L 학습된 가중치 행렬을 통해 각 토큰이 다른 모든 토큰을 봄 → 비구조화 텐서 리프트
- **Grassmann mixing**: 로컬 부분공간 업데이트 시퀀스를 구성 → 다중 스케일 윈도우를 통한 저랭크 부분공간 회전/변형

### 6.2 Interpretability: From Tensor Lifting to Finite-Dimensional Flows

Grassmann 아키텍처는 관련 자유도를 유한 차원의 수학적으로 엄밀한 다양체로 의도적으로 압축한다:
- Gr(2,r)의 차원은 2(r−2) — 고정이고 r에 의해 제어됨
- Plücker 좌표는 알려진 대수적 관계를 만족하는 명시적 특성 벡터

학습 후, Plücker 벡터나 다른 Grassmann 기술자를 후보 설명 불변량으로 취급할 수 있다.

### 6.3 Why Grassmann? A Link to Approximation Theorems

근사 이론의 관점에서, Grassmann 다양체 선택은 추가적인 기하학 인식 편향을 부여하는 것으로 볼 수 있다. 근본적 근사 능력은 변하지 않지만(네트워크는 원칙적으로 여전히 보편적), 그 능력을 실현하는 방식을 제약한다. 모든 비국소적 상호작용은 명시적 구조를 가진 유한 차원 다양체를 통해 팩터링되어야 한다.

### 6.4 Global and Long-Range Invariants as the Next Step

현재 설계는 로컬 윈도우만 사용하며, 장거리 의존성은 깊이와 다중 스케일 윈도우를 통해 암묵적으로 모델링된다. 자연스러운 다음 단계:

- 시퀀스 수준 Grassmann 흐름의 명시적 전역/장거리 불변량을 구성하여 특성으로 피드백
- 예: Plücker 벡터의 전역 평균이나 분산, 레이어 간 Grassmann 각도/거리 등
- "전역 불변량 + 로컬 Grassmann 흐름"이 기하학 인식 추론의 유망한 방향

---

## 7. Related Work (관련 연구)

### 7.1 Efficient and Long-Context Transformers
선형화/커널화 attention, 희소/로컬 attention 패턴, 메모리 증강 아키텍처 등. 모두 L×L 쌍별 가중치 행렬을 유지하지만, 본 연구는 attention을 완전히 제거한다.

### 7.2 State-Space Models
상태 공간 모델과 Grassmann mixing은 시간에 따라 진화하는 구조화된 잠재 상태를 유지한다는 아이디어를 공유하지만, SSM은 선형 동역학계에 초점을, Grassmann은 부분공간 기하학에 초점을 둔다.

### 7.3 Geometric and Manifold-Based Representation Learning
쌍곡면, 구면 등 비유클리드 공간에서의 학습. Grassmann 다양체는 고전적 ML에서 부분공간 클러스터링 등에 사용되었지만, 시퀀스 모델의 주요 믹싱 메커니즘으로의 사용은 덜 탐구되었다.

### 7.4 Interpretability and Attention Analysis
Attention 맵은 해석 가능성의 대리로 자주 사용되지만, attention 가중치가 인과적 중요성과 일치하는 보장이 없다. 본 연구는 분석 대상을 변경한다 — attention 텐서 대신 Grassmann 특성의 진화를 분석할 것을 제안한다.

---

## 8. Conclusion and Future Work (결론 및 향후 과제)

Self-attention을 텐서 리프팅으로 재해석함으로써, 그 힘이 수학적 추적 가능성의 비용으로 온다고 주장했다. 대안으로 시퀀스 상호작용이 L×L attention 행렬이 아닌 **Grassmann 다양체 위의 흐름**에 의해 지배되는 아키텍처를 제안했다.

**Causal Grassmann 아키텍처의 특징:**
- 명시적 attention 가중치 없이도 비자명한 언어 모델링과 NLI 가능
- 시퀀스 길이에 대해 선형 복잡도
- 핵심 연산이 유한 차원 다양체 위에 존재하여 기하학적 분석에 더 적합

**향후 연구 방향:**
- 전역 불변량을 로컬 흐름에 통합
- k > 2 또는 다른 다양체로의 확장
- 최적화된 GPU 구현
- 대규모 모델에서의 검증

> **핵심 메시지**: 강력한 시퀀스 모델링에 근본적으로 필요한 것은 attention 자체가 아니라, 표현이 자신이 거주하는 다양체 위에서 이동하는 원칙적인 방법이다. Grassmann 흐름은 이 아이디어의 하나의 구체적 실현이다.
