---
title: "Mamba-3: 지수-사다리꼴 이산화, 복소 SSM, MIMO를 통한 표현력 높은 상태 공간 모델"
date: 2026-03-30 12:00:00
categories:
  - 인공지능
tags:
  - SSM
  - Mamba
  - State Space Model
  - Linear Attention
  - MIMO
  - Sequence Modeling
  - 논문 리뷰
---

> **논문 링크**: [arXiv 2603.15569](https://arxiv.org/abs/2603.15569)
>
> **저자**: Tri Dao, Albert Gu
>
> **소속**: Together AI / Carnegie Mellon University

---

## 초록 (Abstract)

추론 시간 연산(inference-time compute) 확장이 LLM 성능에 점점 더 중요해지면서, 모델 설계에서 효율성이 핵심이 되었다. Transformer는 품질 면에서 뛰어나지만, **이차적(quadratic) 연산량**과 **선형 메모리**로 인해 추론 병목이 발생한다. 이에 따라 선형 연산량과 상수 메모리를 가진 차선형(sub-quadratic) 모델이 개발되었으나, 많은 모델이 품질과 상태 추적(state tracking) 능력을 희생하면서도 하드웨어 효율성이 낮은 문제가 있었다.

본 논문은 상태 공간 모델(SSM) 원리에 기반하여 **Mamba-3**에 세 가지 방법론적 개선을 도입한다:

1. **지수-사다리꼴(Exponential-Trapezoidal) 이산화** — 더 표현력 높은 순환 구조
2. **복소값(Complex-valued) 상태 업데이트** — 더 풍부한 상태 추적 가능
3. **다입력-다출력(MIMO)** — 디코드 지연시간 증가 없이 성능 향상

1.5B 규모에서 Mamba-3 MIMO는 Transformer 대비 +2.2, Mamba-2 대비 +1.9, Gated DeltaNet 대비 +1.8의 정확도 향상을 달성한다.

---

## 1. 서론 (Introduction)

테스트 시점 연산이 LLM 발전을 주도하고 있다. Chain-of-Thought 추론과 반복적 정제가 새로운 능력을 열어주고, 병렬 에이전트 워크플로가 효율적 추론에 대한 요구를 강화한다.

Transformer는 업계 표준이지만 근본적 병목이 존재한다:
- **KV 캐시 메모리**: 시퀀스 길이에 비례하여 선형 증가
- **Self-Attention 연산**: 이차적으로 증가

<div class="mermaid">
graph LR
    subgraph T["Transformer"]
        A["시퀀스 길이 T"] --> B["KV 캐시: O&#40;T&#41;"]
        A --> C["Attention: O&#40;T²&#41;"]
    end
    subgraph S["SSM &#40;Mamba&#41;"]
        D["시퀀스 길이 T"] --> E["상태 메모리: O&#40;1&#41;"]
        D --> F["연산: O&#40;T&#41;"]
    end
    style T fill:#ff6b6b,color:#fff
    style S fill:#51cf66,color:#fff
</div>

이에 따라 Mamba-2, Gated DeltaNet 등 SSM과 선형 어텐션 기반 모델이 연구되어 왔으며, 현재 하이브리드 아키텍처에서 순수 Transformer와 동등한 성능을 보여주고 있다.

그러나 여전히 품질-효율성 프론티어를 발전시킬 여지가 크다:
- Mamba-2는 훈련 속도를 개선했지만 **표현력을 희생**
- 선형 모델은 **패리티 감지** 같은 상태 추적 작업에서 어려움
- 이론적으로 효율적인 추론이 **하드웨어에서는 비효율적** (낮은 연산 강도)

### Mamba-3의 핵심 기여

<div class="mermaid">
graph TB
    M["Mamba-3"] --> ET["지수-사다리꼴 이산화"]
    M --> CS["복소값 SSM"]
    M --> MI["MIMO"]
    ET --> ET1["Mamba-2의 휴리스틱을 이론적으로 정형화"]
    ET --> ET2["2차 정확도의 상태-입력 적분"]
    ET --> ET3["암시적 너비-2 합성곱 → 외부 합성곱 제거"]
    CS --> CS1["회전 동역학으로 상태 추적 가능"]
    CS --> CS2["데이터 의존 RoPE로 효율적 구현"]
    CS --> CS3["패리티 작업 100% 정확도"]
    MI --> MI1["디코딩 FLOPs 최대 4배 증가"]
    MI --> MI2["상태 크기·지연시간 유지"]
    MI --> MI3["하드웨어 활용률 극대화"]
</div>

**핵심 결과 요약:**

| 지표 | Mamba-3 SISO | Mamba-3 MIMO | Mamba-2 | GDN | Transformer |
|------|:-----------:|:------------:|:-------:|:---:|:-----------:|
| 다운스트림 정확도 (1.5B) | 56.4% | **57.6%** | 55.7% | 55.8% | 55.4% |
| 패리티 정확도 | 100% | — | 0.9% | — | — |
| 동등 PPL 상태 크기 | 64 | 64 | 128 | — | — |

---

## 2. 사전 지식 (Preliminaries)

### 2.1 표기법

- 스칼라: 일반 문자 ($x, y$)
- 텐서(벡터, 행렬): 볼드 문자 ($\mathbf{h}, \mathbf{C}$)
- $T$: 시퀀스 길이, $D$: 모델 차원, $N$: SSM 상태 크기
- $\odot$: 아다마르(원소별) 곱
- $\text{Diag}(\mathbf{v})$: 벡터 $\mathbf{v}$를 대각 행렬로 변환
- $\alpha\_{t \cdots s} = \prod\_{i=s}^{t} \alpha\_i$: 시간 구간에 걸친 곱

### 2.2 SSM 기초

상태 공간 모델은 **연속 시간 선형 동역학**을 두 방정식으로 표현한다:

$$\dot{\mathbf{h}}(t) = \mathbf{A}(t)\mathbf{h}(t) + \mathbf{B}(t)x(t)$$

$$y(t) = \mathbf{C}(t)^\top \mathbf{h}(t)$$

> **직관적 이해**: 은닉 상태 $\mathbf{h}$가 행렬 $\mathbf{A}$에 의해 시간에 따라 진화하며, 입력 $x$가 $\mathbf{B}$를 통해 주입되고, 출력 $y$는 $\mathbf{C}$로 읽어낸다.

<div class="mermaid">
graph LR
    X["입력 x&#40;t&#41;"] -->|"B&#40;t&#41;"| H["은닉 상태 h&#40;t&#41;"]
    H -->|"A&#40;t&#41;"| H
    H -->|"C&#40;t&#41;ᵀ"| Y["출력 y&#40;t&#41;"]
    style H fill:#4dabf7,color:#fff
</div>

이산 시퀀스에서 시간 간격 $\Delta\_t$로 이산화하면 (Mamba-1, Mamba-2 방식):

$$\mathbf{h}\_t = e^{\Delta\_t \mathbf{A}\_t} \mathbf{h}\_{t-1} + \Delta\_t \mathbf{B}\_t x\_t$$

$$y\_t = \mathbf{C}\_t^\top \mathbf{h}\_t$$

#### Mamba-2의 매개변수화

Mamba-2는 상태 전이를 $\mathbf{A}\_t = A\_t \mathbf{I}\_{N \times N}$으로 단순화한다. $\alpha\_t := e^{\Delta\_t A\_t} \in (0, 1)$, $\gamma\_t := \Delta\_t$로 정의하면:

$$\mathbf{h}\_t = \alpha\_t \mathbf{h}\_{t-1} + \gamma\_t \mathbf{B}\_t x\_t, \quad y\_t = \mathbf{C}\_t^\top \mathbf{h}\_t$$

> **직관적 이해**: $\alpha\_t$는 **기억 수평선**을 제어한다. $\Delta\_t$가 크면 빠르게 잊고 현재 토큰에 집중하며, 작으면 이전 상태를 오래 유지한다.

```
α_t ≈ 1.0 (큰 Δ): 이전 상태 거의 유지, 현재 입력 약하게 반영
  h_t ≈ h_{t-1}    "장기 기억 모드"

α_t ≈ 0.0 (작은 Δ): 이전 상태 잊고, 현재 입력 강하게 반영
  h_t ≈ γ_t B_t x_t  "리셋 모드"
```

### 2.3 구조화된 마스크 표현과 상태 공간 이중성 (SSD)

Mamba-2는 SSM의 시간 단위 순환이 **행렬 형태로 벡터화**될 수 있음을 보여주었다:

$$\mathbf{Y} = (\mathbf{L} \odot \mathbf{C}\mathbf{B}^\top) \mathbf{X}$$

여기서 $\mathbf{L} \in \mathbb{R}^{T \times T}$는 구조화된 마스크, $\mathbf{Q} := \mathbf{C}$, $\mathbf{K} := \mathbf{B}$, $\mathbf{V} := \mathbf{X}$로 놓으면 **어텐션과의 연결**이 드러난다. $\mathbf{L}$은 데이터 의존적 마스크 역할을 한다.

<div class="mermaid">
graph LR
    subgraph "상태 공간 이중성 &#40;SSD&#41;"
        R["순환형: h_t = α_t h_{t-1} + γ_t B_t x_t"] <-->|"등가"| P["병렬형: Y = &#40;L ⊙ CB^T&#41; X"]
    end
    R -->|"디코딩: O&#40;1&#41; 메모리"| DEC["효율적 추론"]
    P -->|"훈련: 행렬 곱"| TRN["빠른 학습"]
</div>

Mamba-2의 마스크 구조:

$$\mathbf{L} = \begin{bmatrix}1 & & & \\ \alpha\_1 & 1 & & \\ \alpha\_2\alpha\_1 & \alpha\_2 & 1 & \\ \vdots & & & \ddots \end{bmatrix} \cdot \text{Diag}(\gamma)$$

> **직관적 이해**: 마스크의 $(i,j)$ 원소는 시점 $j$의 입력이 시점 $i$의 출력에 미치는 영향의 가중치다. $\alpha$ 값들이 누적되어 먼 과거일수록 영향이 지수적으로 감소한다.

---

## 3. 방법론 (Methodology)

### 3.1 지수-사다리꼴 이산화 (Exponential-Trapezoidal Discretization)

#### 3.1.1 지수 조정 이산화 개요

연속 시간 SSM을 이산화할 때, "지수 조정 시스템" $e^{-At}x(t)$를 분석하면 상태 전이와 상태 입력 적분을 분리할 수 있다:

$$\mathbf{h}(\tau\_t) = \exp\left(\int\_{\tau\_{t-1}}^{\tau\_t} A(s)ds\right)\mathbf{h}(\tau\_{t-1}) + \int\_{\tau\_{t-1}}^{\tau\_t} \exp\left(\int\_\tau^{\tau\_t} A(s)ds\right)\mathbf{B}(\tau)x(\tau)d\tau$$

> **직관적 이해**: 첫 번째 항은 이전 상태의 감쇠, 두 번째 항은 새 입력의 누적이다. 핵심 차이는 **두 번째 적분을 어떻게 근사하느냐**에 있다.

<div class="mermaid">
graph TB
    INT["상태-입력 적분 근사 방법"] --> ZOH["ZOH: 우측 끝점 고정"]
    INT --> EE["지수-오일러: 오일러 법칙"]
    INT --> ETR["지수-사다리꼴: 사다리꼴 법칙"]
    ZOH --> ZR["h_t = e^&#40;ΔA&#41; h + A⁻¹&#40;e^&#40;ΔA&#41;-I&#41; Bx"]
    EE --> ER["h_t = α h + Δ B_t x_t"]
    ETR --> TR["h_t = α h + β B_{t-1}x_{t-1} + γ B_t x_t"]
    ZR -->|"정확하지만 A⁻¹ 필요"| C1["S4D, S5"]
    ER -->|"단순하지만 1차 정확도"| C2["Mamba-1, Mamba-2"]
    TR -->|"2차 정확도, A⁻¹ 불필요"| C3["Mamba-3"]
    style ETR fill:#4dabf7,color:#fff
    style C3 fill:#4dabf7,color:#fff
</div>

#### Zero-Order Hold (ZOH)

구간 $[\tau\_{t-1}, \tau\_t]$에서 모든 값을 우측 끝점에서의 상수로 취급:

$$\mathbf{h}\_t = \exp(\Delta\_t A\_t)\mathbf{h}\_{t-1} + A\_t^{-1}(\exp(\Delta\_t A\_t) - I)\mathbf{B}\_t x\_t$$

#### 지수-오일러 (Exponential-Euler, Mamba-1/2)

오일러 법칙으로 상태-입력 적분을 근사:

$$\mathbf{h}\_t = e^{\Delta\_t A\_t}\mathbf{h}\_{t-1} + \Delta\_t \mathbf{B}\_t x\_t \quad (4)$$

> **직관적 이해**: Mamba-1과 Mamba-2에서 이론적 정당화 없이 사용된 휴리스틱을 **지수-오일러 이산화**로 공식화한 것이다.

#### 지수-사다리꼴 (Exponential-Trapezoidal, Mamba-3)

**명제 3.1.1**: 일반화된 사다리꼴 법칙으로 상태-입력 적분을 근사하면:

$$\mathbf{h}\_t = e^{\Delta\_t A\_t}\mathbf{h}\_{t-1} + (1-\lambda\_t)\Delta\_t e^{\Delta\_t A\_t}\mathbf{B}\_{t-1}x\_{t-1} + \lambda\_t \Delta\_t \mathbf{B}\_t x\_t \quad (5)$$

계수를 정의하면:

$$\mathbf{h}\_t = \alpha\_t \mathbf{h}\_{t-1} + \beta\_t \mathbf{B}\_{t-1}x\_{t-1} + \gamma\_t \mathbf{B}\_t x\_t \quad (6)$$

| 계수 | 정의 | 역할 |
|------|------|------|
| $\alpha\_t$ | $e^{\Delta\_t A\_t}$ | 상태 감쇠 (이전과 동일) |
| $\beta\_t$ | $(1-\lambda\_t)\Delta\_t e^{\Delta\_t A\_t}$ | **이전** 입력의 기여 (새로운 항) |
| $\gamma\_t$ | $\lambda\_t \Delta\_t$ | **현재** 입력의 기여 |
| $\lambda\_t$ | $\in [0, 1]$, 데이터 의존 | 이전/현재 입력 비율 제어 |

> **직관적 이해**: Mamba-2는 현재 시점의 입력 $x\_t$만 반영하지만, Mamba-3는 **이전 시점 $x\_{t-1}$도 함께** 고려한다. $\lambda\_t$가 이 둘의 비율을 데이터에 따라 조절한다.

```
Mamba-2 (지수-오일러):     h_t = α·h_{t-1} + γ·B_t·x_t
                                  ↑ 감쇠        ↑ 현재만

Mamba-3 (지수-사다리꼴):   h_t = α·h_{t-1} + β·B_{t-1}·x_{t-1} + γ·B_t·x_t
                                  ↑ 감쇠        ↑ 이전 입력 (NEW)    ↑ 현재
```

**특수 경우:**
- $\lambda\_t = 1$ → Mamba-2의 오일러 법칙 복원
- $\lambda\_t = 1/2$ → 고전적 사다리꼴 법칙
- $\lambda\_t = 1/2 + O(\Delta\_t)$ → **2차 정확도**, 오차 $O(\Delta\_t^3)$

#### 이산화 방법 비교 (Table 1)

| 방법 | $\alpha\_t$ | $\beta\_t$ | $\gamma\_t$ | 사용 모델 |
|------|-----------|----------|-----------|---------|
| Forward Euler | $I + \Delta A$ | — | $\Delta$ | — |
| Backward Euler | $(I - \Delta A)^{-1}$ | — | $(I - \Delta A)^{-1}\Delta$ | — |
| Trapezoidal | $(I-\frac{\Delta A}{2})^{-1}(I+\frac{\Delta A}{2})$ | — | $(I-\frac{\Delta A}{2})^{-1}\Delta$ | S4 |
| ZOH | $e^{\Delta A}$ | — | $A^{-1}(e^{\Delta A}-I)$ | S4D, S5 |
| **지수-오일러** | $e^{\Delta\_t A\_t}$ | — | $\Delta\_t$ | **Mamba-1, 2** |
| **지수-사다리꼴** | $e^{\Delta\_t A\_t}$ | $(1-\lambda\_t)\Delta\_t e^{\Delta\_t A\_t}$ | $\lambda\_t \Delta\_t$ | **Mamba-3** |

#### 3.1.2 암시적 합성곱으로서의 지수-사다리꼴 순환

지수-사다리꼴 이산화는 상태-입력 $\mathbf{v}\_t = \mathbf{B}\_t x\_t$에 대해 **너비 2의 데이터 의존 합성곱**을 적용한 후 선형 순환을 수행하는 것과 동일하다.

<div class="mermaid">
graph LR
    subgraph "Mamba-2 &#40;기존&#41;"
        V1["v_t = B_t x_t"] --> R1["h_t = α h_{t-1} + γ v_t"]
    end
    subgraph "Mamba-3 &#40;신규&#41;"
        V2["v_t = B_t x_t"] --> CONV["너비-2 합성곱: β v_{t-1} + γ v_t"] --> R2["h_t = α h_{t-1} + conv_out"]
    end
    style CONV fill:#ffd43b,color:#000
</div>

> **직관적 이해**: 기존 Mamba 모델들은 외부에 별도의 짧은 인과 합성곱(short causal convolution)이 필요했다. Mamba-3는 이산화 자체에 합성곱이 내장되어 **외부 합성곱을 제거**할 수 있다.

#### 3.1.3 병렬 표현

SSD 프레임워크에서 지수-사다리꼴 마스크는 감쇠 행렬과 2-밴드 합성곱 구조의 곱으로 분해된다:

$$\mathbf{L} = \underbrace{\begin{bmatrix}1 & & \\ \alpha\_1 & 1 & \\ \alpha\_2\alpha\_1 & \alpha\_2 & 1 \\ \vdots & & & \ddots \end{bmatrix}}\_{\text{감쇠 행렬}} \cdot \underbrace{\begin{bmatrix}\gamma\_0 & & \\ \beta\_1 & \gamma\_1 & \\ 0 & \beta\_2 & \gamma\_2 \\ \vdots & & & \ddots \end{bmatrix}}\_{\text{2-밴드 행렬 (NEW)}} \quad (7)$$

> **직관적 이해**: Mamba-2의 마스크는 감쇠 × 대각(γ)이었다. Mamba-3는 대각을 **2-밴드 행렬**로 확장하여, 인접 시점 간 합성곱을 인코딩한다.

---

### 3.2 복소값 SSM (Complex-Valued SSMs)

현대 SSM 아키텍처는 상태 전이를 점차 단순화해왔다: S4의 복소 Normal Plus Low Rank → Mamba-2의 스칼라 항등 행렬. 이 단순화는 언어 모델링 성능을 유지했지만, **상태 추적 작업(패리티 등)에서 성능을 크게 저하**시켰다.

핵심 문제: 실수값, 비음수 고유값 전이로는 **회전 동역학**을 표현할 수 없다. 예를 들어 패리티 함수는 $\mathbf{h}\_t = \mathbf{R}(\pi x\_t)\mathbf{h}\_{t-1}$ 형태의 2D 회전이 필요하다.

<div class="mermaid">
graph TB
    subgraph "실수값 SSM &#40;Mamba-2&#41;"
        RS["α ∈ &#40;0,1&#41;: 스케일링만 가능"] --> RL["축소/유지만 가능, 회전 불가"]
        RL --> RF["패리티 정확도: 0.9%"]
    end
    subgraph "복소값 SSM &#40;Mamba-3&#41;"
        CS["α + iθ: 스케일링 + 회전"] --> CL["크기 변화 + 각도 회전 가능"]
        CL --> CF["패리티 정확도: 100%"]
    end
    style RF fill:#ff6b6b,color:#fff
    style CF fill:#51cf66,color:#fff
</div>

#### 3.2.1 지수-오일러 이산화에서의 복소 SSM

**명제 3.2.1 (복소-실수 SSM 등가성):** 상태 차원 $N/2$의 복소값 SSM은 차원 $N$의 실수값 SSM과 등가이며, 이때 **2×2 회전 행렬로 구성된 블록 대각 전이 행렬**을 갖는다.

복소 연속 시스템:

$$\dot{\mathbf{h}}(t) = \text{Diag}(A(t) + i\theta(t))\mathbf{h}(t) + (\mathbf{B}(t) + i\hat{\mathbf{B}}(t))x(t)$$

지수-오일러 이산화 후 등가 실수 형태:

$$\mathbf{h}\_t = e^{\Delta\_t A\_t} \mathbf{R}\_t \mathbf{h}\_{t-1} + \Delta\_t \mathbf{B}\_t x\_t$$

$$y\_t = \mathbf{C}\_t^\top \mathbf{h}\_t$$

여기서 $\mathbf{R}\_t$는 $\theta(t)$에 의해 결정되는 2×2 회전 행렬의 블록 대각 행렬이다.

> **직관적 이해**: 복소수의 곱셈은 기하학적으로 **크기 변화 + 각도 회전**이다. 실수만으로는 크기만 변할 수 있지만, 복소 성분을 추가하면 상태 벡터가 회전할 수 있어 패리티 같은 작업이 가능해진다.

```
실수값 상태 업데이트:        복소값 상태 업데이트:
  h ──[×α]──> h'              h ──[×α]──[회전θ]──> h'

  ●───→───●                   ●───↗
  (축소만)                      ↗  (회전 + 축소)
                              ●
```

**명제 3.2.2 (데이터 의존 RoPE 등가성):** 이산화된 복소 SSM의 출력은 $\mathbf{B}, \mathbf{C}$에 **데이터 의존 회전 위치 임베딩(RoPE)**을 적용한 순수 스칼라 전이 SSM과 동일하다.

$$\mathbf{h}\_t = e^{\Delta\_t A\_t}\mathbf{h}\_{t-1} + \left(\prod\_{i=0}^{t} \mathbf{R}\_i^\top\right)\Delta\_t \mathbf{B}\_t x\_t$$

> **직관적 이해**: 표준 RoPE는 고정 주파수($\theta[i] = 10000^{-2i/N}$)의 **데이터 무관** 회전이다. Mamba-3의 접근은 시퀀스로부터 학습된 **데이터 의존** 회전을 사용한다. "RoPE 트릭"으로 효율적으로 계산 가능하다.

#### 3.2.2 지수-사다리꼴 이산화에서의 복소 SSM

**명제 3.2.3:** 복소 SSM에 지수-사다리꼴 법칙을 적용하면:

$$\mathbf{h}\_t = \alpha\_t \mathbf{h}\_{t-1} + \beta\_t \left(\prod\_{i=0}^{t-1} \mathbf{R}\_i^\top\right)\mathbf{B}\_{t-1}x\_{t-1} + \gamma\_t \left(\prod\_{i=0}^{t} \mathbf{R}\_i^\top\right)\mathbf{B}\_t x\_t$$

$$y\_t = \left[\left(\prod\_{i=0}^{t} \mathbf{R}\_i^\top\right)\mathbf{C}\_t\right]^\top \mathbf{h}\_t$$

<div class="mermaid">
graph LR
    X_prev["x_{t-1}"] -->|"B_{t-1}"| ROPE1["RoPE&#40;0..t-1&#41;"]
    X_curr["x_t"] -->|"B_t"| ROPE2["RoPE&#40;0..t&#41;"]
    ROPE1 -->|"×β_t"| SUM["⊕"]
    ROPE2 -->|"×γ_t"| SUM
    H_prev["h_{t-1}"] -->|"×α_t"| SUM
    SUM --> H["h_t"]
    H -->|"C_t + RoPE"| Y["y_t"]
    style SUM fill:#ffd43b,color:#000
</div>

#### 상태 추적 실험 결과 (Table 5b)

| 작업 | Mamba-2 | Mamba-3 (RoPE 없음) | **Mamba-3** |
|------|:-------:|:------------------:|:-----------:|
| 패리티 | 0.90% | 2.27% | **100%** |
| 산술 (괄호 없음) | 47.81% | 53.33% | **98.51%** |
| 산술 (괄호 있음) | 0.88% | 2.06% | **87.75%** |

---

### 3.3 다입력-다출력 (MIMO)

#### 디코딩 연산 강도 문제

표준 SISO 모델의 디코딩 연산 강도(FLOPs per byte)는 약 **2.5 ops/byte**로, H100 하드웨어에서 최적화된 행렬 곱의 **295 ops/byte**에 비해 극히 낮다. 이는 메모리 바운드 병목으로, 하드웨어 활용률이 매우 낮음을 의미한다.

```
연산 강도 비교 (ops/byte):

H100 최적 행렬곱  |████████████████████████████████████████| 295
Mamba-3 MIMO(R=4) |████████████                            |  ~10
Mamba-2 SISO      |█                                       | 2.5

              0        50       100      150      200      250      300
```

#### SISO에서 MIMO로

**SISO 기본:** $\mathbf{h}\_t \leftarrow \alpha\_t \mathbf{h}\_{t-1} + \Delta\_t \mathbf{B}\_t \mathbf{x}\_t$, 여기서 $\mathbf{B}\_t \in \mathbb{R}^N$

**MIMO 변환:** $\mathbf{B}\_t \in \mathbb{R}^N \rightarrow \mathbf{B}\_t \in \mathbb{R}^{N \times R}$, $\mathbf{x}\_t \in \mathbb{R}^P \rightarrow \mathbf{x}\_t \in \mathbb{R}^{P \times R}$

<div class="mermaid">
graph TB
    subgraph "SISO &#40;기존&#41;"
        S_B["B ∈ ℝ^N"] --- S_X["x ∈ ℝ^P"]
        S_B -->|"외적: Bxᵀ"| S_H["h ∈ ℝ^&#40;N×P&#41;"]
        S_H -->|"연산 강도"| S_AI["~2.5 ops/byte"]
    end
    subgraph "MIMO &#40;신규&#41;"
        M_B["B ∈ ℝ^&#40;N×R&#41;"] --- M_X["x ∈ ℝ^&#40;P×R&#41;"]
        M_B -->|"행렬곱: BXᵀ"| M_H["h ∈ ℝ^&#40;N×P&#41;"]
        M_H -->|"연산 강도"| M_AI["~Θ&#40;R&#41; ops/byte"]
    end
    style S_AI fill:#ff6b6b,color:#fff
    style M_AI fill:#51cf66,color:#fff
</div>

> **직관적 이해**: SISO에서 외적 $\mathbf{B}\_t \mathbf{x}\_t^\top$은 메모리 접근 대비 연산이 적다. MIMO는 이를 행렬 곱으로 바꿔 **동일한 메모리 트래픽으로 $R$배 더 많은 연산**을 수행한다. 텐서 코어가 본래 설계된 대로 활용된다.

#### MIMO SSM 훈련

MIMO SSM은 $R^2$개의 결합된 SISO 시스템으로 분해된다:

$$\mathbf{h}\_t^{(j)} \leftarrow \alpha\_t \mathbf{h}\_{t-1}^{(j)} + \Delta\_t \mathbf{B}\_t^{(j)} \mathbf{x}\_t^{(j)} \quad \text{(개별 SISO)}$$

$$\mathbf{h}\_t = \sum\_{j=0}^{R-1} \mathbf{h}\_t^{(j)} \quad \text{(상태 집약)}$$

$$\mathbf{y}\_t^{(i)} \leftarrow (\mathbf{C}\_t^{(i)})^\top \mathbf{h}\_t \quad \text{(출력 투영)}$$

> **직관적 이해**: $R$개의 독립적인 "채널"이 각자 상태를 업데이트한 후, 모두 합쳐진다. 병렬 처리가 가능하므로 $R^2$가 아닌 **$R$배의 연산 오버헤드**만 발생한다.

#### 청크 알고리즘 연산량

| | SISO | MIMO |
|---|------|------|
| 인트라-청크 FLOPs | $2TC(N+P)$ | $R$배 |
| 인터-청크 FLOPs | $4TNP + \frac{T}{C}2NP$ | $R$배 |
| $C=P=N$ 총합 | $8TN^2$ | $R \cdot 8TN^2$ |

#### MIMO 인스턴스화

파라미터 폭발을 방지하기 위해 선택적 확장:

- **공유 $\mathbf{B}, \mathbf{C}$ 투영**: $DN \rightarrow DNR$로 적당히 증가
- **헤드별 성분 ($\mathbf{x}, \mathbf{y}, \mathbf{z}$)**: SISO 차원 유지, 학습 가능한 벡터로 원소별 스케일링
- **파라미터 오버헤드**: 헤드당 $D\mathcal{P} + \mathcal{P}R$ (vs $D\mathcal{P}R$)
- **보상**: MLP 폭을 6.6% 줄여 총 파라미터 수 일치

---

### 3.4 Mamba-3 아키텍처

Mamba-3는 Llama 구조를 유지하며 Mamba-3 블록과 SwiGLU 블록을 pre-norm으로 교대 배치한다.

<div class="mermaid">
graph TB
    INPUT["입력 토큰"] --> NORM1["RMSNorm"]
    NORM1 --> SSM["Mamba-3 SSM 블록"]
    SSM --> ADD1["+ 잔차"]
    ADD1 --> NORM2["RMSNorm"]
    NORM2 --> MLP["SwiGLU MLP"]
    MLP --> ADD2["+ 잔차"]
    ADD2 --> OUTPUT["출력"]

    subgraph "Mamba-3 SSM 블록 상세"
        direction TB
        QKV["입력 → B,C 투영 + BCNorm + Bias"]
        QKV --> ROPE["데이터 의존 RoPE"]
        ROPE --> ETRAP["지수-사다리꼴 순환"]
        ETRAP --> MIMO_L["MIMO 집약"]
        MIMO_L --> GATE["Output Gate"]
    end
</div>

#### 갱신된 SSM 순환

Mamba-3는 SSD 레이어를 명제 3.2.2의 복소값 지수-사다리꼴 SSM으로 대체한다:

$$\mathbf{h}\_t = \alpha\_t \mathbf{h}\_{t-1} + \beta\_t \left(\prod\_{i=0}^{t-1}\mathbf{R}\_i^\top\right)\mathbf{B}\_{t-1}\mathbf{x}\_{t-1} + \gamma\_t \left(\prod\_{i=0}^{t}\mathbf{R}\_i^\top\right)\mathbf{B}\_t \mathbf{x}\_t$$

$$y\_t = \left[\left(\prod\_{i=0}^{t}\mathbf{R}\_i^\top\right)\mathbf{C}\_t\right]^\top \mathbf{h}\_t$$

실수 부분 $A$는 표준 SSD 기계를 통해 처리되고, 허수 부분 $\Theta$가 RoPE 계산을 구현한다.

#### BC/QK 정규화

$\mathbf{B}, \mathbf{C}$ 투영 후 **RMS 정규화**를 추가한다. 이는 최근 Transformer에서의 Query-Key 정규화와 동일한 역할이다. 이를 통해 순수 모델에서 이전에 필요했던 post-gate RMSNorm을 제거할 수 있다.

#### B,C 편향 (Biases)

BCNorm 후에 학습 가능한 헤드별, 채널별 편향을 추가한다.

> **직관적 이해**: 이 편향은 **합성곱과 유사한 동작**을 유도한다. $\mathbf{B}$와 $\mathbf{C}$에 데이터 무관 성분이 추가되어, 지수-사다리꼴의 상태-입력 합성곱과 시너지를 이루며 **기존에 필수적이었던 외부 짧은 인과 합성곱을 제거**할 수 있게 한다.

---

## 4. 실험 검증 (Empirical Validation)

### 4.1 언어 모델링

100B FineWeb-Edu 토큰, Llama-3.1 토크나이저, 2K 컨텍스트 길이로 훈련. 4개 스케일(180M, 440M, 880M, 1.5B)에서 평가.

**주요 결과 (Table 3, 1.5B 스케일):**

| 모델 | PPL | 다운스트림 평균 |
|------|:---:|:--------------:|
| Transformer | — | 55.4% |
| Mamba-2 | 10.47 | 55.7% |
| GDN (SISO) | 10.37 | 55.8% |
| **Mamba-3 SISO** | 10.35 | 56.4% |
| **Mamba-3 MIMO** | **10.24** | **57.6%** |

Mamba-3는 기존 모델들이 사용하던 **외부 짧은 합성곱 없이** 이 성능을 달성했다.

#### 4.1.1 MIMO 성능

MIMO (R=4)는 1.5B에서 SISO 대비 PPL 0.11 개선, 다운스트림 평균 1.2%p 향상을 보여준다.

#### 4.1.2 검색 능력

Table 4: 실제 검색 및 합성 needle-in-haystack 작업.

| 작업 | Mamba-3 SISO | Mamba-3 (하이브리드 5:1) |
|------|:-----------:|:---------------------:|
| 연관 기억 (AR) | 경쟁적 | 완벽 |
| QA 작업 | 경쟁적 | 우수 |
| SWDE (비구조적 추출) | 28.5% | 개선됨 |
| FDA | 23.4% | 개선됨 |

하이브리드 구성(선형 + 어텐션 5:1 비율)에서 Mamba-3는 다수의 NIAH 변형에서 만점을 달성한다.

### 4.2 SSM 방법론 절삭 실험 (Ablation)

**구성 요소 절삭 (440M, Table 5a):**

| 구성 | PPL |
|------|:---:|
| Mamba-3 (전체) | **15.72** |
| 지수-사다리꼴 제거 | 16.49 |
| 편향 제거 | 16.68 |
| 합성곱 추가 | 15.85 |

> **직관적 이해**: 지수-사다리꼴과 편향이 각각 독립적으로 중요하다. 외부 합성곱을 다시 추가해도 성능 향상이 미미(15.85 vs 15.72)하여, Mamba-3의 내부 메커니즘이 이를 대체함을 확인한다.

### 4.3 추론 효율성-성능 트레이드오프

**핵심 발견**: Mamba-3는 **절반 크기의 상태**로 Mamba-2와 동등한 PPL을 달성한다.

```
상태 크기별 PPL 비교:

dstate=128  Mamba-2: ████████████████ 10.47
            Mamba-3: ████████████████ 10.35  (더 낮음!)

dstate=64   Mamba-2: █████████████████ 10.8x
            Mamba-3: ████████████████ 10.47  (128과 동등!)

→ Mamba-3(dstate=64) ≈ Mamba-2(dstate=128) → 더 빠른 모델, 같은 품질
```

### 4.4 Mamba-3 커널 성능

**토큰별 디코드 지연시간 (Table 6, BF16, dstate=128):**

| 모델 | 지연시간 |
|------|:-------:|
| Mamba-2 | 0.203 ms |
| GDN | 0.257 ms |
| **Mamba-3 SISO** | **0.156 ms** |
| Mamba-3 MIMO | 0.179 ms |

**Prefill + Decode 지연시간 (Table 7, 2048 토큰):**

| 모델 | Prefill (ms) | Prefill+Decode (ms) |
|------|:-----------:|:-------------------:|
| vLLM (Llama) | 0.26 | — |
| Mamba-2 | 0.51 | 18.62 |
| **Mamba-3 SISO** | 0.51 | **17.57** |
| Mamba-3 MIMO | 0.60 | 18.96 |

**시퀀스 길이별 스케일링:**

| 시퀀스 길이 | Mamba-3 SISO Prefill | Prefill+Decode |
|:-----------:|:-------------------:|:--------------:|
| 512 | 0.51 ms | 4.39 ms |
| 1024 | 1.01 ms | 8.78 ms |
| 2048 | 2.02 ms | 17.57 ms |
| 4096 | 4.01 ms | 35.11 ms |
| 16384 | 16.22 ms | 140.61 ms |

---

## 5. 관련 연구 (Related Work)

### 5.1 선형 시간 시퀀스 믹서

선형 어텐션, 테스트 타임 훈련/회귀, 구조화된 상태 공간 모델의 세 가지 범주가 있다. S4는 구조화된 상태 전이를 사용했고, Mamba-1은 시간 변동적 입력 의존 선택성(selectivity)을 SSM에 도입했다. Mamba-2는 아키텍처를 단순화하면서 언어 모델링 성능을 대체로 유지했지만, 특정 상태 추적 작업에서 능력이 감소했다.

### 5.2 상태 추적과 복소 상태 공간 모델

실수, 비음수 고유값 전이가 패리티 같은 상태 추적 작업에서 능력을 저하시킨다는 연구가 있다. S4를 포함한 이전 SSM 문헌에 복소값 상태 공간이 등장했으나, 이후 아키텍처에서 종종 단순화되어 제거되었다. Mamba-3는 데이터 의존 회전 임베딩을 통한 효율적 구현으로 이 능력을 재도입한다.

### 5.3 다입력-다출력

차선형 모델은 이론적으로 선형 연산을 달성하지만, 낮은 연산 강도로 인해 디코딩 시 하드웨어 활용이 저조한 경우가 많다. MIMO는 메모리 요구를 비례적으로 증가시키지 않으면서 연산 밀도를 높여 이를 해결한다.

### 5.4 상태 공간 모델 관점

저자들은 이 방법론적 기여가 **SSM 중심 관점에서 자연스럽게 도출**되지만, 선형 어텐션이나 테스트 타임 회귀 등 다른 관점에서는 즉각적이지 않다고 주장한다. SSM 프레임워크가 실용적 설계 선택에 이론적 근거를 제공한다.

---

## 6. 결론 및 향후 연구 (Conclusion and Future Work)

세 가지 핵심 기여를 요약한다:

1. **지수-사다리꼴 이산화**: 더 표현력 높은 순환 제공
2. **복소값 상태 업데이트**: 상태 추적 가능
3. **MIMO**: 추론 효율성 향상

이들이 결합하여 차선형 모델의 **성능-지연시간 파레토 프론티어**를 발전시킨다.

**향후 연구 방향:**
- 2K 훈련 길이를 넘어선 긴 컨텍스트 탐구
- 대안적 MIMO 매개변수화
- 언어 모델링 이외의 아키텍처 패러다임으로 일반화

---

## 부록 (Appendices)

### 부록 A: 지수-사다리꼴 이산화 상세

#### A.1 마스크 행렬

식 (7)의 마스크 $\mathbf{L}$이 감쇠 행렬과 2-밴드 합성곱 구조의 곱으로 분해됨을 상세 유도한다.

#### A.2 오차율

**비고 3**: 지수-사다리꼴 이산화는 상태-입력 적분의 2차 이산화이며, 표준 안정성 가정 하에서 오차가 $O(\Delta\_t^3)$으로 스케일링된다.

#### A.3 매개변수화

$\lambda\_t$의 실제 매개변수화 세부사항을 기술한다.

### 부록 B: 복소 SSM 증명

#### B.1 명제 3.2.1 증명

복소-실수 SSM 등가성의 완전한 수학적 증명.

#### B.2 RoPE 트릭 등가성 증명

데이터 의존 RoPE 등가성의 유도 과정.

#### B.3 지수-사다리꼴 + RoPE 증명

명제 3.2.3의 완전한 증명.

### 부록 C: Mamba-3용 MIMO

MIMO 파라미터 매칭 세부사항. 파라미터 폭발을 방지하면서 공정한 비교를 위한 설정.

### 부록 D: 실험 세부사항

| 설정 | 값 |
|------|-----|
| 훈련 데이터 | 100B FineWeb-Edu 토큰 |
| 토크나이저 | Llama-3.1 |
| 컨텍스트 길이 | 2K |
| 모델 스케일 | 180M, 440M, 880M, 1.5B |
| 절삭 스케일 | 440M (2×Chinchilla 최적 토큰) |
| MIMO 랭크 | R = 4 |
| 상태 크기 실험 | dstate ∈ {16, 32, 64, 128} |
| MLP 내부 차원 축소 | 6.6% (1.5B, 파라미터 매칭용) |

### 부록 F: 아키텍처 절삭

BC 편향 매개변수화 연구. 지수-사다리꼴 이산화와 BC 편향의 조합이 외부 짧은 인과 합성곱 없이도 효과적임을 검증.

### 부록 G: 추론 커널 지연시간 분석

#### G.1 커널 구현 및 융합 구조

| 단계 | 구현 | 특징 |
|------|------|------|
| Prefill | Triton | 유연하고 하드웨어 친화적 |
| Decode | CuTe DSL | 텐서 코어 특화 |

지수-사다리꼴, 복소 상태 전이(RoPE 트릭), MIMO 투영이 모두 저지연 커널 연산으로 통합된다.

#### G.2 확장 지연시간 측정

- **하드웨어**: NVIDIA H100-SXM5
- **배치 크기**: 128 토큰 (디코드)
- **정밀도**: FP32, BF16
- **시퀀스 길이**: 512 ~ 16,384

**dstate=64 디코드 지연시간 (BF16):**

| 모델 | 지연시간 |
|------|:-------:|
| Mamba-2 | 0.127 ms |
| GDN | 0.176 ms |
| **Mamba-3 SISO** | **0.110 ms** |
| Mamba-3 MIMO | 0.137 ms |

Mamba-3의 아키텍처 추가 사항들(지수-사다리꼴, 복소 상태, MIMO)이 Mamba-2 대비 **무시할 수 있는 수준의 지연시간 패널티**만 발생시킴을 확인했다.
