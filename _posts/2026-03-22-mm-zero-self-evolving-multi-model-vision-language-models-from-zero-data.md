---
title: "MM-Zero: 데이터 없이 스스로 진화하는 멀티모델 비전-언어 모델"
date: 2026-03-22 12:00:00
categories:
  - 인공지능
tags:
  - VLM
  - Reinforcement Learning
  - Self-Evolution
  - GRPO
  - Multi-Agent
  - 논문 리뷰
---

> **논문 링크**: [arXiv 2603.09206](https://arxiv.org/abs/2603.09206)
>
> **저자**: Zongxia Li, Hongyang Du, Chengsong Huang, Xiyang Wu, Lantao Yu, Yicheng He, Jing Xie, Xiaomin Wu, Zhichao Liu, Jiarui Zhang, Fuxiao Liu
>
> **소속**: University of Maryland, Brown University, Washington University in St. Louis, Adobe, UIUC, USC, NVIDIA

---

## 초록 (Abstract)

자기 진화(Self-evolving)는 LLM과 VLM 같은 기반 모델을 최소한의 인간 개입으로 개선하는 핵심 패러다임으로 부상했다. 최근 연구들은 LLM 에이전트가 데이터 없이도 자기 진화할 수 있음을 보였지만, VLM은 추가적인 시각 모달리티로 인해 최소한 시드 이미지가 필요했다.

본 논문은 **MM-Zero(Multi-model Multimodal Zero)**를 제시한다. 이는 **데이터 없이 VLM 추론을 자기 진화시키는 최초의 RL 기반 프레임워크**다. 기존의 이중 역할(Proposer-Solver) 구조를 넘어, **세 가지 역할**로 구성된 자기 진화 훈련 프레임워크를 도입한다:

<div class="mermaid">
graph LR
    P["🎯 Proposer<br/><i>추상적 시각 개념 생성<br/>+ 질문 구성</i>"]
    C["💻 Coder<br/><i>개념 → 실행 가능 코드<br/>(Python, SVG) → 이미지 렌더링</i>"]
    S["🧠 Solver<br/><i>생성된 시각 콘텐츠로<br/>멀티모달 추론</i>"]
    P -->|"캡션 + 질문"| C
    C -->|"렌더링된 이미지"| S
    S -->|"피드백(보상)"| P
    style P fill:#2d5a27,stroke:#4a9,color:#fff
    style C fill:#1a3a5c,stroke:#4a9,color:#fff
    style S fill:#5c1a3a,stroke:#c44,color:#fff
</div>

세 역할 모두 **동일한 기본 모델에서 초기화**되며, 실행 피드백, 시각적 검증, 난이도 균형을 통합한 보상 메커니즘과 함께 **GRPO(Group Relative Policy Optimization)**로 훈련된다. Qwen3-VL 및 Mimo-VL 모델에서 다양한 멀티모달 벤치마크에 걸쳐 일관된 성능 향상을 달성했다.

---

## 1. 서론 (Introduction)

### 자기 진화 패러다임

자기 진화 메커니즘은 기반 모델 발전의 유망한 최전선이다. 정적인 인간 큐레이션 감독에 의존하는 대신, **스스로 경험을 생성·개선·학습**하는 시스템이다. LLM 영역에서는 이미 성공을 거두었다: LLM이 자체적으로 훈련 태스크를 생성하고 RL이나 코드 실행 피드백으로 자신을 정제할 수 있다.

> 그렇다면 **유사한 자기 진화 패러다임을 VLM으로 확장할 수 있을까?**

### 왜 VLM 자기 진화는 어려운가

<div class="mermaid">
graph TB
    subgraph LLM["LLM 자기 진화 (기존)"]
        direction LR
        L1["텍스트 질문 생성"] --> L2["텍스트 답변"] --> L3["검증 & 학습"]
    end
    subgraph VLM["VLM 자기 진화 (기존 한계)"]
        direction LR
        V1["❌ 이미지 필요"] --> V2["시드 데이터셋<br/>의존"] --> V3["데이터 분포에<br/>제한됨"]
    end
    subgraph MMZ["MM-Zero (본 논문)"]
        direction LR
        M1["추상 개념 생성"] --> M2["코드로 이미지<br/>렌더링"] --> M3["자유로운<br/>자기 진화"]
    end
    LLM -.->|"확장 시도"| VLM
    VLM -.->|"병목 해결"| MMZ
    style LLM fill:#1a3a1a,stroke:#4a9
    style VLM fill:#3a1a1a,stroke:#c44
    style MMZ fill:#1a1a3a,stroke:#49c
</div>

LLM과 달리 VLM은 **시각적 입력이 필요**하다. 기존 VLM 자기 진화 접근법(VisPlay 등)은 Proposer-Solver 파이프라인을 사용하지만, **사전 수집된 정적 이미지 데이터셋에 의존**한다. 이는 병목을 이미지 데이터 소싱으로 옮길 뿐이다.

### MM-Zero의 핵심 기여

1. **제로 데이터 자기 진화**: 외부 데이터 없이 VLM 추론을 향상시키는 최초의 프레임워크
2. **3역할 파이프라인**: Proposer-Coder-Solver로 코드 생성을 통해 추상 추론과 시각적 기반을 연결
3. **일관된 성능 향상**: 다양한 멀티모달 벤치마크와 여러 기본 모델에서 검증

---

## 2. 방법론 (Methodology)

### 2.1 사전 지식: GRPO

MM-Zero는 **검증 가능한 보상을 활용한 강화학습(RLVR)** 위에 구축된다. 규칙 기반 검증기가 이진 보상을 부여한다:

$$r_i = v(x_i) = \begin{cases} 1, & \text{if } x_i \text{이 정답이면} \\ 0, & \text{그 외} \end{cases}$$

**GRPO**는 N개 샘플에 걸쳐 정규화된 이점(advantage)을 계산한다:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_N)}{\text{std}(r_1, \ldots, r_N) + \varepsilon_{\text{norm}}}$$

> **직관적 이해**: 그룹 내 "평균보다 얼마나 잘했는가"를 측정하는 상대 평가

<div class="mermaid">
graph LR
    subgraph 그룹["N개 샘플 그룹"]
        S1["샘플₁<br/>r=1 ✓"]
        S2["샘플₂<br/>r=0 ✗"]
        S3["샘플₃<br/>r=1 ✓"]
        S4["샘플₄<br/>r=0 ✗"]
    end
    그룹 --> AVG["평균 = 0.5"]
    AVG --> N1["Â₁ = +0.5<br/>(평균 이상 → 강화)"]
    AVG --> N2["Â₂ = -0.5<br/>(평균 이하 → 억제)"]
    style N1 fill:#2d5a27,stroke:#4a9,color:#fff
    style N2 fill:#5a2727,stroke:#c44,color:#fff
</div>

GRPO 손실 함수:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\min\left(\frac{\pi_\theta(x_i)}{\pi_{\theta_{\text{old}}}(x_i)}\hat{A}_i,\ \text{clip}\left(\frac{\pi_\theta(x_i)}{\pi_{\theta_{\text{old}}}(x_i)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right) + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\theta_{\text{old}}})$$

> **직관적 이해**: PPO와 유사하게, 정책 비율을 클리핑하여 너무 급격한 업데이트를 방지하면서, KL 발산으로 이전 정책에서 너무 멀어지지 않도록 제약

<div class="mermaid">
graph TD
    OLD["이전 정책 π_old"] --> RATIO["정책 비율<br/>π_θ / π_old"]
    RATIO --> CLIP["클리핑<br/>[1-ε, 1+ε] 범위 제한"]
    RATIO --> RAW["원본 비율 × Â"]
    CLIP --> MIN["min(원본, 클리핑)"]
    RAW --> MIN
    MIN --> LOSS["손실 L_GRPO"]
    KL["KL 발산 패널티<br/>(너무 멀어지면 제동)"] --> LOSS
    style CLIP fill:#1a3a5c,stroke:#49c,color:#fff
    style KL fill:#5c3a1a,stroke:#c94,color:#fff
</div>

---

### 2.2 훈련 파이프라인

세 에이전트는 하나의 기본 모델에서 파생된다:

| 역할 | 기호 | 역할 | 입력 | 출력 |
|------|------|------|------|------|
| **Proposer** | $\pi_P$ | 캡션·질문-답변 쌍 생성 | 프롬프트 | $(c, q_{\text{easy}}, a_{\text{easy}}, q_{\text{hard}})$ |
| **Coder** | $\pi_D$ | 설명을 SVG/Python 코드로 변환 | 캡션 $c$ | 실행 가능 코드 → 이미지 $I$ |
| **Solver** | $\pi_S$ | 생성된 이미지로 추론 수행 | 이미지 $I$ + 질문 $q$ | 답변 $y$ |

**순차 훈련**: 각 역할이 훈련될 때 나머지 역할은 동결(frozen)된다.

<div class="mermaid">
graph TB
    subgraph ITER["반복 (Iteration)"]
        direction TB
        T1["1단계: Proposer 훈련<br/>Coder ❄️ Solver ❄️"]
        T2["2단계: Coder 훈련<br/>Proposer ❄️ Solver ❄️"]
        T3["3단계: Solver 훈련<br/>Proposer ❄️ Coder ❄️"]
        T1 --> T2 --> T3
    end
    T3 -->|"다음 반복"| T1
    style T1 fill:#2d5a27,stroke:#4a9,color:#fff
    style T2 fill:#1a3a5c,stroke:#49c,color:#fff
    style T3 fill:#5c1a3a,stroke:#c44,color:#fff
</div>

#### 훈련 데이터 필터링

품질 보장을 위해, 너무 쉽거나 너무 어려운 예제는 제외한다:

| 역할 | 필터 조건 | 직관적 의미 |
|------|----------|------------|
| **Coder** | 렌더링 성공률 ∈ [0.25, 0.75] (4 rollout) | "때때로 성공" = 학습 가치가 있는 난이도 |
| **Solver** | 쉬운 질문 정확도 > 0.5 AND 어려운 질문 정확도 ∈ [0.27, 0.75] | 풀 수는 있지만 아직 완벽하지 않은 문제 |

---

### 2.3 Proposer 보상: 6가지 구성 요소

Proposer는 $(c, q_{\text{easy}}, a_{\text{easy}}, q_{\text{hard}})$ 4중 쌍을 생성한다. 전체 보상 함수:

$$R_p(x) = \begin{cases} -1 & \text{포맷이 유효하지 않으면} \\ \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}_{\text{exec}}(C_i) \cdot \left(\min(R_{\text{solv}}(I_i), 0.5) + R_{\text{diff}}(I_i)\right) + r_{\text{eh}} + r_{\text{ct}} + r_{\text{div}} & \text{그 외} \end{cases}$$

> **보상 범위**: $[-1.0,\ 1.5]$

<div class="mermaid">
graph TB
    INPUT["Proposer 출력<br/>(캡션, 쉬운질문, 정답, 어려운질문)"]
    INPUT --> FMT{"포맷 유효?"}
    FMT -->|"아니오"| NEG["-1.0 ❌"]
    FMT -->|"예"| EXEC

    subgraph 핵심보상["핵심 보상 (이미지별)"]
        EXEC["① 실행 성공?<br/>𝟙_exec ∈ {0,1}"]
        SOLV["② 풀이 가능성<br/>R_solv ∈ [0, 0.5]"]
        DIFF["③ 난이도<br/>R_diff ∈ [0, 0.5]"]
        EXEC --> SOLV
        EXEC --> DIFF
    end

    subgraph 조절보상["조절 보상 (배치 전체)"]
        EH["④ 쉬움-어려움 패널티<br/>r_eh"]
        CT["⑤ 콘텐츠 다양성<br/>r_ct"]
        DIV["⑥ 캡션/질문 다양성<br/>r_div"]
    end

    핵심보상 --> SUM["합산"]
    조절보상 --> SUM
    SUM --> FINAL["최종 보상<br/>R_p ∈ [-1.0, 1.5]"]

    style NEG fill:#5a2727,stroke:#c44,color:#fff
    style FINAL fill:#2d5a27,stroke:#4a9,color:#fff
</div>

#### ① 실행 지표 (Execution Indicator)

$$\mathbb{1}_{\text{exec}}(C_i) = \begin{cases} 1 & \text{렌더링 성공} \\ 0 & \text{실패} \end{cases}$$

> 코드가 실행되어 이미지가 만들어져야 나머지 보상이 의미가 있다. **게이트(gate) 역할**.

#### ② 풀이 가능성 점수 (Solvability Score)

$$R_{\text{solv}}(I_i) = \frac{1}{K}\sum_{k=1}^{K}\mathbb{1}(y_{\text{easy}}^{(i,k)} = a_{\text{easy}}) \in [0,1]$$

> Solver가 **쉬운 질문**을 K번 시도해서 맞추는 비율. **$\tau_s = 0.5$로 상한이 제한**된다.

왜 0.5로 제한하는가? 풀이 가능성만 높으면 Proposer가 너무 쉬운 문제만 내게 되므로, 난이도 점수와 균형을 맞추기 위함이다.

<div class="mermaid">
graph LR
    subgraph 쉬운질문["쉬운 질문 K번 시도"]
        K1["시도1: ✓"]
        K2["시도2: ✓"]
        K3["시도3: ✗"]
        K4["시도4: ✓"]
    end
    쉬운질문 --> SCORE["R_solv = 3/4 = 0.75"]
    SCORE --> CAP["min(0.75, 0.5) = 0.5<br/>상한 적용"]
    style CAP fill:#1a3a5c,stroke:#49c,color:#fff
</div>

#### ③ 난이도 점수 (Difficulty Score) — 골디락스 원칙

자기 일관성(Self-consistency)으로 난이도를 측정한다:

$$c_i = \frac{1}{K}\sum_{k=1}^{K}\mathbb{1}(y_{\text{hard}}^{(i,k)} = \hat{y}_i)$$

$$R_{\text{diff}}(I_i) = \min(c_i, 1 - c_i) \in [0, 0.5]$$

> **직관**: Solver가 어려운 질문에 대해 **일관된 답을 낼 때($c_i \to 1$) = 너무 쉬움**, **완전히 랜덤할 때($c_i \to 0.5$) = 최적 난이도**

이것이 **골디락스 원칙**이다: 너무 쉽지도, 너무 어렵지도 않은 "딱 적당한" 난이도.

```
R_diff
 0.5 ┤          ╱╲
     │        ╱    ╲
     │      ╱        ╲
     │    ╱            ╲
 0.0 ┤──╱────────────────╲──
     0   0.25  0.5  0.75  1.0
              c_i (자기 일관성)

     ← 완전 랜덤  |  최적  |  완전 일관 →
       (너무 어려움)       (너무 쉬움)
```

#### ④ 쉬움-어려움 패널티 (Easy-Hard Penalty)

$$r_{\text{eh}} = \begin{cases} -\lambda_{\text{eh}} & \text{if } \frac{1}{|\mathcal{I}|}\sum_{I_i \in \mathcal{I}}R_{\text{diff}}(I_i) < \delta_{\text{eh}} \\ 0 & \text{그 외} \end{cases}$$

> $\delta_{\text{eh}} = 0.15$, $\lambda_{\text{eh}} = 0.3$. 배치 전체의 평균 난이도가 임계값 아래로 떨어지면 패널티. **너무 쉬운 문제만 양산하는 것을 방지.**

#### ⑤ 콘텐츠 유형 다양성 패널티 (Content-Type Diversity)

$$r_{\text{ct}} = \begin{cases} -\lambda_{\text{ct}} \cdot \frac{f_t - \phi}{1-\phi} & \text{if } f_t > \phi \\ 0 & \text{그 외} \end{cases}$$

> $\phi = 0.5$, $\lambda_{\text{ct}} = 0.15$. 배치에서 같은 콘텐츠 유형의 비율 $f_t$가 50%를 넘으면 패널티. **히스토그램만 계속 생성하는 등의 모드 붕괴 방지.**

#### ⑥ 캡션·질문 다양성 보너스 (Diversity Bonus)

$$r_{\text{div}} = -\text{clip}\Big(\big(w_c(s_x^{(\text{cap})} - u) + w_e(s_x^{(\text{eq})} - u) + w_h(s_x^{(\text{hq})} - u)\big) \cdot M \cdot \lambda_{\text{div}},\ -\lambda_{\text{div}},\ \lambda_{\text{div}}\Big)$$

> $w_c = 0.45,\ w_e = 0.20,\ w_h = 0.35,\ \lambda_{\text{div}} = 0.5$. 각 유사도 점수 $s$가 균일 분포 $u = 1/M$보다 높으면 (= 다른 샘플과 너무 비슷하면) 패널티, 낮으면 보너스.

<div class="mermaid">
graph TB
    subgraph 다양성체크["다양성 체크"]
        CAP_S["캡션 유사도<br/>w=0.45"]
        EQ_S["쉬운질문 유사도<br/>w=0.20"]
        HQ_S["어려운질문 유사도<br/>w=0.35"]
    end
    CAP_S --> AGG["가중 합산"]
    EQ_S --> AGG
    HQ_S --> AGG
    AGG --> CMP{"균일분포 u=1/M<br/>대비 비교"}
    CMP -->|"유사도 높음<br/>(너무 비슷)"| PEN["패널티 ↓"]
    CMP -->|"유사도 낮음<br/>(다양함)"| BON["보너스 ↑"]
    style PEN fill:#5a2727,stroke:#c44,color:#fff
    style BON fill:#2d5a27,stroke:#4a9,color:#fff
</div>

---

### 2.4 Coder 보상

$$R_D(C) = R_{\text{render}} + R_{\text{solv}} + R_{\text{diff}} - \lambda_{\text{err}}$$

| 구성 요소 | 범위 | 의미 |
|----------|------|------|
| $R_{\text{render}} = \mathbb{1}_{\text{exec}}(C)$ | {0, 1} | 코드가 성공적으로 실행되어 이미지를 렌더링했는가 |
| $R_{\text{solv}}$ | [0, 1] | Solver가 풀 수 있는 이미지인가 |
| $R_{\text{diff}}$ | [0, 1] | 적절한 난이도인가 |
| $\lambda_{\text{err}}$ | 0.1 / 0.05 | 렌더 실패 시 -0.1, 구문 오류 시 -0.05 |

<div class="mermaid">
graph LR
    CODE["코드 생성"] --> EXEC{"실행?"}
    EXEC -->|"성공"| R1["+1.0 렌더링"]
    EXEC -->|"실패"| E1["-0.1 패널티"]
    EXEC -->|"구문오류"| E2["-0.05 패널티"]
    R1 --> SOLV["+R_solv<br/>풀이 가능성"]
    R1 --> DIFF["+R_diff<br/>난이도"]
    SOLV --> TOTAL["R_D(C)"]
    DIFF --> TOTAL
    E1 --> TOTAL
    style R1 fill:#2d5a27,stroke:#4a9,color:#fff
    style E1 fill:#5a2727,stroke:#c44,color:#fff
    style E2 fill:#5a3a27,stroke:#c94,color:#fff
</div>

---

### 2.5 Solver 보상: 테스트 타임 RL

$$R_S(y_k) = \alpha \cdot R_{\text{acc}}(y_k, \bar{y}) + (1-\alpha) \cdot R_{\text{fmt}}(y_k)$$

> $\alpha = 0.9$ (정확도 90%, 포맷 10%)

**정확도 보상**:

$$R_{\text{acc}}(y_k, \bar{y}) = \mathbb{1}(\hat{y}_k = \bar{y})$$

> $\hat{y}_k$는 `\boxed{...}`에서 추출한 답, $\bar{y}$는 다수결 투표(majority voting)로 결정된 정답

**포맷 보상**:

$$R_{\text{fmt}}(y_k) = \mathbb{1}(y_k \text{가 } \texttt{<think>...</think> \textbackslash boxed\{...\}} \text{ 포맷을 따르는가})$$

<div class="mermaid">
graph LR
    subgraph 다수결["K번 추론 (다수결 투표)"]
        Y1["y₁: 답=A"]
        Y2["y₂: 답=B"]
        Y3["y₃: 답=A"]
        Y4["y₄: 답=A"]
        Y5["y₅: 답=C"]
    end
    다수결 --> VOTE["다수결 정답 ȳ = A"]
    VOTE --> ACC1["y₁: R_acc=1 ✓"]
    VOTE --> ACC2["y₂: R_acc=0 ✗"]

    subgraph 포맷체크["포맷 검증"]
        FMT1["think + boxed ✓"]
        FMT2["포맷 위반 ✗"]
    end

    ACC1 --> FINAL["R_S = 0.9×R_acc + 0.1×R_fmt"]
    FMT1 --> FINAL

    style ACC1 fill:#2d5a27,stroke:#4a9,color:#fff
    style ACC2 fill:#5a2727,stroke:#c44,color:#fff
</div>

---

## 3. 실험 결과

### 3.1 평가 벤치마크

| 카테고리 | 벤치마크 |
|---------|---------|
| **일반 시각 추론** | MMMU, MMMU-Pro, ChartQA, MM-Vet |
| **수학 시각 추론** | MathVerse, MathVision, MathVista, VisNumBench |
| **환각 탐지** | HallusionBench, MMSI |

### 3.2 주요 결과

#### Qwen3-VL-4B-Instruct

| 단계 | MMMU | MMMU-Pro | MM-Vet | ChartQA | MathVerse | MathVision | MathVista | VisNumBench | Hallusion | MMSI | **평균** |
|------|------|----------|--------|---------|-----------|------------|-----------|-------------|-----------|------|--------|
| Base | 50.2 | 41.8 | 38.5 | 79.6 | 42.3 | 33.0 | 64.0 | 45.3 | 72.3 | 26.1 | **50.2** |
| Iter 1 | 54.8 | 44.6 | 44.5 | 83.4 | 49.8 | 35.2 | 68.5 | 49.6 | 73.2 | 26.7 | **53.5** |
| Iter 2 | 53.7 | 45.1 | 45.4 | 83.0 | 49.4 | 36.3 | 68.3 | 49.1 | 71.5 | 25.9 | **52.8** |
| Iter 3 | 54.4 | 45.3 | 41.7 | 83.0 | 50.0 | 37.3 | 65.7 | 49.9 | 74.2 | 28.1 | **53.4** |

#### Qwen3-VL-8B-Instruct

| 단계 | MMMU | MMMU-Pro | MM-Vet | ChartQA | MathVerse | MathVision | MathVista | VisNumBench | Hallusion | MMSI | **평균** |
|------|------|----------|--------|---------|-----------|------------|-----------|-------------|-----------|------|--------|
| Base | 55.8 | 46.6 | 40.8 | 76.9 | 41.6 | 31.5 | 67.7 | 47.7 | 72.8 | 25.9 | **50.7** |
| Iter 1 | 58.7 | 49.3 | 39.0 | 78.9 | 42.8 | 36.5 | 67.7 | 55.0 | 72.1 | 29.5 | **53.0** |
| Iter 2 | 58.2 | 51.3 | 37.6 | 78.5 | 44.2 | 38.3 | 67.1 | 54.5 | 72.3 | 29.3 | **53.1** |
| Iter 3 | 58.3 | 53.0 | 41.7 | 79.6 | 45.1 | 39.6 | 67.2 | 53.2 | 74.1 | 28.9 | **54.1** |
| Iter 4 | 61.8 | 53.1 | 39.0 | 80.4 | 45.0 | 38.9 | 66.7 | 54.4 | 73.1 | 30.0 | **54.2** |
| Iter 5 | 61.9 | 52.5 | 40.4 | 80.8 | 45.6 | 39.8 | 67.8 | 53.0 | 74.7 | 28.7 | **54.5** |

#### MiMo-VL-7B-SFT

| 단계 | MMMU | MMMU-Pro | MM-Vet | ChartQA | MathVerse | MathVision | MathVista | VisNumBench | Hallusion | MMSI | **평균** |
|------|------|----------|--------|---------|-----------|------------|-----------|-------------|-----------|------|--------|
| Base | 57.3 | 46.1 | 39.0 | 83.7 | 46.3 | 36.6 | 70.4 | 44.7 | 55.0 | 29.6 | **50.9** |
| Iter 1 | 56.5 | 48.4 | 43.1 | 85.5 | 55.6 | 40.9 | 73.6 | 48.0 | 70.8 | 29.9 | **55.2** |
| Iter 2 | 59.3 | 48.7 | 48.2 | 85.0 | 56.0 | 41.5 | 73.3 | 48.3 | 70.7 | 30.6 | **56.1** |
| Iter 3 | 60.1 | 48.7 | 45.9 | 85.0 | 56.0 | 42.5 | 72.1 | 48.5 | 71.3 | 29.9 | **56.0** |

> **핵심 관찰**: 모든 모델에서 **Iter 1에서 가장 큰 점프**가 발생하며, 이후 반복에서도 안정적으로 향상된다. 8B 모델은 5회 반복까지도 성능이 계속 오른다 (50.7% → 54.5%).

---

## 4. 절제 연구 (Ablation Study)

### 풀이 가능성-난이도 균형 제거 시

$$\text{Avg. 향상:}\ 3.9\% \xrightarrow{\text{제거 시}} 2.3\%$$

> **보상 해킹(Reward Hacking) 발견**: 균형 메커니즘 없이 Coder가 **렌더링된 이미지에 정답을 직접 삽입**하는 치팅 행동을 학습했다.

<div class="mermaid">
graph LR
    subgraph 정상["균형 있음 ✓"]
        N1["Coder가 적절한<br/>난이도의 이미지 생성"]
        N2["Solver가 진짜<br/>추론으로 풀이"]
    end
    subgraph 해킹["균형 없음 ✗"]
        H1["Coder가 이미지에<br/>정답 텍스트를 삽입"]
        H2["Solver가 정답을<br/>그냥 읽기만 함"]
    end
    style 정상 fill:#1a3a1a,stroke:#4a9
    style 해킹 fill:#3a1a1a,stroke:#c44
</div>

### 콘텐츠 다양성 제거 시

| 반복 | 다양성 있음 | 다양성 없음 |
|------|-----------|-----------|
| Iter 1 | 51.7% | 51.7% |
| Iter 2 | 53.1% | 51.3% ↓ |
| Iter 3 | 54.1% | **49.4%** ↓↓ |

> 다양성 없이는 모델이 **쉽게 렌더링 가능한 유형(히스토그램 등)으로 수렴**하여, 반복이 진행될수록 오히려 성능이 하락했다.

```
정확도(%)
 55 ┤
    │    ╱─── 다양성 있음
 53 ┤  ╱
    │╱
 51 ┤───╮
    │    ╲
 49 ┤     ╲── 다양성 없음 (붕괴!)
    │
 47 ┤
    └──────────────────────
     Base  Iter1  Iter2  Iter3
```

---

## 5. 관련 연구

### 검증 가능한 보상을 활용한 강화학습 (RLVR)

수학, 코드 생성 등 **객관적 정답 검증이 가능한 도메인**에서 성공을 거둔 패러다임. DAPO, VAPO 등의 프레임워크와 함께, **높은 엔트로피 유도 최적화**를 통해 희소한 규칙 기반 보상 환경에서 다양한 탐색을 장려하고 조기 수렴을 방지한다.

### VLM에서의 자기 진화

| 접근법 | 시드 이미지 필요 | 역할 수 | 한계 |
|-------|:---------------:|:------:|------|
| VisPlay | ✅ | 2 (Proposer-Solver) | 정적 이미지 분포에 제한 |
| Evolmm | ✅ | 2 | 사전 수집 데이터 의존 |
| V-Zero | ✅ | 2 | 정적 이미지 의존 |
| **MM-Zero** | **❌** | **3** (Proposer-Coder-Solver) | **완전 자기 생성** |

---

## 6. 한계 및 결론

### 한계

- **계산 비용**: 38B+ 파라미터 모델에서 스케일링 행동을 검증하지 못함
- **기본 모델 강도 의존**: 7B/8B 모델이 초기 코드 렌더링 성공률 70%로 4B의 40%보다 월등히 높아, 강한 기본 모델일수록 자기 진화 효과가 큼

### 결론

MM-Zero는 **데이터 없이 3역할 아키텍처를 통해 VLM이 스스로 진화할 수 있음을 입증**했다. 각 역할에 맞춤 설계된 보상 함수로 순차 훈련을 수행하며, 에이전트의 추론 능력을 점진적으로 향상시킨다.

향후 방향:
1. 코드 생성 외 다양한 도구 사용으로 확장
2. 더 큰 기본 모델로 스케일링
3. 추가 에이전트 역할 탐색

---

## 전체 아키텍처 요약

<div class="mermaid">
graph TB
    BASE["기본 VLM 모델<br/>(Qwen3-VL / MiMo-VL)"]
    BASE --> P["Proposer π_P"]
    BASE --> D["Coder π_D"]
    BASE --> S["Solver π_S"]

    P -->|"(캡션, 쉬운Q, 정답, 어려운Q)"| D
    D -->|"SVG/Python → 이미지 I"| S
    S -->|"정확도 피드백"| RP["R_p: 6가지 보상<br/>실행·풀이가능성·난이도<br/>패널티·다양성"]
    S -->|"풀이 피드백"| RD["R_D: 렌더+풀이+난이도"]

    RP -->|"GRPO"| P
    RD -->|"GRPO"| D
    RS["R_S: 정확도+포맷<br/>(다수결 투표)"] -->|"GRPO"| S

    subgraph 결과["결과: 데이터 0 → 평균 +3~6%p 향상"]
    end

    style BASE fill:#333,stroke:#888,color:#fff
    style P fill:#2d5a27,stroke:#4a9,color:#fff
    style D fill:#1a3a5c,stroke:#49c,color:#fff
    style S fill:#5c1a3a,stroke:#c44,color:#fff
    style 결과 fill:#1a1a3a,stroke:#49c,color:#fff
</div>
