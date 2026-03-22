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

RLVR(Reinforcement Learning with Verifiable Rewards)은 모델 출력의 정확성을 검증할 수 있는 도메인에 적용 가능한 VLM 훈련 패러다임이다. 규칙 기반 검증기 $v: X \to \{0, 1\}$가 각 생성물 $x_i$에 이진 보상을 부여한다:

$$r_i = v(x_i) = \begin{cases} 1, & \text{if } x_i \text{이 정답이면} \\ 0, & \text{그 외} \end{cases}$$

이러한 검증 가능한 보상은 수학이나 코드 생성 등 정확성을 객관적으로 평가할 수 있는 추론 집약적 태스크에 특히 효과적이다. 정확성 외에도 답변 다양성이나 원하는 출력 포맷 준수 등의 추가 기준도 인코딩할 수 있다.

본 논문은 **GRPO(Group Relative Policy Optimization)**를 채택한다. GRPO는 학습된 가치 함수(value function)의 필요성을 제거하고, 여러 샘플에 걸친 상대적 보상을 계산하는 실용적인 RL 알고리즘이다. 프롬프트 $p$가 주어지면, 현재 정책이 N개의 응답과 해당 보상을 생성하고, 이 보상을 그룹 내에서 정규화하여 응답 수준의 이점(advantage)을 산출한다:

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

> **직관적 이해**: 양의 상대적 이점을 가진 응답을 상향 조정하면서, 과도한 정책 이탈을 패널티로 제한함으로써, RLVR 프레임워크 하에서 VLM의 추론 및 생성 품질을 향상시키는 효과적인 메커니즘을 제공한다.

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

MM-Zero는 추가 데이터 없이 동일한 기본 모델에서 진화한 세 개의 독립적 모델 에이전트를 포함한다. 각 모델은 역할별 보상 함수를 사용해 GRPO로 순차 최적화되며, 폐쇄 훈련 루프를 형성한다. **각 역할의 훈련 단계에서 나머지 두 역할은 동결(frozen)**된다.

#### Proposer 상세

Proposer는 다양한 시각 도메인에 걸쳐 **시각적 캡션과 질문-답변 쌍**을 생성하도록 훈련된다. 대상 도메인에는 차트 이해, 객체/도형 인식, OCR, 시각적 수학 추론이 포함된다.

Proposer의 보상을 계산하기 위해, Coder와 Solver 양쪽에 대한 **vLLM 서비스 포트를 초기화**한다. Coder가 생성된 캡션과 질문을 받아 SVG 코드를 생성하고, 이를 **병렬로 렌더링하여 PNG 이미지로 변환**한다. 성공적으로 렌더링된 이미지와 관련 질문은 Solver에 전달되어 답변을 생성한다. Solver가 Proposer의 보상을 계산한다. 계산 비용 절감을 위해 Coder는 **rollout 크기 4**, Solver는 **rollout 크기 5**를 사용한다.

#### Coder 상세

가장 최근의 Proposer 체크포인트가 **약 4,000개의 캡션 및 질문-답변 쌍**을 생성하며, 이것이 Coder의 훈련 데이터가 된다. Coder는 캡션으로부터 **이미지를 렌더링하는 SVG 코드를 생성**하도록 훈련된다. 렌더링된 이미지는 Solver에 전송되어 보상을 계산한다.

#### Solver 상세

Solver의 훈련 데이터는 가장 최근의 Proposer와 Coder 체크포인트를 사용하여 구성된다. Proposer가 제안을 생성하고, Coder가 이를 이미지로 렌더링한다. **성공적으로 렌더링된 이미지와 관련 질문만** 유지하여 Solver의 훈련 세트로 사용한다.

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

### 2.3 보상 설계 개요

이 절에서는 세 역할 각각에 대한 보상 설계를 설명한다.

---

### 2.4 Proposer 보상: 6가지 구성 요소

Proposer 정책 $\pi_P$는 **4중 쌍(Quadruple)** $x = (c, q\_{\text{easy}}, a\_{\text{easy}}, q\_{\text{hard}})$를 생성한다:

| 요소 | 의미 |
|------|------|
| $c$ | 시각 장면의 **세밀한 텍스트 설명(캡션)** |
| $q\_{\text{easy}}$ | 캡션이 이미지로 렌더링되면 답이 명확해지는 **쉬운 질문** (렌더링 품질 검증용) |
| $a\_{\text{easy}}$ | 쉬운 질문의 **정답** |
| $q\_{\text{hard}}$ | 렌더링된 이미지에 대해 **다단계 추론이 필요한 어려운 질문** (Solver 훈련·평가용) |

훈련을 통해 Proposer는 더 풍부한 캡션과 더 어려운 질문을 생성하는 법을 학습하여, Coder가 더 정보적인 렌더링을 생성하고 Solver가 더 깊은 추론을 수행하도록 압박한다.

$x$를 평가하기 위해, 포맷 유효성, 풀이 가능성, 난이도를 Solver $\pi_S$를 통해 검증하는 **다단계 계층적 보상 메커니즘**을 사용한다. 전체 보상 함수:

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

### 2.5 Coder 보상: 실행과 유효성

Coder $\pi_D$는 캡션 $c$를 받아 코드 $C$를 생성한다. 목표는 Proposer의 의도를 충실히 나타내는 **실행 가능한 코드**를 생성하는 것이다. 보상 $R_D$는 실행 상태, 의미적 정확성, 태스크 실현 가능성의 가중합이다:

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

### 2.6 Solver 보상: 테스트 타임 강화학습 (TTRL)

Solver 정책 $\pi_S$는 Proposer가 생성한 어려운 질문 $(I, q\_{\text{hard}})$에 대해 훈련된다. 이 생성된 질문에는 **정답 레이블이 없으므로**, Test-Time Reinforcement Learning(TTRL)을 다수결 투표 방식으로 사용한다.

주어진 입력에 대해 Solver가 **K개의 독립적인 추론 경로**를 생성한다. 다수결 투표로 실버 답변을 식별한다: $\bar{y} = \text{Mode}(\{y_1, \ldots, y_K\})$. $k$번째 응답의 보상은 답변 정확도와 구조적 유효성의 가중합이다:

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

결과는 **Qwen-2.5-14B-Instruct를 LLM-as-a-Judge 평가자**로 사용한다.

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

**교차 모델 일반화**: 4B 모델은 더 작은 향상(3%)을 보이는데, 이는 초기 이미지 렌더링 성공률이 낮기 때문이다(~40% vs 7B/8B의 70%).

**렌더링 진행**: 훈련 전반에 걸쳐 Coder의 렌더링 성공률과 이미지 풀이 가능성이 **꾸준히 향상**되며, 이는 점진적으로 더 충실한 시각적 생성이 이루어짐을 나타낸다.

### 3.3 논의 (Discussion)

사례 연구를 통해 반복별 시각적 개선 과정을 관찰할 수 있다:

| 반복 | 시각적 품질 | 질문 난이도 |
|------|-----------|-----------|
| **초기** | 겹치는 요소가 있는 어수선한 레이아웃 | 단순 |
| **Iter 1** | 구성이 개선되나, 답이 주석으로 이미지에 포함됨 | 중간 |
| **Iter 2** | 더 깔끔한 레이아웃, 다단계 추론 요구 | 높음 |
| **Iter 3** | 세련된 그래프, 값 추출 후 백분율 적용 등 **진정한 구성적 추론** 필요 | 매우 높음 |

<div class="mermaid">
graph LR
    subgraph I0["초기"]
        A1["어수선한 레이아웃<br/>겹치는 요소"]
    end
    subgraph I1["Iter 1"]
        B1["구성 개선<br/>but 답이 이미지에 포함"]
    end
    subgraph I2["Iter 2"]
        C1["깔끔한 레이아웃<br/>다단계 추론"]
    end
    subgraph I3["Iter 3"]
        D1["세련된 그래프<br/>구성적 추론 필요"]
    end
    I0 --> I1 --> I2 --> I3
    style I0 fill:#5a2727,stroke:#c44
    style I1 fill:#5a3a27,stroke:#c94
    style I2 fill:#2d4a27,stroke:#4a9
    style I3 fill:#1a3a5c,stroke:#49c
</div>

---

## 4. 절제 연구 (Ablation Study)

### 풀이 가능성-난이도 균형 제거 시

Proposer 보상에는 $\min(R\_{\text{solv}}, 0.5) + R\_{\text{diff}}$가 포함되어 있으며, 풀이 가능성에 상한을 두어 사소한 인스턴스를 방지한다. 이 상한을 제거하면:

$$\text{Avg. 향상:}\ 3.9\% \xrightarrow{\text{제거 시}} 2.3\%$$

상한 없이는 **불균형한 보상 신호**가 발생하여, 난이도를 무시하고 풀이 가능성에 대해 불균형적으로 최적화하게 된다.

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

> $r\_{\text{ct}}$ 항은 시각적 유형 간 다양성을 장려한다. 다양성을 제거하면, 분석 결과 모델이 렌더링하기 쉬운 **좁은 시각적 유형 부분 집합(예: 히스토그램)**으로 수렴하여, **좁은 시각적 문제 유형에 과적합(overfitting)**하는 결과를 보였다.

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

---

## 부록 A: 프롬프트 템플릿

### A.1 Proposer 프롬프트

```
역할: 당신은 SVG를 사용하여 풍부하고 복잡한 데이터 시각화 및 다이어그램 사양을
설계하는 전문 Visual Content Designer입니다. 시각적으로 흥미롭고, 데이터가
밀집되어 있으며, 해석에 진정한 추론이 필요한 시각화를 설계하는 것이 목표입니다.
```

**출력 형식**: 정확히 6개의 XML 블록:

| 필드 | 설명 |
|------|------|
| `<content_type>` | `data_chart`, `diagram`, `geometry`, `timeline`, `map`, `table`, `other` 중 하나 |
| `<caption>` | 풍부하고 상세한 시각화 사양 |
| `<easy_question>` | 이미지에서 직접 읽을 수 있는 단순 질문 |
| `<easy_answer>` | 쉬운 질문의 정답 |
| `<hard_question>` | 다단계 추론이 필요한 도전적 질문 |
| `<hard_answer>` | 어려운 질문의 정답 |

**복잡도 요구사항**: 캡션에는 다음 중 최소 3가지 포함:
- 다중 데이터 시리즈, 주석, 보조 패널, 색상/마커, 파생 값, 비자명한 패턴, 기하학적 구성

**어려운 질문 제약**: 다단계 추론을 요구하며, 시각화에서 최소 하나의 값을 추출해야 함. 질문 자체에 모든 값을 명시하지 않아야 한다.

**답변 형식**: 단일 숫자, 단어, 또는 짧은 구문 (예: "42", "Q1", "blue")

### A.2 CodeGen 프롬프트

```
당신은 데이터 시각화를 위한 SVG 코드 생성기입니다.
차트 설명(캡션)과 질문-답변이 주어집니다.

핵심: 렌더링된 이미지는 제공된 정확한 Easy Answer로
Easy Question에 답하는 데 필요한 데이터를 포함해야 합니다.

<svg ...>로 시작하는 순수 SVG 마크업을 작성하세요.
Python 코드를 작성하지 마세요.
```

**SVG 가이드라인**:
- `viewBox` 사용
- `<text>`로 레이블 표시
- font-size ≥ 12px 유지
- 구분되는 색상 적용
- 자체 완결적 코드

### A.3 Solver 프롬프트

```
이미지를 주의 깊게 보고 질문에 답하세요.
먼저, <think>...</think> 태그 안에서 단계별로 생각하세요.
그런 다음, \boxed{} 안에 최종 답을 넣으세요.
단일 숫자, 단어, 또는 짧은 구문만 (예: \boxed{42}, \boxed{blue})
— 단위, 완전한 문장 없이.
```

### A.4 LLM-as-a-Judge 프롬프트

Qwen2.5-14B-Instruct가 벤치마크 전반의 모델 출력을 평가한다.

```
시스템: 당신은 답변 정확성 판단자입니다.
질문, 정답(gold), 모델 답변이 주어지면,
모델 답변이 정확한지 판단하세요.

수치 동등성(14 vs 14.0), 선택지 동등성(A vs A.),
의역(paraphrase)을 고려하세요.

정확히 한 단어로 답하세요: Yes 또는 No.
```

---

## 부록 B: 훈련 설정

### 메인 스크립트 파라미터 (Table 4)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| GPU\_MEM | 80 | GPU 메모리 (GB) |
| TRAIN\_STEPS | 10 | 모델당 반복당 훈련 스텝 |
| NUM\_ITER | 3 | 자기 진화 반복 횟수 |
| PROP\_PER\_GPU | 413 | GPU당 제안 수 (Proposer/CodeGen) |
| PROP\_PER\_S | 625 | GPU당 제안 수 (Solver) |
| PROP\_ROLL\_BS | 18 | Proposer rollout 배치 크기 |
| GLOBAL\_BS\_P | 18 | Proposer 전역 배치 크기 |
| ROLLOUT\_N | 4 | 제안당 rollout 수 (Proposer/CodeGen) |
| SOLVER\_N\_R | 5 | Solver rollout 수 |
| ROLLOUT\_BS | 320 | CodeGen rollout 배치 크기 |
| ROLL\_BS\_S | 512 | Solver rollout 배치 크기 |
| GLOBAL\_BS\_S | 64 | Solver 전역 배치 크기 |

### 역할별 설정 (Table 5) — 80GB GPU 기준

| 설정 | Proposer | CodeGen | Solver |
|------|----------|---------|--------|
| GPU 수 | 3 | 4 | 8 |
| 최대 프롬프트 길이 | 4096 | 4096 | 8192 |
| 최대 응답 길이 | 2048 | 4096 | 4096 |
| Rollout 배치 크기 | 18 | 256 | 512 |
| 전역 배치 크기 | 18 | 64 | 64 |
| 학습률 | $1 \times 10^{-6}$ | $1 \times 10^{-6}$ | $1 \times 10^{-6}$ |
| 가중치 감쇠 | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ |
| Rollout 온도 | 1.0 | 0.7 | 1.0 |
| Rollout top\_p | 0.99 | 0.95 | 0.99 |
| Rollout n | 4 | 4 | 8 |
| Tensor parallel | 1 | 1 | 2 |

세 모델 모두 **전체 파인튜닝(full fine-tuning)**을 사용하며 LoRA rank는 0이다. **비전 타워(vision tower)는 훈련 가능** 상태를 유지한다. 총 훈련 가능 파라미터: 약 **8.77B** (Qwen3-VL-8B-Instruct 기준).

<div class="mermaid">
graph LR
    subgraph GPU할당["GPU 할당 (80GB 기준)"]
        P_GPU["Proposer<br/>3 GPU"]
        C_GPU["CodeGen<br/>4 GPU"]
        S_GPU["Solver<br/>8 GPU"]
    end
    subgraph 온도["Rollout 온도 전략"]
        P_T["Proposer: 1.0<br/>(최대 다양성)"]
        C_T["CodeGen: 0.7<br/>(안정적 코드)"]
        S_T["Solver: 1.0<br/>(탐색 장려)"]
    end
    style P_GPU fill:#2d5a27,stroke:#4a9,color:#fff
    style C_GPU fill:#1a3a5c,stroke:#49c,color:#fff
    style S_GPU fill:#5c1a3a,stroke:#c44,color:#fff
</div>

---

## 부록 C: 렌더링 파이프라인

SVG 전용 렌더링 변형은 생성된 SVG 마크업을 Solver 입력용 PNG 이미지로 변환한다:

<div class="mermaid">
graph LR
    SVG["SVG 문자열"] -->|"cairosvg"| PNG["PNG 변환"]
    PNG --> VAL{"유효성 검증"}
    VAL -->|"종횡비 ≤ 100<br/>최대 차원 ≤ 16384px"| OK["Base64 인코딩<br/>→ vLLM & Solver"]
    VAL -->|"무효"| DISCARD["폐기"]
    TIMEOUT["타임아웃: 30초/스니펫"] -.-> PNG
    PARALLEL["ProcessPoolExecutor<br/>(병렬 워커)"] -.-> PNG
    style OK fill:#2d5a27,stroke:#4a9,color:#fff
    style DISCARD fill:#5a2727,stroke:#c44,color:#fff
</div>

| 파라미터 | 값 |
|---------|-----|
| 변환 라이브러리 | cairosvg |
| 스니펫당 타임아웃 | 30초 |
| 병렬 처리 | ProcessPoolExecutor |
| 최대 종횡비 | 100 |
| 최대 차원 | 16384 px |
| 출력 형식 | Base64 인코딩 PNG |
