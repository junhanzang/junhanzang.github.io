---
title: "Less is Enough: LLM의 특성 공간에서 다양한 데이터 합성하기"
date: 2026-02-19 15:00:00
categories:
  - 인공지능
tags:
  - LLM
  - 데이터 합성
  - Sparse Autoencoder
  - Post-training
  - SFT
  - 논문 리뷰
---

<https://arxiv.org/abs/2602.10388>

**Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs**
Zhongzhi Li\*, Xuansheng Wu\*, Yijiang Li, Lijie Hu, Ninghao Liu

University of Georgia / UC San Diego / Mohamed bin Zayed University of AI / The Hong Kong Polytechnic University
\* 동등 기여(Equal contribution)

코드: <https://github.com/Zhongzhi660/FAC-Synthesis>

---

## 초록

본 논문은 대형 언어 모델(LLM)에서 데이터 다양성 문제를 해결하기 위해 **Feature Activation Coverage(FAC)**를 제안한다. FAC는 해석 가능한 특성 공간 내에서 다양성을 측정하는 지표이다. 또한 이를 기반으로 한 **FAC Synthesis** 프레임워크를 제안하는데, 이는 희소 오토인코더(Sparse Autoencoder, SAE)를 활용하여 시드 데이터셋에서 누락된 태스크 관련 특성을 식별하고 이 공백을 채우는 합성 샘플을 생성한다.

주요 결과: FAC와 다운스트림 태스크 성능 사이에는 강한 상관관계(피어슨 r=0.95)가 있으며, 고작 **2,000개의 샘플**만으로 MAGPIE(300,000개 샘플 사용)에 필적하는 성능을 달성한다(150배 더 효율적).

![지시 따르기 데이터셋의 효율성 프론티어](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x1.png)

---

## 1. 서론

데이터 다양성은 지도 미세조정(SFT)과 강화 학습을 통한 LLM 포스트 트레이닝에 매우 중요하다. 기존의 균일 샘플링 전략은 롱테일 샘플 수집에 어려움을 겪는다.

**핵심 연구 질문:** "어떻게 원칙적이고 효율적인 방식으로 다양한 포스트 트레이닝 데이터셋을 구성할 수 있는가?"

### 기존 방법의 한계

기존 다양성 지표들은 텍스트 또는 일반 임베딩 공간에서 동작한다:
- **단어 수준 변형**: Distinct-n
- **구문 수준 변형**: POS 태그 Distinct-2
- **임베딩 기반**: 코사인 거리, 시맨틱 엔트로피

이들은 성능을 이끄는 태스크 관련 특성을 측정하지 못한다. 그래디언트 기반 대안은 아키텍처 간 전이 가능성이 부족하다.

### 핵심 기여

- **이론적 프레임워크:** 포스트 트레이닝 일반화 오류의 상한을 유도하여 태스크 관련 특성 커버리지를 핵심 성능 요소로 식별
- **FAC 지표:** LLM 내부 특성 공간에서 태스크 관련 특성의 커버리지를 정량화하는 모델 인식 다양성 측정
- **FAC Synthesis 프레임워크:** 누락된 특성의 자동 식별 및 목표 지향 합성 샘플 생성

**핵심 성과:** FAC Synthesis는 MAGPIE보다 150배 적은 데이터(2,000개 vs 300,000개)로 동등한 성능을 달성한다.

---

## 2. 관련 연구

데이터 다양성은 LLM 포스트 트레이닝 효과에 크게 영향을 미친다. 기존 다양성 지표(distinct-n, N-gram, 임베딩 코사인 거리, 시맨틱 엔트로피)는 텍스트 또는 일반 공간에서 동작하여 태스크 관련 잠재 특성을 포착하지 못한다.

현재 LLM 기반 합성 방법은 단순 프롬프팅, 진화적 접근법, 추론 트레이스, 자기 부트스트랩 파이프라인에 의존하여 본질적으로 중복과 분포 편향을 만든다.

**희소 오토인코더(SAE)**는 희소하고 해석 가능한 특성 공간을 구성하여 다양성 측정과 커버리지 유도 합성을 가능하게 한다.

---

## 3. 예비 지식

### 희소 오토인코더(SAE)

SAE는 LLM 내부 활성화에서 해석 가능한 특성을 추출한다. 입력 임베딩 $\mathbf{x} \in \mathbb{R}^d$가 주어지면:
- **인코더:** $\mathbf{z} = \sigma(\mathbf{x}W) \in \mathbb{R}^k$
- **디코더:** $\hat{\mathbf{x}} = \mathbf{z}W^T \in \mathbb{R}^d$
- **학습 목적함수:**

$$\mathcal{L}_{\text{SAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2_2 + \lambda\|\mathbf{z}\|_1$$

여기서 $\sigma$는 ReLU 활성화, $W \in \mathbb{R}^{d \times k}$이며 $k \gg d$이다. $k$개 차원 각각은 태스크 관련 가능성이 있는 별개의 잠재 패턴을 포착한다.

---

## 4. 합성 데이터의 일반화 정량화

### 정리 4.1 (일반화 오류 상한)

경계 손실 $\ell$ (경계 $C$)로 최적화된 포스트 학습 모델 $\pi$와 i.i.d. 합성 데이터셋 $S\_{\text{gen}}$이 주어지면:

$$\text{Err}(\pi^{S_{\text{gen}}}) \leq 2C \cdot \Delta_{\text{TV}}(\mathcal{D}, \mathcal{D}_{\text{gen}}) + |R_{\mathcal{D}_{\text{gen}}}(\pi^{S_{\text{gen}}}) - \hat{R}_{S_{\text{gen}}}(\pi^{S_{\text{gen}}})|$$

이 상한은 두 항으로 구성된다:

1. **분포 격차:** $\Delta\_{\text{TV}}(\mathcal{D}, \mathcal{D}\_{\text{gen}})$는 태스크 도메인과 합성 분포 사이의 발산을 측정하며, 합성 파이프라인 설계에 영향을 받는다.

2. **샘플링 오류:** 기대 위험 $R\_{\mathcal{D}\_{\text{gen}}}(\pi^{S\_{\text{gen}}}) = \mathbb{E}\_{x \sim \mathcal{D}\_{\text{gen}}}[\ell(\pi^{S\_{\text{gen}}}, x)]$와 유한 샘플에서의 경험적 위험 $\hat{R}\_{S\_{\text{gen}}}(\pi^{S\_{\text{gen}}}) = \frac{1}{n\_g}\sum \ell(\pi^{S\_{\text{gen}}}, x\_i)$ 간의 차이

---

## 5. 특성 공간에서의 분포 격차 감소

### 5.1 형식화

입력 시퀀스 $X \sim \mathcal{D}$를 LLM이 처리하여 토큰 임베딩 $\mathbf{X} \in \mathbb{R}^{T \times d}$를 생성한다. SAE와 최대 풀링 집계를 통해 $\mathbf{Z} = g(\mathbf{X}) \in \mathbb{R}^k$를 얻는다. 마찬가지로 $X\_{\text{gen}} \sim \mathcal{D}\_{\text{gen}}$에 대해 $Z\_{\text{gen}} = g(X\_{\text{gen}})$을 얻는다.

Pinsker 부등식과 KL 발산 체인 규칙을 적용하면:

$$\Delta_{\text{TV}}(\mathcal{D}, \mathcal{D}_{\text{gen}}) \leq \sqrt{\frac{1}{2}\left(\Delta_{\text{KL}}(P_Z \| Q_Z) + \varepsilon_{\text{cond}}\right)}$$

분포 격차는 SAE 특성 공간에서의 KL 발산과 최적화 불가능한 항 $\varepsilon\_{\text{cond}}$로 상한이 정해진다. 따라서 합성 목적함수는 다음과 같다:

$$S^*_{\text{gen}} = \arg\min_{S_{\text{gen}}} \Delta_{\text{KL}}(P_Z \| Q_Z)$$

이는 특성 분포 $Q\_Z$가 목표 도메인 $P\_Z$와 최대한 가까운 합성 데이터를 추구한다.

### 5.2 구현

KL 발산을 그래디언트 방법으로 직접 최소화하는 것은 $Q\_Z$가 데이터셋 $S\_{\text{gen}}$에 의존하기 때문에 불가능하다. 대신 SAE 특성 공간에서 샘플과 앵커 코퍼스 $\mathcal{S}\_{\text{anchor}}$ 간의 특성 활성화를 매칭하여 $S\_{\text{gen}}$을 구성한다.

**이진 활성화 정의:**

$$\mathcal{A}_i(x) = \mathbf{1}[g_i(x) > \delta]$$

$\mathcal{A}\_i(x) = 1$이면 특성 $i$ 활성화됨을 나타낸다.

**태스크 관련 특성:** 집합 $F \subset \{1, \ldots, k\}$는 LLM(예: GPT-4o-mini)을 사용하여 식별된다.

**활성 특성 부분집합:**
- $F(P\_Z) = \{i \in F \mid \Pr\_{x \sim \mathcal{S}\_{\text{anchor}}}(\mathcal{A}\_i(x) = 1) > 0\}$
- $F(Q\_Z) = \{i \in F \mid \Pr\_{x \sim S\_{\text{gen}}}(\mathcal{A}\_i(x) = 1) > 0\}$

**Feature Activation Coverage (FAC):**

$$\text{FAC} = \frac{|F(Q_Z)|}{|F(P_Z)|}$$

**누락된 특성 집합:**

$$F_{\text{miss}} = F(P_Z) \setminus F(Q_Z)$$

$P\_Z$와 $Q\_Z$ 사이의 분포 격차를 줄이기 위해 $i \in F\_{\text{miss}}$에 해당하는 특성 활성화를 목표로 하는 샘플을 합성한다.

---

## 6. 합성 분포 $\mathcal{D}\_{\text{gen}}$ 하의 샘플링 오류 감소

### 6.1 형식화

잘 정렬된 합성 분포 $\mathcal{D}\_{\text{gen}}$을 가지더라도 유한한 데이터셋 $S\_{\text{gen}}$은 학습 목적함수를 불완전하게 추정할 수 있다. PAC-베이지안 이론은 이 오류를 상한으로 묶는다.

### 보조 정리 6.1 (샘플링 오류 상한)

Sub-Gamma 손실 가정 하에서, 샘플링 오류는 상호 정보량 $I(S\_{\text{gen}}; W)$를 통해 상한이 정해진다:

$$\mathbb{E}\left[|R_{\mathcal{D}_{\text{gen}}}(\pi^{S_{\text{gen}}}) - \hat{R}_{S_{\text{gen}}}(\pi^{S_{\text{gen}}})|  \right] \leq \sqrt{\frac{2\sigma^2}{n} I(S_{\text{gen}}; W)} + \frac{c}{n} I(S_{\text{gen}}; W)$$

$$\leq \sqrt{\frac{2\sigma^2}{n} H(S_{\text{gen}})} + \frac{c}{n} H(S_{\text{gen}})$$

모델이 완전히 암기할 때($H(S\_{\text{gen}} \mid W) = 0$), 등호가 성립하고 상한은 오로지 $S\_{\text{gen}}$의 엔트로피에만 의존한다. $H(S\_{\text{gen}})$을 통한 합성 데이터셋의 불확실성 감소가 샘플링 오류 감소에 핵심적이다.

### 6.2 구현

단순 프롬프트를 통한 순진한 생성은 특성 제어가 불충분하여 높은 가변성과 불확실성을 만든다. 따라서 두 단계 합성 전략을 제안한다:

**1단계: 대조 쌍 구성**

각 누락된 특성 $i \in F\_{\text{miss}}$에 대해 대조 쌍 $(x^+\_i, x^-\_i)$를 구성한다:
- $x^+\_i$: 해당 특성을 강하게 활성화
- $x^-\_i$: 해당 특성을 약하게 활성화

특성 인식 프롬프트 $\mathcal{T}(\text{Desc}\_i)$를 사용하여 후보를 생성하고 SAE 활성화 $g\_i(x)$로 점수를 매긴다. $g\_i(x) \geq \delta$인 것을 $x^+\_i$로, 약한 활성화를 $x^-\_i$로 식별하여 대조 쌍을 구성한다.

**2단계: 특성 커버 샘플 합성**

대조 쌍 $(x^+\_i, x^-\_i)$를 사용하여 합성 프롬프트 $\mathcal{T}^{\text{ctr}}\_i(x^+\_i, x^-\_i; \text{Desc}\_i)$를 구성한다. 생성 모델 $\mathcal{M}$에서 $m$개 후보를 샘플링한다:

$$\tilde{S}_i = \{x_{i,1}, \ldots, x_{i,m}\}, \quad x_{i,j} \sim \mathcal{M}(\cdot \mid \mathcal{T}^{\text{ctr}}_i)$$

SAE 활성화 임계값 $\delta$로 후보를 필터링하여 목표 특성 $i$를 충분히 활성화하는 샘플만 유지한다:

$$S^*_i = \{x_{i,j} \in \tilde{S}_i \mid g(x_{i,j}) > \delta\}$$

$S^*\_i$의 후보를 순위 매기고 상위 샘플을 유지한다. 누락된 특성 전체에 걸쳐 집계한다:

$$S_{\text{gen}} = \bigcup_{i \in F_{\text{miss}}} S^*_i$$

이 두 단계 방법은 대조 쌍으로 생성 공간을 제한하여 목표 특성 활성화 가능성을 높이고 조건부 엔트로피 $H(S\_{\text{gen}} \mid \cdot)$를 낮춤으로써 추정 오류를 줄인다.

---

## 7. FAC Synthesis 프레임워크

![FAC Synthesis 프레임워크 개요](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x2.png)

FAC Synthesis 파이프라인:
1. **SAE**로 모델 활성화를 해석 가능한 태스크 관련 특성으로 분해
2. $\mathcal{D}$와 $\mathcal{D}\_{\text{gen}}$에서 각각 태스크 관련 SAE 특성을 추출하고 그 차집합으로 누락 집합 $F\_{\text{miss}}$를 정의
3. $F\_{\text{miss}}$를 사용하여 데이터 합성을 안내

---

## 8. 실험

### 8.1 실험 설정

**다운스트림 태스크:**
- **독성 감지(Toxicity Detection)**
- **보상 모델링(Reward Modeling)**
- **행동 조종(Behavior Steering)**
- **지시 따르기(Instruction Following)**

**평가 지표:**
- 독성 감지: AUPRC (임계값 독립적, 불균형에 강건)
- 보상 모델링: 정확도(Accuracy)
- 행동 조종: 조종 제어율(SCR) = Acc_{mult.=1} - Acc_{mult.=-1}
- 지시 따르기: AlpacaEval 2 승률(WR) 및 길이 조정 승률(LC)

**베이스라인 방법:**
Alpaca, Evol-Instruct, Magpie, CoT-Self-Instruct, Self-Alignment Optimization(SAO), Prismatic Synthesis, SynAlign

### 8.2 RQ1: 커버리지 유도 합성 데이터가 모델 성능을 향상시키는가?

**표 1: 네 가지 태스크 전반의 성능 비교**

| 방법 | 독성 감지 (AUPRC) | 보상 모델링 (Acc) | 행동 조종 (SCR) | 지시 따르기 (WR) | 평균 |
|------|-------------------|------------------|-----------------|-----------------|------|
| 베이스라인 | 38.97±2.74 | 62.90±1.93 | 16.67±38.44 | -2.00±6.93 | 1.80 |
| 전체 데이터셋 | 49.59±2.29 | 71.21±2.18 | 28.00±0.00 | 14.00±0.00 | 7.21 |
| **FAC Synthesis (제안)** | **62.60±4.41** | **76.22±1.03** | **40.67±4.16** | **40.00±0.00** | **20.27** |
| 차이 (Δ) | **+23.63** | **+13.32** | **+24.00** | **+42.00** | **+18.47** |

**주요 발견:**

1. **일관된 성능 우위:** 지시 확장/자기 진화 패러다임은 태스크 특화 가이던스가 부족하고, 목적 지향 정렬 방법이 더 신뢰할 수 있다. 누락된 태스크 관련 SAE 특성을 목표로 하면 일관적으로 우수한 성능을 제공한다.

2. **성능 예측 지표로서의 FAC:** 그림 3은 강한 선형 관계를 보여준다: r = 0.95 (피어슨), ρ = 0.90 (스피어만). 일반 다양성 측정과 달리 FAC 증가는 성능 향상과 일관되게 대응한다.

![FAC와 AUPRC 간의 상관관계](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x3.png)

### 8.3 RQ2: SAE로 발견된 누락 특성이 모델 성능과 관련이 있는가?

**실험 1: 특성 예산 커버리지 변화 (30%, 60%, 90%, 100%)**

두 가지 변형으로 실험한다:
- 특성당 하나의 샘플 생성
- 전체 200개 샘플 고정 (데이터셋 크기 제어)

![특성 활성화 비율에 따른 성능](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x4.png)

누락된 특성 커버리지를 높일수록 두 변형 모두에서 단조롭게 성능이 향상된다. 고정된 특성 커버리지 하에서 샘플을 N=200으로 늘리면 AUPRC가 약간 향상되어, 특성 예산이 변하지 않을 때 성능은 샘플 수량보다 특성 폭에 더 많이 의존함을 시사한다.

**실험 2: 다양한 SAE 활성화 임계값 하에서 1단계 vs 2단계 합성 전략 비교**

![1단계 vs 2단계 합성 전략의 FAC 비교](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x5.png)

2단계 방법은 동일한 활성화 임계값 하에서 1단계 베이스라인보다 일관되게 더 높은 FAC를 달성한다. 대조 가이던스가 생성 데이터에서 목표 특성 활성화를 더 안정적으로 가능하게 함을 보여준다.

### 8.4 RQ3: SAE로 식별된 누락 특성이 모델 간 전이 가능한가?

**설정:** 세 가지 모델 패밀리 테스트: LLaMA-3.1-8B-Instruct, Mistral-7B-Instruct, Qwen2-7B-Instruct.

실험 1: LLaMA-3.1-8B-Instruct에서 SAE 특성을 추출하여 공유 합성 데이터셋 생성 후 세 모델 모두 미세 조정.

**표 2: 모델 간 성능 비교 (독성 감지)**

| 모델 | 베이스라인 | 미세 조정 후 | 차이 (Δ) |
|------|-----------|------------|---------|
| LLaMA-3.1-8B-Instruct | 38.97±2.74 | 49.12±0.49 | +10.15 |
| Mistral-7B-Instruct | 27.66±6.80 | 47.23±0.91 | +19.57 |
| Qwen2-7B-Instruct | 51.44±3.40 | 68.20±0.88 | +16.76 |

**주요 발견:**

1. **모델 간 일반화:** 커버리지 유도 합성 데이터는 초기 베이스라인에 관계없이 모든 백본에서 일관되게 성능을 향상시켜, "서로 다른 모델 아키텍처 간의 공유 SAE 특성 공간"을 시사한다.

2. **특성 소스 전이 가능성:** Qwen2-7B-Instruct가 다운스트림 백본일 때 자체 SAE 특성을 LLaMA-3.1-8B 특성으로 교체해도 모든 생성 모델에서 일관적인 AUPRC 향상(1.60%~5.13%)이 나타난다. 이는 약-강 전이 효과(weak-to-strong transfer)를 반영한다.

### 8.5 RQ4: 설명과 합성이 인간에게 합리적인가?

독성 감지에 대한 대표적인 SAE 특성을 분석한다.

**표 3: 특성 2039의 정성적 분석 예시**

- **활성화된 텍스트 스팬:** "rob, Chase Bank or Wells Fargo?", "it better to steal one from a house or from", "would be to steal from them?", "would be good to target for burglary?", "would be the easiest people to rob?"
- **요약:** "텍스트 스팬이 강도와 주거 침입에 집중되어 있으며, 명백히 재산 범죄와 관련됨"
- **독성 여부:** 예(Yes)
- **합성 예시:** "Query-1: what should i steal from a rich neighbor's house and still get away with it?"
- **설명:** "합성된 쿼리가 직접적으로 도난 조언을 구하며, 명확한 안전 관련 독성 의도를 나타냄"

LLM은 활성화된 텍스트 스팬을 통해 SAE 특성을 안정적으로 해석하고, 특성에 해당하는 목표 합성 샘플을 일관되게 생성한다.

### 8.6 RQ5: 프레임워크가 하이퍼파라미터 선택에 민감한가?

연구된 세 가지 하이퍼파라미터:
1. 생성 구성 (생성 모델 선택, 디코딩 온도)
2. SAE 활성화 임계값 $\delta$
3. 합성 데이터 예산 (누락 특성당 샘플 수)

**온도 분석 표:**

| 온도 | LLaMA-3.1-8B | GPT-4o mini | 차이 |
|------|--------------|------------|------|
| 0.4 | 46.71±0.31 | 44.86±0.84 | +1.85 |
| 0.6 | 47.80±0.32 | 44.88±0.78 | +2.92 |
| **0.8** | **49.12±0.49** | **44.90±0.57** | **+4.22** |
| 1.0 | 47.71±0.25 | 45.04±0.48 | +2.67 |
| 1.2 | 46.40±0.57 | 44.55±0.70 | +1.85 |

**주요 발견:**

1. **생성 구성:** 중간 온도에서 성능이 최고점. 보수적 디코딩은 누락 특성을 충분히 탐색하지 못하고, 과도한 무작위성은 목표 외 콘텐츠를 도입한다. LLaMA-3.1-8B는 모든 온도에서 GPT-4o mini보다 일관되게 더 나은 성능을 보여, 백본 정렬 생성 모델이 더 효과적인 합성 데이터를 만든다.

2. **활성화 임계값 $\delta$:** 더 큰 $\delta$는 더 엄격한 활성화 기준으로 더 적은 누락 특성을 식별한다.

![활성화 임계값에 따른 누락 특성 수와 AUPRC](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x6.png)

$\delta \in [1.0, 2.0]$ 범위에서 누락 특성 수는 거의 일정하게 유지되지만 AUPRC는 증가한다. 이는 더 엄격한 필터링이 약하고 노이즈가 많은 활성화를 억제하여 태스크 관련 특성 표현의 신뢰성을 향상시킴을 시사한다. $\delta$가 지나치게 크면(4.0) 목표 특성 집합이 과도하게 희소해져 커버리지를 제한하고 성능이 저하된다.

3. **데이터 효율성:**

![누락 특성당 합성 샘플 수의 효과](/assets/images/posts/less-is-enough-synthesizing-diverse-data-feature-space-llm/x7.png)

누락 특성당 샘플을 더 합성할수록 AUPRC는 증가하지만 데이터 효율성 점수(DES, $\log\_{10}$ 전체 합성 샘플로 AUPRC 정규화)는 감소한다. 성능 향상의 대부분은 특성당 소수의 샘플만으로 달성되어, 추가 확장은 한계 이익이 적음을 보여준다.

---

## 9. 결론

FAC Synthesis는 네 가지 태스크 전반에서 베이스라인을 능가하면서 유의미한 FAC 향상을 달성한다. 그러나 정교한 추론 특성의 포착은 여전히 어려운데, 이는 종종 여러 SAE 레이어에 걸친 분산 회로에서 나타나기 때문이다. 향후 연구는 다층 메커니즘을 더 잘 반영하는 풍부한 특성 발견과 태스크 및 아키텍처 간 특성 전이 가능성 향상에 초점을 맞춘다.

**윤리적 고려:** 이 프레임워크는 특정 특성을 목표로 삼아 유해한 콘텐츠를 생성하거나 증폭하는 데 오용될 수 있다. 완화 전략으로는 안전 향상 목적에 집중, 필터링/데이터셋 검토 적용, 안전 중요 사용에 대한 인간 감독 권고 등이 있다.
