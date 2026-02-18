---
title: "Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment"
date: 2025-07-29 17:23:08
categories:
  - 인공지능
tags:
  - APO
---

<https://arxiv.org/abs/2408.06266>

[Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment](https://arxiv.org/abs/2408.06266)

초록  
대규모 언어 모델(Large Language Models, LLMs)은 종종 대조적 정렬 목적(contrastive alignment objectives)과 선호 쌍(preference pair) 데이터셋을 사용해 정렬(alignment)된다. 그러나 모델, 쌍 데이터, 정렬 목적 간의 상호작용은 정렬 과정을 복잡하게 만들며, 때로는 기대 이하의 결과를 낳는다. 본 연구에서는 이를 분석하여 다음과 같은 두 가지 주요 발견을 제시한다. (i) 응답 자체가 대조적(contrastive)일 때 선호 데이터가 더 나은 학습 신호를 제공하며, (ii) 정렬 목적이 학습 중 모델에 대한 제어력을 더 많이 부여할수록 성능이 개선된다.

이러한 통찰을 바탕으로, 우리는 **Contrastive Learning from AI Revisions (CLAIR)**라는 새로운 데이터 생성 방법을 제안한다. 이는 보다 대조적인 선호 쌍을 생성하도록 설계되었다. 또한, 우리는 제어 가능성이 높고 더 안정적인 정렬 목적 함수인 **Anchored Preference Optimization (APO)**를 제시한다.

Llama-3-8B-Instruct 모델을 다양한 비교 가능한 데이터셋과 정렬 목적 함수로 정렬하고, 인간 평가와 높은 상관관계를 보이는 **MixEval-Hard** 점수를 통해 성능을 측정하였다. CLAIR로 생성한 선호 데이터는 모든 데이터셋 중 가장 강력한 성능을 보였으며, APO는 덜 제어 가능한 정렬 목적들보다 일관되게 우수한 결과를 냈다. 최종적으로, 32K CLAIR 선호 데이터와 APO를 활용해 학습한 모델은 Llama-3-8B-Instruct 대비 7.65% 성능 향상을 달성했으며, GPT-4 turbo와의 성능 격차를 45%까지 좁혔다.

### 1 서론

![](/assets/images/posts/586/img.png)

**그림 1 설명:** 정렬(alignment)은 선호 데이터와 학습 목적의 관점에서 명확히 정의되지 않는다.  
(A) 선호 쌍(preference pair)은 무관한 측면에서도 차이가 날 수 있는데, **Contrastive Learning from AI Revisions (CLAIR)**은 특정 측면만을 목표로 한 선호 신호를 생성한다.  
(B) 모델의 품질 자체가 정렬 학습에 영향을 미칠 수 있으며, **Anchored Preference Optimization (APO)**은 이를 명시적으로 고려한다.

언어 모델을 선호도(preferences)와 정렬하는 것은 LLM 개발의 핵심 구성 요소이며, 모델의 **성능, 안전성, 인간 가치에 대한 부합성**을 크게 향상시킨다(Christiano et al., 2017; Ouyang et al., 2022; Bai et al., 2022). 이러한 선호도는 입력 x에 대한 두 출력

![](/assets/images/posts/586/img_1.png)

의 **선호 쌍(preference pair)** 형태로 표현될 수 있으며, 이는 단일 출력보다 풍부한 학습 신호를 제공하고 더 표현력 있는 학습 목적을 가능하게 한다. 최근에는 **대조 학습 목적(contrastive learning objectives)**을 통해 정렬이 보다 쉽게 수행될 수 있게 되었다(Rafailov et al., 2024b).

하지만 이러한 장점에도 불구하고 정렬 결과는 여전히 최적과 거리가 있을 수 있다(Eisenstein et al., 2023; Feng et al., 2024; Park et al., 2024). 본 논문에서는 정렬의 본질을 (i) **데이터가 표현하는 선호 신호**, 그리고 (ii) **대조적 목적 함수의 학습 동역학** 두 측면에서 분석한다. 우리는 이 두 축에서 기존 정렬 방법이 충분히 명시되지 않았음을 발견했다. 이를 해결하기 위해, (i) 선호 데이터는 최소한의 대조성을 가져야 하며, (ii) 정렬 목적은 모델과 데이터 간의 **특정한 정렬 상황**을 고려해야 한다고 주장한다(그림 1 참조). 이러한 분석은 왜 기존 방법들이 서브옵티멀(suboptimal)한 결과를 보이는지 설명한다. 예를 들어, 5장에서 우리는 **고품질 출력으로 정렬한 모델조차도 쌍이 여러 통제 불가능한 측면에서 다를 경우 성능이 저하될 수 있음**을 보여준다.

이러한 통찰에 기반해 두 가지 주요 기여를 제안한다.

1. **Contrastive Learning from AI Revisions (CLAIR)**
   - 한 출력만 최소한으로 수정해 선호도를 표현하는 새로운 선호 쌍 생성 방법.
   - 판정자(judge)가 우수한 응답을 선택하는 기존 방법과 달리, **보다 정밀한 학습 신호**를 제공한다.
2. **Anchored Preference Optimization (APO)**
   - 모델과 데이터 간의 관계를 명시적으로 고려하는 대조적 목적 함수(contrastive objective) 계열.
   - 이러한 맞춤형 학습 동역학은 기존 목적 대비 더 우수한 정렬 성능을 달성한다.

우리는 **(i) 최소 대조성(minimally contrastive)을 갖춘 선호 데이터**, **(ii) 정렬 학습 동역학의 차별성**이라는 두 요소의 역할을 연구하기 위해, 네 가지 선호 데이터셋과 다섯 가지 정렬 목적 함수를 조합해 모델을 개별적으로 정렬하였다.

- **데이터셋 구성:**
  - CLAIR 방법으로 생성한 데이터셋
  - 두 가지 판정자 기반(judge-based) 데이터셋 (Reinforcement Learning from AI Feedback; Bai et al. 2022)
  - CLAIR의 대조성을 제거(ablated)한 변형 데이터셋
- **정렬 목적 함수:**
  - DPO (Rafailov et al., 2024b)
  - KTO (Ethayarajh et al., 2024)
  - 선호 응답에 대한 지속적 지도학습(SFT)
  - 제안한 APO의 두 변형(variants)

각 모델에 대해 **MixEval-Hard 정확도(Ni et al., 2024)**와 **길이 통제 AlpacaEval 점수(Dubois et al., 2024)**를 측정했으며, 두 벤치마크 모두 인간 평가 결과와 높은 상관성을 보인다(Chiang et al., 2024).

**실험 설정:**

- 정렬 대상 모델: **Llama-3-8B-Instruct (Dubey et al., 2024)**
- 선호 판정 및 수정: **GPT-4 turbo (Achiam et al., 2023)** 활용

**결과:**  
32K CLAIR 선호 데이터와 APO로 정렬한 가장 강력한 모델은 **MixEval-Hard에서 7.65% 성능 향상**을 기록하였으며, GPT-4 turbo와의 성능 격차를 **45%**까지 좁혔다. 분석 결과, **CLAIR 선호 데이터의 대조성(contrastiveness)**이 성능 향상의 핵심 요인임을 확인하였다. 또한 모든 정렬 데이터셋에서 **APO가 가장 우수한 성능을 달성**하였다. 마지막으로, 특정 모델과 선호 데이터셋 조합에 적합한 APO 변형을 선택하는 방법을 제시하고, 최근 정렬 연구들과 CLAIR 및 APO의 관계를 심층적으로 논의한다.

### 2 정렬(Alignment)에서의 불충분 명세(Underspecification)

정렬 과정은 **대상 모델(target model)**, **선호 데이터셋(preference dataset)**, **정렬 목적(alignment objective)** 간의 복잡한 상호작용을 만들어낸다. 본 장에서는 선호 데이터를 기반으로 한 모든 정렬 시도에서 나타나는 **실패 사례(failure case)**를 데이터와 목적 측면으로 나누어 분석한다.

#### 2.1 선호 데이터의 문제

![](/assets/images/posts/586/img_2.png)

그러나 이 **응답 쌍(pair)**은 여러 측면에서 차이가 날 수 있으며, 이 중 일부는 선호도와 무관한 **허위(spurious) 차이**일 수 있다. 이러한 허위 차이는 학습 시 **크레딧 할당 문제(credit assignment problem)**를 야기하여 정렬 학습을 어렵게 만든다. 반대로, **최소 대조성(minimally contrastive)**을 갖춘 응답 쌍은 차이가 적은 축(axis)에서만 발생하므로 허위 차이가 줄어든다. 따라서 선호 쌍이 명확한 최소 대조성을 보일수록 정렬 학습 신호는 더 선명해진다.

현재 사용되는 선호 데이터셋들은 **대조성의 정도(contrastiveness)**에서 큰 차이를 보인다. 예를 들어, **Stanford Human Preferences dataset (Ethayarajh et al., 2022)**에서는 두 출력이 같은 Reddit 게시물에 대한 응답일 뿐이며, 반드시 서로 비교 가능하도록 설계된 것은 아니다. **이상적인 선호 데이터셋은 두 응답 간의 차이가 매우 통제된(controlled) 형태로 구성되어야 한다.** 이러한 통찰은 이후 3장에서 소개할 **CLAIR** 방법의 기반이 된다.

#### 2.2 정렬 목적의 문제

선호 삼중항은 단지 한 응답이 다른 응답보다 낫다는 정보만을 제공한다. 그러나 **우수한 응답이 정말로 "좋은(good)" 응답인지**에 대한 정보는 제공하지 않으므로 **모호성(ambiguity)**이 발생한다.

![](/assets/images/posts/586/img_3.png)

대표적인 사례로, **UltraFeedback (Cui et al., 2024)** 데이터셋의 승자 응답 중 약 80%는 **Chatbot Arena Elo (Chiang et al., 2024)** 기준으로 **Llama-3-8B-Instruct보다 성능이 낮은 모델**에서 생성된 것이다. 이런 데이터셋으로 Llama-3-8B-Instruct를 단순 정렬하면 모델 성능이 악화될 수 있다. 이러한 사례는 4장에서 소개할 **Anchored Preference Optimization (APO)**의 필요성을 보여준다.

![](/assets/images/posts/586/img_4.png)

#### 그림 2 설명

프롬프트에 대한 Llama-3-8B-Instruct의 응답과, 이에 대한 GPT4-turbo의 수정(revision) 예시. 두 응답 간 차이가 하이라이트되어 있으며, 수정본은 원래 응답의 개요는 유지하면서도 개선 가능한 부분을 향상시켰다. 예를 들어, 원본에서 잘못된 파리 식당 수(2개)를 수정본에서는 정확히 3개로 고쳤다.

### 요약

현재 정렬 접근법은 두 가지 주요 축에서 **명세가 불충분(underspecified)**하다:

1. **비대조적(non-contrastive) 데이터**로 인해 선호 신호가 약하게 표현될 수 있다.
2. 정렬 목적은 **모델-데이터 관계(model-data relation)**를 고려해야 한다.

이후 장에서는 이 두 축을 개선하기 위한 방법을 제시한다.

### 3 CLAIR: 수정 기반 대조 학습 (Contrastive Learning from AI Revisions)

이번 장에서는 **Contrastive Learning from AI Revisions (CLAIR)**을 소개한다. CLAIR은 **최소한의 대조성(minimally contrasting)을 갖춘 선호 쌍(preference pair)**을 생성하기 위한 일반 절차다.

#### 3.1 방법 개요

![](/assets/images/posts/586/img_5.png)

본 연구에서는 더 강력한 LLM을 수정자로 활용하며, 수정 프롬프트는 **명확성(clarity)**, **정확성(correctness)**, **흥미도(engagement)**를 개선하도록 설계되었다(자세한 프롬프트와 데이터셋은 부록 A 참고). 그림 2는 이 방식으로 생성된 삼중항 예시를 보여준다. 여기서 패자 응답은 Llama-3-8B-Instruct가 생성했으며, GPT4-turbo가 이를 수정하였다. 수정본은 원본의 대부분을 유지하면서도 세부 사항을 개선한 형태다.

Dubey et al. (2024)도 llama-3.1 모델 개발 시 인간 수정(human revision)을 사용했지만, 이 과정은 **최소 대조성**을 만드는 것이 아니라 품질 차이를 크게 만드는 데 초점을 맞췄다는 점에서 CLAIR과 다르다.

#### 3.2 기존 방법과의 차별점

기존 선호 데이터 수집 방식과 CLAIR의 가장 큰 차별점은 **데이터 생성 방식**이다. 예를 들어, **on-policy judge paradigm**(Reinforcement Learning from AI Feedback; Bai et al. 2022)에서는 다음과 같이 두 출력을 모델 M(x)에서 샘플링하고, **판정자(Judge)**가 승자와 패자를 결정한다.

![](/assets/images/posts/586/img_6.png)

또한 **off-policy judge paradigm**에서는 대상 모델과 다른 모델 M′, M′′에서 출력된 응답을 비교해 판정자가 승패를 가른다

![](/assets/images/posts/586/img_7.png)

이 두 가지 판정 기반 접근법은 CLAIR과 비교할 수 있는 유용한 기준선(baseline) 역할을 한다.

#### 3.3 추가 기준선: Stronger Preferred

![](/assets/images/posts/586/img_8.png)

#### 3.4 데이터셋 구성

정렬 실험(섹션 5)에서 사용하기 위해, 우리는 식 (1)~(4) 절차를 통해 **네 가지 선호 데이터셋**을 구축했다. 모든 데이터셋은 **UltraFeedback (Cui et al., 2024)**에서 균일하게 샘플링된 **32K 프롬프트**를 기반으로 한다. UltraFeedback은 다양한 도메인을 포괄하는 널리 사용되는 선호 데이터셋이다.

- 대상 모델 M: **Llama-3-8B-Instruct**
- 오프-폴리시 판정 데이터셋: UltraFeedback의 기존 판정 결과 활용  
  (이 데이터의 승자 출력 중 약 80%는 Chatbot Arena Elo 기준으로 Llama-3-8B-Instruct보다 약한 모델에서 생성됨)

#### 3.5 최소 대조성 평가

CLAIR의 핵심 목표 중 하나는 **최소 대조성(minimally contrastive)** 선호 쌍을 만드는 것이다. 이를 평가하기 위해 두 가지 단순한 휴리스틱 지표를 사용했다.

- **Jaccard Similarity (↑ 높을수록 좋음)**: 승자와 패자의 토큰 집합 간 교집합/합집합 비율
- **Levenshtein Edit Distance (↓ 낮을수록 좋음)**: 승자와 패자 간 문자 단위 편집 거리

이 지표에서 최소 대조성이 높을수록 **Jaccard 유사도는 높고, Levenshtein 거리는 낮아야 한다.**

#### 3.6 결과

**표 1:** Llama-3-8B-Instruct 기반 네 가지 데이터셋에서 승자와 패자 응답 간 유사도 분석

![](/assets/images/posts/586/img_9.png)

CLAIR 데이터셋은 두 지표 모두에서 가장 높은 품질의 대조성을 보여주며, 다른 방식보다 훨씬 우수한 결과를 기록했다.

---

![](/assets/images/posts/586/img_10.png)

그러면 증류모델이나 교사모델인거지. 거의 교사모델에 가까운건데 이게 뭐가 새롭지?

### 증류(Knowledge Distillation)

- **교사 모델의 완전한 출력**을 그대로 학생 모델이 학습
- 학생 모델은 **교사의 전체 행동 패턴**을 따라가게 됨
- 결과적으로 교사 모델의 편향까지 전이될 수 있고, **불필요하게 큰 차이**를 학습하게 됨

### CLAIR

- **대상 모델의 출력(y\_l)**을 기준으로 **최소 수정(minimal revision)**만 수행
- 교사 모델은 완전한 답을 재작성하지 않고, **“차이를 최소화하면서 필요한 부분만 개선”**
- 학습 신호는 **대상 모델의 한계점**에만 집중
- 결과적으로 **대조 학습(contrastive learning)에 최적화된 데이터셋** 생성

CLAIR은 단순히 "데이터셋을 바꾸는 것"처럼 보이지만,  
사실상 **대상 모델의 출력을 교사 모델이 교정하는 과정을 통해 데이터셋 자체를 새롭게 정의**하는 방법이고,  
이게 기존 RLHF나 전통 증류와 달리 **최소 대조성을 보장하는 정렬 데이터셋**을 만든다는 점에서 새로운 접근이야.

그러면 만약에 CLAIR를 수행하는 모델이 계속 잘못되게하면 이상한거 아니야? 그리고 해당 모델들은 이미 dpo, ppo 방식으로 진행되서 실제 쌍을 완전하게 분리시킬수 있는지도 모르고

즉, CLAIR의 전제는 **“대상 모델과 Reviser 모델의 품질 격차가 존재해야 함”**

- 둘 다 이미 DPO/PPO 정렬된 최신 모델이라면, 쌍을 만들 때 "분리되는 특징"이 적어서 학습 효과가 낮아질 수 있음.

이미 꽤 괜찮은 모델이 존재하고, 그걸 더 정교하게 정렬하려는 실무적 최적화 느낌

---
