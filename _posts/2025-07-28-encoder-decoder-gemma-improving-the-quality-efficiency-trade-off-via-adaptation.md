---
title: "Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation"
date: 2025-07-28 23:17:09
categories:
  - 인공지능
---

<https://arxiv.org/abs/2504.06225>

[Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation

While decoder-only large language models (LLMs) have shown impressive results, encoder-decoder models are still widely adopted in real-world applications for their inference efficiency and richer encoder representation. In this paper, we study a novel prob

arxiv.org](https://arxiv.org/abs/2504.06225)

초록  
디코더 전용(Decoder-only) 대규모 언어 모델(LLM)은 인상적인 성능을 보여주고 있지만, 인코더-디코더(Encoder-Decoder) 모델은 추론 효율성과 풍부한 인코더 표현 덕분에 여전히 실제 응용에서 널리 사용되고 있습니다. 본 논문에서는 사전 학습된 디코더 전용 LLM을 인코더-디코더 구조로 **적응(adaptation)시키는** 새로운 문제를 다루며, 두 접근 방식의 장점을 결합해 품질과 효율성 간의 균형을 개선하는 것을 목표로 합니다. 우리는 이러한 적응이 디코더 전용 LLM의 능력을 계승할 수 있을 뿐만 아니라, 처음부터 사전 학습하는 것에 비해 계산 비용을 줄일 수 있다고 주장합니다. 이를 위해 다양한 사전 학습 목표와 매개변수 초기화·최적화 기법을 체계적으로 탐구했습니다.

Gemma 2 (2B 및 9B) 모델과 새롭게 사전 학습한 mT5 크기(최대 1.6B) 모델을 기반으로 한 광범위한 실험을 통해, **적응 방식의 효과와 인코더-디코더 LLM의 우위**를 입증합니다. 유사한 추론 예산 하에서 인코더-디코더 LLM은 디코더 전용 모델과 동등하거나 종종 더 나은 사전 학습 성능을 보이며, 미세 조정(finetuning) 성능에서는 크게 앞섭니다. 예를 들어, **Gemma 2B-2B 모델은 지시(instruction) 튜닝 후 Gemma 2B 대비 약 7% 높은 성능**을 기록합니다. 또한 인코더-디코더 적응은 다양한 크기의 모델을 유연하게 조합할 수 있게 하며, **Gemma 9B-2B 모델은 Gemma 2B-2B 대비 3% 이상 우수한 성능**을 보였습니다. 적응된 인코더 표현은 SuperGLUE에서도 더 나은 결과를 제공합니다. 우리는 향후 연구를 촉진하기 위해 학습된 체크포인트를 공개할 예정입니다.

### 1. 서론

신경망 아키텍처는 입력 데이터에 대한 특정 가정이나 **귀납적 편향(inductive bias)**을 포함하도록 설계되는 경우가 많으며, 이는 모델 성능 향상이나 계산 효율성 개선, 혹은 두 가지 모두로 이어집니다. 대규모 언어 모델(LLM)에 널리 사용되는 **디코더 전용(decoder-only) 아키텍처**(Brown et al., 2020)와 달리, **인코더-디코더(encoder-decoder) 아키텍처**는 입력 이해를 위한 인코더와 출력 생성을 위한 디코더라는 별도의 모듈을 채택합니다(Vaswani et al., 2017). 이러한 분리는 기능별 매개변수를 독립적으로 관리할 수 있게 하여 문맥 표현(contextual representation)과 복잡한 과제를 다루는 데 더 높은 자유도를 제공합니다(Tay et al., 2022; Wang et al., 2022). 또한 인코더와 디코더의 크기를 유연하게 조정(예: 큰 인코더와 작은 디코더 조합)하여 **품질-효율성(quality-efficiency) 간의 균형**을 제어할 수 있는데(Kasai et al., 2020; Zhang et al., 2022), 이는 LLM 배포에서 점점 더 중요한 요소가 되고 있습니다(Gemini et al., 2024). 그러나 이러한 장점에도 불구하고, **인코더-디코더 LLM**에 대한 연구는 최근 거의 주목받지 못하고 있습니다.

본 논문에서는 다음 질문을 탐구하며 이 고전적인 아키텍처를 다시 조명합니다.  
**“기존 사전 학습된 디코더 전용 LLM을 적응(adaptation)시켜 강력한 인코더-디코더 LLM을 만들 수 있는가?”**

사전 학습은 막대한 자원을 필요로 하지만, 다양한 크기의 강력한 디코더 전용 모델은 이미 널리 사용 가능하므로(Dubey et al., 2024; Team et al., 2024; Liu et al., 2024a; Yang et al., 2024; Jiang et al., 2024), 우리는 **새로운 모델을 처음부터 학습하기보다는 적응 방식이 더 실질적이라고 판단**합니다. 우리의 가설은, 디코더 전용 모델의 매개변수를 재사용함으로써 학습 속도를 가속화하고 내부 지식을 효과적으로 인코더-디코더 구조로 이전시켜 성능을 유지하거나 심지어 향상시킬 수 있다는 것입니다. 또한 적응 방식은 **서로 다른 크기의 디코더 전용 모델을 조합**해 특정 품질-효율성 요구를 충족할 수 있게 합니다. 그러나 **이러한 적응을 위한 최적의 방법과 성능 향상의 정도는 여전히 미해결 과제**이며, 본 논문에서는 이를 체계적으로 분석하고자 합니다.

![](/assets/images/posts/585/img.png)

#### 그림 1: 제안된 접근법 개요

우리는 사전 학습된 디코더 전용 모델을 기반으로 인코더-디코더 모델을 구성합니다. 모델의 아키텍처와 매개변수는 디코더 전용 모델에서 계승하되, **교차 어텐션(cross-attention)**에 대해서는 인코더 및 디코더 크기에 따라 다른 초기화 방법을 적용합니다. “ROPE”는 로터리 임베딩(rotary embedding), “FFN”은 피드포워드(feed-forward) 계층을 의미합니다.

우리는 Gemma 2 (Team et al., 2024)를 실험 환경으로 사용합니다. 그림 1에서 보이듯, 인코더-디코더 아키텍처는 기본적으로 원래 Transformer(Vaswani et al., 2017) 구조를 따르면서 Gemma 2의 수정 사항을 반영합니다. 핵심 아이디어는 **사전 학습된 디코더 전용 모델의 매개변수를 초기화 단계(warmup)로 활용하고, 이후 자기 지도 학습(self-supervised learning)으로 모든 매개변수를 사전 학습 혹은 적응시키는 것**입니다.

또한 인코더와 디코더의 설정이 동일한지 여부에 따라 **교차 어텐션 레이어의 초기화 및 최적화 전략**을 달리 제안합니다. 그리고 지식 증류(knowledge distillation, Hinton et al., 2015)를 결합한 prefix 언어 모델링(prefix LM)과 UL2(Tay et al., 2022) 등 다양한 사전 학습 목표를 비교합니다. Gemma 2 2B와 9B 모델 외에도, **다양한 규모의 소형 모델**을 사전 학습해 규모별 적응 효과를 분석했습니다.

모델 성능 평가를 위해, 사전 학습 모델과 지시 튜닝(instruction-tuning) 모델에 각각 적합한 **다양한 벤치마크**를 사용했습니다. 또한 **SuperGLUE(Wang et al., 2019a)**를 통해 학습된 문맥 표현의 품질을 측정했습니다.

#### 주요 발견 사항

- **사전 학습된 디코더 전용 LLM 활용은 강력한 인코더-디코더 LLM 구축에 효과적**이며, 특히 유사한 추론 FLOPs 환경에서 지시 튜닝 이후 다운스트림 성능이 크게 향상됩니다.
- **제안된 적응 방식은 매우 유연**하며, 예를 들어 **대형 인코더-소형 디코더(9B-2B)** 조합으로 Gemma 2 2B 대비 유사한 생성 지연(latency)에서 상당한 품질 향상을 달성할 수 있습니다.
- 적응 방식은 **계산 효율성이 높을 뿐만 아니라, 처음부터 사전 학습하는 것보다 효과적**입니다.
- **사전 학습 목표의 차이가 중요**합니다. Prefix LM + 지식 증류 모델은 **생성 과제에 유리**하고, UL2 모델은 **인코더 표현 품질이 더 우수**합니다.

### 2. 관련 연구 (Related Work)

디코더 전용(decoder-only) 아키텍처는 현재 LLM의 사실상 표준(de facto standard)으로 자리잡았지만, **인코더-디코더와 디코더 전용 모델링 간의 논쟁은 여전히 결론이 나지 않은 상태**입니다. 이전에 발표된 많은 연구들은 강력한 인코더-디코더 모델을 사전 학습하는 다양한 접근 방식을 제안해 왔습니다. 예를 들어, **MASS**(Song et al., 2019), **T5**(Raffel et al., 2020), **mT5**(Xue et al., 2021), **byT5**(Xue et al., 2022), **BART**(Lewis et al., 2020), **OpenBA**(Li et al., 2023) 등이 있습니다.

Tay et al. (2022)는 서로 다른 사전 학습 목표를 비교하며 **UL2와 인코더-디코더 모델링의 우수성**을 강조했습니다. Zhang et al. (2022)는 기계 번역 환경에서 두 아키텍처의 스케일링 동작(scaling behavior)을 체계적으로 분석하며, 적절한 학습 목표를 적용할 경우 두 모델이 유사한 성능을 낼 수 있음을 보였습니다. Wang et al. (2022)는 다양한 모델링 선택지와 사전 학습 목표를 심층 탐구하며, **LLM의 제로샷 일반화(Zero-Shot Generalization)**에 중점을 두었습니다. 이들은 특히 지시 튜닝(instruction tuning) 이후 인코더-디코더 LLM이 가장 뛰어난 성능을 달성한다는 점을 밝혀냈으며, 이는 본 연구의 실험 결과와도 일치합니다. 다만, 그들의 적응 연구는 **사전 학습 목표 간 적응**을 다룬 것이며, 본 연구처럼 **디코더 전용 LLM에서 인코더-디코더 LLM으로의 적응**은 아닙니다.

#### 사전 학습된 모델을 활용한 인코더-디코더 모델링

**BERT 시대(Devlin et al., 2019)**에는 사전 학습된 BERT를 활용하여 인코더-디코더 성능을 강화하는 다양한 연구가 진행되었습니다. 예를 들어 기계 번역(Zhu et al., 2020; Clinchant et al., 2019; Yang et al., 2020), 문법 오류 교정(Kaneko et al., 2020), 요약(Liu & Lapata, 2019), 텍스트 생성(Chen et al., 2019) 등 다운스트림 과제에 적용되었습니다. 우리 연구 역시 이러한 맥락을 따르지만, **사전 학습된 디코더 전용 LLM**을 기반으로 하며, **범용 인코더-디코더 LLM** 개발에 중점을 둡니다.

#### 추론 친화적(inference-friendly) LLM 연구

다른 관련 방향으로는 **추론 효율성을 개선하는 기술**들이 있습니다. 여기에는

- **양자화(quantization)** (Dettmers & Zettlemoyer, 2023)
- **Key-Value 캐시 최적화** (Corallo & Papotti, 2024)
- **순환 모델링(recurrent modeling)** (Gu & Dao, 2023; Botev et al., 2024)
- **강력한 소형 LLM 개발** (Abdin et al., 2024; Liu et al., 2024b)

등 다양한 기법이 포함됩니다. 이들 기법은 **효율성 향상**에 상당한 기여를 하지만, 본 논문에서 제안하는 **인코더-디코더 적응 방식과는 근본적으로 별개의 초점**을 가지며, **두 접근법을 결합해 더 큰 효율성을 실현할 수 있다는 점에서 상호 보완적**입니다.

### 3. 접근법: 인코더-디코더 적응 (Encoder-Decoder Adaptation)

![](/assets/images/posts/585/img_1.png)

표 1은 모델 구성 요소를 나타내며, **레이어 수(#Layers)**, **모델 차원/FFN 차원/헤드 수(d\_model/ffn/head)**, **쿼리·키-값 헤드 수(q/kv heads)**, 그리고 **모델 파라미터 수(#Params)**를 포함합니다. 인코더-디코더 모델의 경우 균형 잡힌 구조(예: 2B-2B)의 파라미터 수를 표기했습니다. 예를 들어, **9B-2B 모델은 10.4B 파라미터**를 가집니다.

### 3.1 아키텍처 (Architecture)

대규모 언어 모델(LLM)의 사전 학습은 **막대한 계산량과 시간**이 소요됩니다. 이를 줄이기 위해 본 연구는 **기존 디코더 전용 LLM을 인코더-디코더로 적응(adaptation)**시켜 사전 학습된 체크포인트를 초기화에 활용하는 방법을 제안합니다(그림 1 참조).

이로 인해, 우리는 **원래 디코더 전용 모델과 가능한 한 유사한 아키텍처를 유지**하며, 필요한 경우에만 변경을 도입합니다. 결과적으로 다음과 같은 구조를 갖습니다.

1. **인코더(Encoder)**는 디코더 전용 모델과 동일한 구조를 가지지만, **자기 어텐션(self-attention)**을 **인과적(causal)**에서 **양방향(bidirectional)**으로 전환합니다. 섹션 6의 소거 실험(ablations)에서 양방향 어텐션이 다운스트림 성능에 미치는 중요한 효과를 보여줍니다.
2. **디코더(Decoder)** 블록에서는 FFN과 자기 어텐션 부분은 디코더 전용 모델과 동일하며, **교차 어텐션(cross-attention)**은 자기 어텐션과 동일한 헤드 수와 헤드 차원을 가지지만 인코더의 전체 출력을 참조(attend)합니다.

본 연구는 **Gemma 2(Team et al., 2024)**를 기반으로 하지만, 제안된 접근법은 특정 아키텍처에 제한되지 않고 **LLaMA(Dubey et al., 2024), QWen(Yang et al., 2024), DeepSeek(Liu et al., 2024a)** 등 다른 모델 패밀리에도 적용 가능합니다. 이론적으로는 서로 다른 모델 패밀리를 조합하여 **예: LLaMA 인코더 + QWen 디코더**와 같은 형태도 가능합니다.

또한, 본 접근법은 **불균형 인코더-디코더(unbalanced encoder-decoder) 모델**도 지원하며, 이는 인코더가 디코더보다 훨씬 큰 경우입니다. 이러한 구성은 **입력 처리 능력이 생성 능력보다 중요한 응용 분야(예: 요약)**에서 유리합니다. 이 경우, 새로운 정보를 생성할 필요가 없으므로 **생성 시간(latency)**을 크게 줄이면서도 경쟁력 있는 품질을 유지할 수 있습니다.

### 3.2 초기화 (Initialization)

디코더 전용 체크포인트에서 인코더-디코더 모델을 초기화할 때, **각 레이어를 디코더 전용 체크포인트의 가장 유사한 가중치에 매핑**하려고 시도합니다.

- 인코더는 **새로운 가중치를 도입하지 않으므로 디코더 전용 체크포인트에서 완전히 초기화**됩니다.
- 디코더의 **FFN 및 자기 어텐션 부분**은 대응되는 디코더 전용 체크포인트의 가중치에서 초기화됩니다.
- **교차 어텐션(cross-attention)**의 경우:
  - **균형 구조(예: 2B-2B)**에서는 자기 어텐션 가중치로 초기화합니다.
  - 그렇지 않을 경우(불균형 구조)에는 **처음부터 교차 어텐션을 초기화한 후, 초기 K 스텝 동안 교차 어텐션만 미세 조정(finetune)하고 나머지 매개변수는 동결**합니다.
  - 이후 K 스텝이 지나면 모든 모델 매개변수를 함께 튜닝합니다.

### 3.3 사전 학습 목표 (Pretraining Objective)

디코더 전용 사전 학습은 일반적으로 **단일 시퀀스에 대한 인과적 언어 모델링(causal LM)**을 채택합니다. 반면, **인코더-디코더 적응**은 입력 시퀀스와 목표 시퀀스를 분리하여 각각 인코더와 디코더에 공급해야 합니다. 이를 위해 두 가지 고전적 사전 학습 목표를 탐구합니다.

1. **Prefix Language Modeling (PrefixLM)**
   - 인과적 언어 모델링과 유사하지만, **prefix 조건**을 추가한 방식입니다.
   - 전처리를 단순화하기 위해 시퀀스를 절반으로 나누어 **전반부는 입력**, **후반부는 목표(target)**로 사용합니다.
   - 또한, 이 방식은 디코더 전용 모델로부터의 **지식 증류(knowledge distillation)**를 쉽게 적용할 수 있습니다.
2. **UL2 (Tay et al., 2022; Wang et al., 2022)**
   - 더 복잡한 방식으로, **여러 수준의 난이도를 가진 복원(denoising) 작업**으로 구성됩니다.
   - 데이터 준비는 Tay et al. (2022)를 따릅니다.

이 두 가지 사전 학습 목표의 성능은 실험에서 비교 평가합니다.

---

# ? Transformer Attention 구조 비교

? 디코더 전용 (GPT, LLaMA)

입력: "안녕하세요"

↓

Self-Attention 1

↓

Self-Attention 2

↓

Self-Attention N

↓

출력: "안녕하세요!"

? Attention 흐름

각 토큰이 다른 모든 토큰을 참조

"안녕" ↔ "하세요" 관계 학습

? 인코더-디코더 (T5, BART)

입력: "Hello"

↓

인코더 Self-Attention

↓

Cross-Attention

↓

디코더 Self-Attention

↓

출력: "안녕하세요"

? Attention 흐름

인코더: "Hello" 내부 관계

디코더 → 인코더 참조

디코더: "안녕하세요" 내부 관계

### ? 핵심 포인트

**❌ 잘못된 생각:**  
"디코더 전용에는 attention이 없다"

**✅ 올바른 이해:**  
"디코더 전용에는 Self-Attention만 있다"

**? 기억할 점:**  
• **모든 Transformer = Attention 기반**  
• 디코더 전용: Self-Attention만 사용  
• 인코더-디코더: Self-Attention + Cross-Attention  
• GPT도 Self-Attention으로 문맥을 이해합니다!

## 1. 공통점

- **모두 Query, Key, Value로 연산하는 Attention 메커니즘**을 사용
- 수학적으로는 동일한 연산 (scaled dot-product attention)
- 결국 토큰 간 관계를 계산하는 구조라는 점에서는 같습니다
---

## 2. 차이점 (적용 위치와 컨텍스트 범위)

1. **Self-Attention (자기 어텐션)**
   - 같은 시퀀스 내 토큰들끼리 관계 계산
   - 디코더 전용에서는 **미래 토큰 차단(causal mask)** 적용
   - 인코더에서는 **양방향(bidirectional)**로 전체 시퀀스를 봄
2. **Cross-Attention (교차 어텐션)**
   - **디코더 쪽에서만 등장**
   - 디코더의 Query가 인코더의 Key/Value와 관계를 맺음
   - 입력 시퀀스(인코더 출력)와 출력 시퀀스(디코더 입력) 사이의 연결고리
---

## 3. "문장 단위 vs 문단 단위" 비유로 보면?

- Self-Attention = **자기 안에서 문맥 파악** (한 문장 안의 단어들이 서로 관계 맺기)
- Cross-Attention = **다른 문장에서 정보 끌어오기** (인코더가 요약해둔 문단 전체 정보를 디코더가 활용)
---

### 결론

- **연산 자체는 동일한 Attention**이지만,
- **디코더 전용 모델은 Self-Attention만 쓰고**,
- **인코더-디코더 모델은 Self + Cross-Attention을 모두 사용**해 입력-출력 관계를 표현합니다.
---

### 4. 설정 (Setup)

#### **데이터 설정 (Data Setting)**

우리의 사전 학습(pretraining) 및 지시 튜닝(instruction tuning) 데이터 – 지도 미세조정(SFT)과 인간 피드백 기반 강화 학습(RLHF) 포함 – 은 **Gemma 2(Team et al., 2024)**를 따릅니다.

- **적응(adaptation)** 과정에서는 Gemma 2 사전 학습 데이터(8조 토큰)를 **PrefixLM**과 **UL2** 형식으로 전처리합니다.
- Gemma 2 사전 학습 데이터는 **지식 증류(knowledge distillation)** 정보를 포함하고 있으며, PrefixLM에서는 이를 보존하고 UL2에서는 **ground-truth 타깃**을 사용합니다. (UL2로 교사 logits을 매핑하는 것은 비직관적이기 때문입니다.)
- 전처리된 데이터의 입력-출력 시퀀스 길이는 PrefixLM의 경우 **4096-4096**, UL2의 경우 **8192-8192**입니다.
- 모델 적응은 최대 **2조 토큰**을 사용하여 진행합니다.

#### **모델 설정 (Model Setting)**

- **기반 모델(Base Model):**
  - Gemma 2 (2B 및 9B) 디코더 전용 LLM 사용
- **소형 모델:**
  - mT5(Xue et al., 2021) 구성(Small, Base, Large, XL)을 Gemma 2 프레임워크에서 사전 학습 후 인코더-디코더 LLM으로 적응
- 세부 모델 구성은 **표 1**에 제시

#### **평가 (Evaluation)**

다양한 학술 벤치마크를 사용해 LLM의 능력을 평가합니다. 구체적으로 다음과 같습니다:

**Pretraining (PT) 벤치마크**

- **Boolq** (Clark et al., 2019)
- **SIQA** (Sap et al., 2019)
- **PIQA** (Bisk et al., 2020)
- **ARC-c & ARC-e** (Clark et al., 2018)
- **MMLU** (Hendrycks et al., 2021)
- **MMLU Pro** (Wang et al., 2024)
- **HellaSwag** (Zellers et al., 2019)
- **Winogrande** (Sakaguchi et al., 2021)
- **TruthfulQA** (Lin et al., 2021)
- **AGIEval** (Zhong et al., 2023)
- **BBH** (Suzgun et al., 2022)
- **DROP** (Dua et al., 2019)
- **GPQA** (Rein et al., 2023)
- **GSM8K** (Cobbe et al., 2021)
- **HumanEval** (Chen et al., 2021)
- **Lambada** (Paperno et al., 2016)
- **MATH-500** (Hendrycks et al., 2021)
- **MBPP** (Austin et al., 2021)
- **NQ** (Kwiatkowski et al., 2019)
- **TriviaQA** (Joshi et al., 2017)
- **WMT23** (Kocmi et al., 2023)

> 사전 학습된 LLM에는 **zero/few-shot prompting**을 적용하고, 평균 결과를 PT 점수로 보고합니다.

**Instruction-Tuning (IT) 벤치마크**

- GSM8K, MMLU, MMLU Pro, MBPP, HumanEval, MATH-500, BBH, GPQA (Diamond), WMT23, MGSM (Shi et al., 2022)

> 지시 튜닝 모델에는 **작업별 지시어(task-specific instruction)**와 함께 **zero/few-shot prompting**을 수행하고, 평균 결과를 IT 점수로 보고합니다.

**SuperGLUE** (Wang et al., 2019b)

- 학습된 문맥 표현 품질을 평가하기 위해 사용
- **인코더-디코더 모델**: 인코더의 마지막 토큰 표현 위에 **task-specific head**를 쌓음
- **디코더 전용 모델**: 디코더의 마지막 토큰 표현 위에 **task-specific head**를 쌓음
- 모든 파라미터를 학습 세트에서 미세 조정하며, 학습률·배치 크기·드롭아웃은 각 작업마다 **그리드 탐색(grid search)**으로 최적화
- 모든 작업을 **분류(classification)**로 재구성하고, COPA, WIC, WSC, RTE, MultiRC, CB, Boolq의 개발 세트(dev-set) 평균 정확도를 보고

#### **학습 설정**

- 생성 과제(generative tasks)에는 항상 **탐욕적 샘플링(greedy sampling)** 사용
- 사전 학습, SFT, RLHF는 **Gemma 2 레시피**를 따르되,
  - 인코더-디코더 LLM에는 학습률을 별도로 경험적(empirical)으로 튜닝
- 불균형 인코더-디코더 적응(예: 9B-2B)의 경우, **교차 어텐션 워밍업 스텝(K)**은 1000으로 설정

### 5. 결과 (Results)

![](/assets/images/posts/585/img_2.png)

#### **그림 2 설명**

- 그림 2: **적응(Adaptation) 과정에서 사전 학습 토큰 수에 따른 사전 학습(PT) 성능 변화**
- 인코더-디코더 적응은 특히 **균형 아키텍처**에서 매우 빠르게 수렴하는 모습을 보임

#### **주요 관찰 결과**

- 적응 방식은 **사전 학습된 파라미터를 초기화에 활용**하지만, 이것이 모델 수렴에 얼마나 도움이 되는지는 명확하지 않았음
- 그림 2 결과에 따르면, 적응은 **계산 효율성이 매우 높아**, 수십억 토큰만으로도 **디코더 전용 모델과 유사한 성능**에 도달
- **균형 아키텍처(2B-2B, 9B-9B)**는 모든 파라미터가 사전 학습된 모델에서 초기화되기 때문에 **불균형 아키텍처(9B-2B)**보다 훨씬 빠르게 수렴
  - 불균형 아키텍처는 교차 어텐션이 무작위 초기화되므로 학습 속도가 상대적으로 느림

#### **추가 사전 학습의 효과**

- 추가 사전 학습은 **균형 모델의 평균 성능을 소폭 향상**시키지만,
  - **GSM8K**나 **DROP**과 같은 특정 과제에서는 상당한 이득을 보임
- **9B-2B 모델**은 적응 과정에서 성능이 꾸준히 향상되며,
  - 빠르게 **Gemma 2 2B를 초과**하고
  - **Gemma 2 9B**의 성능에 점점 근접함

#### **시사점**

- 이 결과는 **서로 다른 크기의 디코더 전용 LLM으로부터 인코더-디코더 모델을 적응하는 것이 가능함**을 보여줌
- 또한, **사전 학습된 모델의 지식을 효과적으로 재활용할 수 있음**을 시사

![](/assets/images/posts/585/img_3.png)

#### (a) PT 및 IT 벤치마크 결과

- 사전 학습(PT) 및 지시 튜닝(IT) 모델 성능 비교

![](/assets/images/posts/585/img_4.png)

#### (b) SuperGLUE에서의 미세 조정(finetuned) 성능

- 문맥 표현 품질 평가
- 인코더-디코더 모델은 **인코더 마지막 토큰**, 디코더 전용 모델은 **디코더 마지막 토큰** 표현 기반으로 분류 헤드 추가 후 미세 조정

### 표 2: PT, IT 및 SuperGLUE 벤치마크 주요 결과

- **설명**
  - “Gemma 2”: 디코더 전용 모델 결과
  - “+PrefixLM / UL2”: Prefix 언어 모델링(지식 증류 포함) 또는 UL2로 적응된 인코더-디코더 모델 결과
  - 공간 절약을 위해 Gemma 2 결과를 해당 인코더-디코더 행에 함께 표기 (예: 2B-2B의 Gemma 2 = Gemma 2 2B)
  - 괄호 안 수치는 **RLHF 모델**의 결과
  - 가장 높은 성능은 **굵게 표시**
  - PT 점수와 IT 점수는 서로 다른 과제 평균값이므로 직접 비교 불가

![](/assets/images/posts/585/img_5.png)

(a) Results for pretrained models.

![](/assets/images/posts/585/img_6.png)

(b) Results for RLHFed models.

### 표 3: PT 및 RLHF 모델의 세부 결과

- **설명**
  - 다양한 개별 과제에서의 성능을 PT 모델과 RLHF 모델 기준으로 상세 비교
  - Gemma 2 (디코더 전용)와 PrefixLM으로 적응한 인코더-디코더 모델 비교
  - 가장 높은 성능은 **굵게 표시**

### 사전 학습 목표의 중요성: UL2와 PrefixLM의 특성 비교

이전 연구(Tay et al., 2022)에서는 UL2가 PrefixLM보다 우수하다고 보고되었지만, 본 연구의 PrefixLM은 **지식 증류(knowledge distillation)**가 결합되어 있어 특히 소형 모델에서 성능이 크게 향상되었습니다. 표 2에서 두 사전 학습 목표를 비교한 결과는 다음과 같습니다.

- **UL2의 강점**:  
  UL2는 더 강력한 **문맥 표현(contextual representation)**을 제공하며, SuperGLUE에서 대부분의 모델 스케일에서 PrefixLM보다 우수합니다. 이는 이전 연구 결과와도 일치합니다.
- **PrefixLM의 강점**:  
  PrefixLM은 생성 중심의 학습 목표와 지식 증류 효과 덕분에 더 강력한 **생성 능력(generative capability)**을 지니며, PT 및 IT 벤치마크 대부분에서 UL2보다 높은 성능을 기록합니다. 특히 9B-2B 모델에서 PT와 IT 모두 UL2를 최대 3.6 포인트 앞서며, 이는 의미 있는 격차입니다.

생성형 LLM이 주류가 된 현재, 이후 분석은 PrefixLM 기반으로 진행하며, PrefixLM과 UL2의 결합 가능성은 다음 섹션에서 논의합니다.

### 인코더-디코더 LLM은 디코더 전용 LLM보다 특히 지시 튜닝 후 우수

표 2에서 볼 수 있듯, **적응된 인코더-디코더 LLM은 사전 학습 단계에서 디코더 전용 모델과 유사하거나 약간 더 나은 성능**을 보이지만, **지시 튜닝(instruction tuning) 이후에는 성능 격차가 크게 벌어집니다**.

- 예시:
  - 9B-9B 인코더-디코더 LLM은 Gemma 2 9B 대비 PT에서 1.4, IT에서 4.9 더 높음
  - 2B-2B 스케일에서는 이 격차가 PT에서 1.8, IT에서 7.1로 더 커짐

2B 이하 모델에서는 PT 성능이 다소 떨어질 수 있지만, IT 성능 개선은 여전히 유의미하며(예: XL-XL에서 7.2 포인트 상승) 실질적인 장점이 있습니다.

또한 PT/IT 모델, 사전 학습 목표, 모델 크기에 상관없이 **인코더-디코더 LLM은 SuperGLUE에서 일관되게 우수한 성능**을 보입니다. 이는 양방향 자기 어텐션 덕분에 문맥 표현 품질이 높아졌음을 시사합니다.

### 과제별 세부 분석의 필요성

위 분석은 **종합 성능 평균**에 기반하지만, 개별 다운스트림 과제에서는 결과가 다를 수 있습니다.

- 예: 사전 학습 후 ARC-C에서는 Gemma 2 9B가 9B-9B보다 4.1 포인트 높지만, Winogrande에서는 4.4 포인트 낮음
- 지시 튜닝 이후에도 일부 과제(예: WMT23)에서는 9B-9B가 Gemma 2 9B보다 0.9 포인트 낮음

이는 LLM 평가 시 **특정 과제 편향(bias)에 따른 잘못된 결론**을 피하기 위해 다양한 과제를 포함한 평가가 필요함을 보여줍니다.

### 품질-추론 효율성(quality-inference efficiency) 균형

![](/assets/images/posts/585/img_7.png)

![](/assets/images/posts/585/img_8.png)

![](/assets/images/posts/585/img_9.png)

#### 그림 3

- 인코더-디코더 모델은 **추론 FLOPs 대비 품질**에서 디코더 전용 모델보다 우수한 **품질-효율성 프런티어(frontier)**를 형성
- 예: 2B-2B 모델은 Gemma 2 2B와 비슷한 FLOPs에서 더 높은 품질 제공

![](/assets/images/posts/585/img_10.png)

#### 그림 4

- GSM8K 기준 지연(latency) 측정: 1배치(batch size 1), 200개 추론 질문 기준 ms 단위
- 9B-9B와 2B-2B는 Gemma 2 9B, 2B와 유사한 지연 시간에서 명확히 더 높은 성능 기록
- 특히 **9B-2B 모델(대형 인코더+소형 디코더 조합)**은 Gemma 2 2B와 유사한 지연 시간으로 2B-2B보다 훨씬 더 나은 성능 제공

### 결론

- 인코더-디코더 적응은 **품질과 추론 효율성의 균형을 맞추는 데 매우 유연한 접근**임을 확인
- 특히 **불균형 구조(큰 인코더-작은 디코더)**는 품질 손실 없이 지연을 줄이는 데 효과적임

### 6. 논의 (Discussion)

#### **적응 후 성능 향상이 단순히 추가 사전 학습 계산량 덕분인가?**

그렇지 않습니다. 우리는 **Gemma 2 2B 모델**에 추가로 6조 토큰을 더 학습시켜 PT 점수를 확인했는데, 48.57에 그쳤습니다. 반면, 인코더-디코더 적응 모델은 49.7로 여전히 더 높았습니다. 이는 단순히 계산량 증가만으로는 성능 향상을 설명할 수 없으며, **인코더-디코더 모델링의 귀납적 편향(inductive bias)**이 중요한 역할을 한다는 것을 시사합니다.

#### **불균형 인코더-디코더에서 교차 어텐션 워밍업이 중요한가?**

그렇습니다. **9B-2B 모델과 UL2**를 사용해 800B 토큰으로 사전 학습한 예비 실험에서, 워밍업 없이 학습하면 Boolq와 GSM8K의 PT 성능이 62.5에서 61.8로 감소했습니다. 또한, 워밍업 스텝을 1K에서 5K로 늘리면 성능이 60.2까지 더 떨어졌습니다. 이는 최적 성능을 달성하려면 **적절한 워밍업 스텝 수가 필요**함을 보여줍니다.

#### **인코더에서 Grouped-Query Attention(GQA) 대신 Multi-Head Attention(MHA)으로 전환 가능한가?**

가능하지만 결과는 엇갈립니다.

- Gemma 2는 디코딩 효율성을 높이기 위해 GQA를 사용합니다.
- 하지만 인코더는 추론 시 완전 병렬화가 가능하므로, MHA로 전환하는 것도 합리적입니다.
- **Gemma 2 2B 모델의 인코더 자기 어텐션을 GQA에서 MHA로 확장(헤드 파라미터 복제)한 결과:**
  - PrefixLM에서 PT 성능은 **50.2로 0.5 상승**
  - 그러나 IT 성능은 **43.5로 2.9 하락**
- 결론적으로 Gemma 2 2B와 9B의 적응에서는 여전히 GQA를 유지합니다.

#### **인코더에서 양방향 자기 어텐션이 중요한가?**

그렇습니다.

- 인코더-디코더와 디코더 전용 LLM의 핵심 차이는 **양방향 자기 어텐션(bidirectional self-attention)** 사용 여부입니다.
- 인코더 자기 어텐션을 인과적(causal)으로 유지한 2B-2B 모델을 테스트한 결과:
  - PT 점수: 45.6
  - IT 점수: 41.7
  - 이는 양방향 모델 대비 각각 **4.1, 4.7 낮은 성능**입니다.
- 주목할 점은, causal 2B-2B 모델도 여전히 **Gemma 2 2B 대비 IT에서 2.7 높은 점수**를 기록한다는 것입니다.
- 이는 **양방향 자기 어텐션이 적응 성공에 크게 기여**하지만, 그것만이 유일한 요인은 아님을 보여줍니다.

![](/assets/images/posts/585/img_11.png)

### 표 4

**PrefixLM으로 적응(Adaptation)한 인코더-디코더 모델**과 **처음부터 사전 학습(Scratch)한 모델**의 결과 비교

- **SG:** SFT 모델의 SuperGLUE 점수

![](/assets/images/posts/585/img_12.png)

![](/assets/images/posts/585/img_13.png)

![](/assets/images/posts/585/img_14.png)

### 그림 5

**2단계 최적화(two-stage optimization)**에 따른 품질 변화

- **UL2 → PrefixLM:** 마지막 10% 토큰 구간에서 학습 목표를 UL2에서 PrefixLM으로 전환
- **PrefixLM → UL2:** 반대로 PrefixLM에서 UL2로 전환

![](/assets/images/posts/585/img_15.png)

![](/assets/images/posts/585/img_16.png)

### 그림 6

**PT 성능과 해당 IT / SuperGLUE 성능 간의 상관관계 분석**

### **처음부터 인코더-디코더 LLM을 사전 학습하면 더 나은 성능을 낼 수 있을까?**

그렇지 않습니다. 새로운 LLM을 개발할 때 **처음부터 사전 학습(pretraining from scratch)**하는 방법은 흔히 사용됩니다. 우리도 PrefixLM을 사용해 8조 토큰으로 인코더-디코더 LLM을 처음부터 학습했습니다(표 4 참고). 그러나 더 많은 사전 학습 토큰을 사용했음에도 불구하고, **소규모 모델(S-S, B-B)에서만 소폭 성능 우위**를 보였고, 그 이상의 규모에서는 **적응(adaptation)** 방식이 확실히 더 우수했습니다. 따라서 적응 방식이 강력한 인코더-디코더 LLM을 개발하는 데 있어 훨씬 계산 효율적인 방법임을 확인했습니다.

### **PT 점수로 IT/SuperGLUE 점수를 예측할 수 있을까?**

부분적으로 그렇습니다. LLM 개발에서 일반적으로 **PT 성능이 다운스트림 응용 성능의 지표**로 사용될 수 있다는 가정이 있습니다. 우리는 모든 소거 실험(ablations)을 정리해 그림 6에 제시했습니다.

- **전체 데이터 포인트 및 모델 크기 전체**로 보면 상관관계는 매우 높습니다:
  - IT vs PT: Spearman ρ = 0.97
  - SuperGLUE vs PT: Spearman ρ = 0.89
- 그러나 **각 모델 크기 내 데이터 포인트별로 구분**해 보면,
  - IT vs PT 평균 ρ = 0.42
  - SuperGLUE vs PT 평균 ρ = 0.05
  - 즉, 유의미한 상관관계가 사라집니다.

실제로 PT 성능이 낮은 체크포인트가 IT나 SuperGLUE에서 더 높은 성능을 보이는 경우도 있습니다. 따라서 특정 모델 크기에서 체크포인트를 선택할 때는 **PT 결과만 보지 말고 IT 결과도 함께 확인**해야, 편향이나 과적합 문제를 피할 수 있습니다.

### **PrefixLM과 UL2를 모두 활용해 최적의 성능을 낼 수 있을까?**

쉽지 않습니다.

1. **체크포인트 병합 시도**
   - PrefixLM과 UL2로 각각 학습한 체크포인트를 동일 가중치로 병합해 보았으나,
   - 성능은 비슷하거나 오히려 나빠졌습니다.
   - 두 학습 목표는 **훈련 동역학과 수렴 지점(local minima)**이 매우 달라 단순 가중치 병합은 효과가 없었습니다.
2. **2단계 최적화(two-stage optimization) 시도**
   - PrefixLM → UL2 전환(마지막 10% 토큰): 성능 저하 발생
   - UL2 → PrefixLM 전환: IT 성능은 개선되지만 PT와 SuperGLUE 성능은 하락
3. **동시 최적화(joint optimization)**
   - PrefixLM과 UL2를 동시에 최적화하는 접근은 잠재적 방향이지만, 본 연구에서는 **미래 과제**로 남깁니다.
---

완전 자기 마음대로 해석하는 느낌인데. encoder decoder에서는 parameter가 더 많아서 좋아보이는 효과있다 -> 토큰을 동일한 모델에 더 학습시켰는데 그렇게 안됬다 이미 파라미터부터 잘못된거 아니야? 아예 새롭게 learning자체를 시켜야지?

ChatGPT의 말:

지금 지적하신 포인트는 굉장히 핵심적이에요. 논문이 주장하는 “encoder-decoder 구조 자체의 inductive bias로 인한 성능 향상”이라는 결론을 **파라미터 수 증가와 학습 토큰 분배**라는 변수로 검증했는지 살펴볼 필요가 있습니다. 정리해 드리면:
---

### 1. **논문 주장 vs 실제 구조**

- **논문 주장:**
  - Decoder-only LLM을 encoder-decoder로 변환(adaptation) → 성능 향상
  - 이는 “추가 pretraining compute 때문이 아니라 inductive bias 때문”이라고 강조
- **현실적 의문:**
  - Encoder-decoder 변환 시 **모델 파라미터가 2배로 증가** (예: 2B → 2B-2B = 4B)
  - 그러면 당연히 학습 용량(파라미터 수)이 커져서 성능이 좋아 보이는 효과가 있을 수 있음
  - 논문에서 비교 실험을 하긴 했지만, 동일한 파라미터 스케일로 엄밀하게 통제된 실험은 부족함
---

### 2. **논문에서 제시한 통제 실험**

- Gemma 2 2B 모델을 추가 6조 토큰 학습 → PT 48.57
- Encoder-decoder 적응(2B-2B) → PT 49.7
- 논문 해석: **추가 학습만으로는 성능 향상을 설명할 수 없다**

#### 하지만 문제점:

- 2B-2B는 **실제 파라미터 수가 4B**임 (인코더+디코더)
- 반면 Gemma 2 2B는 2B 파라미터 → 비교 자체가 동등하지 않음
- 파라미터 수와 학습 compute(토큰 수)를 분리해 분석하지 않았기 때문에 **진짜 inductive bias 때문인지 불명확**
---

### 3. **“그럼 아예 처음부터 학습해야 하나?”**

- 완전히 새로 학습하면 **모델 크기, 파라미터 구조, 학습 목표를 통제한 상태에서 비교 가능**
- 하지만 이 논문의 의도는 \*\*“기존 디코더-only LLM을 활용해 효율적으로 성능 향상”\*\*에 초점이 있어서
  - from-scratch 학습과 효율 비교에서는 adaptation이 더 실용적이라고 주장
  - 그러나 근본적인 모델 설계 우위를 검증하려면 **동일 파라미터 크기의 scratch 모델**과 비교가 필요
---

### 4. **내가 보기엔…**

- 이 논문은 효율성 측면에서는 의미 있지만, \*\*구조적 우월성(encoder-decoder inductive bias)\*\*을 입증하는 데는 불충분
- 진짜 bias 효과를 보려면:
  1. 동일 파라미터 수로 decoder-only vs encoder-decoder 비교
  2. 동일 compute로 학습된 scratch encoder-decoder vs adaptation encoder-decoder 비교
  3. objective(UL2 vs PrefixLM) 영향 분리
---

### 7. 결론 및 향후 연구 (Conclusion and Future Work)

본 논문에서는 **사전 학습된 디코더 전용 LLM을 기반으로 강력하고 범용적인 인코더-디코더 LLM을 구축하는 방법**을 제시했습니다. 이러한 적응(adaptation) 방식은 다양한 종류와 계열의 디코더 전용 모델을 활용하거나, 서로 다른 크기의 모델을 조합하는 데 높은 유연성을 제공합니다.

Gemma 2를 기반으로 한 광범위한 실험을 통해, 우리는 다음을 입증했습니다.

- 제안한 적응 방식은 **인코더-디코더 LLM이 지시 튜닝(instruction tuning) 이후 디코더 전용 모델보다 현저히 우수**한 성능을 달성하게 하며,
- 품질-추론 효율성(quality-inference efficiency) 트레이드오프에서도 우위를 점합니다.  
  또한, 인코더-디코더 LLM은 **SuperGLUE 평가에서 더 나은 문맥 표현(contextual representation)**을 제공합니다.

우리는 본 연구 결과가 학계와 산업계 연구자들에게 **LLM 개발에서 인코더-디코더 패러다임을 재조명**하는 계기가 되길 바랍니다. 이를 위해 코드와 체크포인트를 XXX(곧 공개 예정)에 배포할 계획입니다.

**한계점 및 향후 연구 방향**







우리의 연구에는 몇 가지 한계가 존재합니다.

- Gemma 2 모델(최대 9B)까지만 실험했으며, 제안 방법은 다른 LLM 계열에도 적용 가능하지만 이를 검증하지는 못했습니다.
- 향후에는 모델 크기를 27B 수준으로 확장하고, LLaMA 등 다른 LLM에 대한 실험, 더 다양한 불균형 설정 탐구, 밀집(dense) 모델과 MoE(Mixture-of-Experts) 모델의 조합 검증을 계획하고 있습니다.
- 또한 PrefixLM, 지식 증류, UL2를 더 효과적으로 활용할 방법을 모색할 예정입니다.
- 마지막으로, **비전-언어, 음성-언어**와 같은 **크로스/멀티모달리티 모델링**으로의 확장도 흥미로운 연구 방향이 될 것입니다.

### 감사의 글 (Acknowledgements)

Enrique Alfonseca, Tris Warkentin, Xiaodan Song, Sugato Basu, Inderjit Dhillon, Alexander Grushetsky, Pandu Nayak, Ramakrishnan Srikant, 그리고 Slav Petrov에게 원고에 대한 건설적인 피드백을 제공해 주셔서 감사드립니다. 또한 이 프로젝트를 지원해 준 Srinivasan Venkatachary에게 감사를 표합니다.
