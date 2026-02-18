---
title: "SmolLM3: smol, multilingual, long-context reasoner"
date: 2025-07-21 15:18:01
categories:
  - 인공지능
tags:
  - smollm3
---

<https://huggingface.co/blog/smollm3>

[SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3)

소형 언어 모델(Small Language Models)은 점점 더 중요한 위치를 차지하고 있습니다. 사용자는 효율적으로 배포할 수 있으면서도 강력한 성능을 가진 모델을 원하고 있기 때문입니다. 커뮤니티는 이와 같은 요구에 부응하여, 이 규모에서도 가능한 한계를 계속해서 넓혀가는 다양한 소형 모델들을 만들어냈습니다.

SmolLM3는 이러한 흐름 속에서 새로운 경쟁력 있는 완전 공개(open) 30억 파라미터(3B) 모델로 기여하고자 합니다.

- **Base 모델**: <https://hf.co/HuggingFaceTB/SmolLM3-3B-Base>
- **Instruct 및 추론 특화 모델**: <https://hf.co/HuggingFaceTB/SmolLM3-3B>

SmolLM3는 **효율성과 성능의 균형점**에 위치한 모델입니다. 이 30억 파라미터 모델은 Llama-3.2-3B와 Qwen2.5-3B를 능가하며, 40억 파라미터급 모델인 Qwen3와 Gemma3와도 경쟁할 수 있는 성능을 보여줍니다.

또한, 단순한 성능 수치 그 이상으로, 우리는 **공개 데이터셋과 훈련 프레임워크**를 사용하여 SmolLM3를 어떻게 구축했는지 그 과정을 투명하게 공유하고자 합니다.

![](/assets/images/posts/583/img.png)

모델 요약:

- **30억(3B) 파라미터 모델**, 총 **11조(11T) 토큰**으로 학습됨
- **3B 규모에서 SoTA(최첨단 성능)**을 달성했으며, **4B 모델들과도 경쟁 가능**
- **지시 따르기(Instruct) 특화 모델**로, **이중 추론 모드(dual mode reasoning)** 지원 — think / no\_think 모드
- **6개 언어 지원**: 영어, 프랑스어, 스페인어, 독일어, 이탈리아어, 포르투갈어
- 최대 **128k 토큰의 장문 컨텍스트** 처리 가능 — **NoPE**와 **YaRN** 기술 사용

**전체 설계 레시피도 함께 공개합니다.** SmolLM3는 단순한 모델 공개를 넘어, **아키텍처 세부 사항**, **도메인별 성능을 점진적으로 향상시키는 3단계 사전학습 전략에서의 정확한 데이터 혼합 비율**, 그리고 **하이브리드 추론 모델을 구성하는 방법론**까지 모두 포함한 **엔지니어링 청사진(blueprint)**을 제공합니다.

보통 이런 수준의 결과를 얻기 위해서는 수개월의 리버스 엔지니어링이 필요하지만, 우리는 이를 대신해 **전체 방법론을 투명하게 공개**합니다.

![](/assets/images/posts/583/img_1.png)

당신이 직접 모델을 구축하고 있든, 아니면 이 규모에서의 성능을 끌어올리는 핵심 요소가 무엇인지 이해하고 싶든, 이 블루프린트는 **경쟁력 있는 3B 모델이 어떻게 만들어졌는지에 대한 엔지니어링 스토리**를 보여줍니다.

이제 사전학습(Pretraining) 단계로 들어가 보겠습니다.

### 사전학습 (Pretraining)

SmolLM3는 이전 버전들과 비교해 **아키텍처**와 **데이터 혼합 구성(data mixture)** 모두에 변화를 주었습니다. 먼저, **모델 아키텍처와 학습 설정(training configuration)**부터 살펴보겠습니다!

### 아키텍처 및 학습 세부사항

![](/assets/images/posts/583/img_2.png)

SmolLM3는 **SmolLM2와 유사한 tied embedding 기반 Transformer 디코더 아키텍처**를 따르며, **Llama 아키텍처**를 기반으로 몇 가지 핵심적인 변경 사항을 통해 **효율성과 장문 컨텍스트 성능**을 최적화하였습니다.

#### ✅ Grouped Query Attention (GQA)

기존의 multi-head attention을 **4개의 그룹으로 나눈 grouped-query attention**으로 대체했습니다. 3B 모델을 FineWeb-Edu 데이터 1,000억(100B) 토큰으로 학습한 결과, **성능은 기존 multi-head attention과 유사하면서도 추론 시 KV 캐시 크기를 크게 줄이는 효과**를 확인했습니다.

#### ✅ NoPE

논문 "RoPE to NoRoPE and Back Again: A New Hybrid Attention Strategy" (Yang et al., 2025)에서 제안된 **NoPE 기법**을 적용했습니다. 구체적으로는 **매 4번째 레이어마다 rotary position embedding(RoPE)을 제거**하여,

- **짧은 컨텍스트 성능은 유지하면서도**
- **긴 컨텍스트 처리 성능은 향상**됨을 어블레이션 실험을 통해 확인했습니다.

#### ✅ 문서 간 마스킹 (Intra-Document Masking)

하나의 학습 시퀀스 안에 포함된 **서로 다른 문서의 토큰들이 서로 attend하지 않도록** attention 마스킹을 적용했습니다. 이는 **Llama 3와 유사한 방식**으로,

- **장문 컨텍스트 학습의 안정성과 속도**를 높이고
- 짧은 컨텍스트 성능은 그대로 유지할 수 있게 해줍니다.

#### ✅ 학습 안정성 개선

OLMo 2를 따라, **임베딩 레이어에 대해 weight decay를 제거**하여 학습 안정성을 높였습니다. 이 변경은 **임베딩 가중치의 L2 노름(norm)을 자연스럽게 안정적인 수준으로 수렴**시키며, 성능 저하 없이 더 **일관되고 안정적인 학습**을 가능하게 했습니다.

이러한 모든 변경 사항은 **동일한 3B 아키텍처**에 대해 **FineWeb-Edu 100B 토큰으로 학습하여 어블레이션 검증**을 수행한 결과이며, 각 변경이 **성능을 유지하거나 향상시키면서 추가적인 이점을 제공함**을 확인했습니다.

### 학습 구성 (Training Configuration)

- **시퀀스 길이**: 4,096
- **글로벌 배치 크기**: 2.36M 토큰
- **러닝레이트**: 2e-4
- **옵티마이저**: AdamW (beta1=0.9, beta2=0.95)
- **Weight decay**: 0.1
- **Gradient clipping**: 1
- **스케줄러**: WSD (Warmup-Stable-Decay)
  - 워밍업 스텝: 2,000
  - 마지막 10% 스텝 동안 선형 감소(linear decay)
- **학습 프레임워크**: nanotron
- **데이터 처리**: datatrove
- **평가 도구**: lighteval
- **하드웨어**: **H100 GPU 384개**, 총 **24일간 분산 학습**

> 아래 그림에서는 분산 학습 구성(distributed training setup)을 확인할 수 있습니다.

![](/assets/images/posts/583/img_3.png)

아키텍처 변경 외에도, **전체 학습 레시피 자체를 어블레이션하고 개선**하였습니다. 다음 섹션에서는 그 **학습 레시피의 세부 조정 과정**을 더 자세히 살펴보겠습니다.

### 데이터 혼합 구성 및 학습 단계

SmolLM3는 SmolLM2에서 사용한 **다단계(multistage) 학습 전략**을 계승하여, 총 **11.2조(11.2T) 토큰**을 활용한 **3단계 사전학습 전략**을 따릅니다. 이 과정에서 **웹(Web)**, **수학(Math)**, **코드(Code)** 데이터를 비율을 달리하며 조합했고, 각 단계별 데이터 구성은 **3B 모델을 500억 ~ 1,000억(50B ~ 100B) 토큰으로 학습한 어블레이션 결과**를 통해 최적화하였습니다.

![](/assets/images/posts/583/img_4.png)

### 사전학습 구성 단계 (상단 그림 참조)

#### ? **1단계: 안정화 단계 (Stable Phase)**

**0T → 8T 토큰**  
모델의 **기초 언어 능력**을 구축하기 위한 단계로, 핵심 데이터셋으로 구성된 안정적인 혼합비를 사용합니다.

- **웹(Web)**: **85%** (그 중 **12%는 다국어**)
  - FineWeb-Edu, DCLM, FineWeb2, FineWeb2-HQ
- **코드(Code)**: **12%**
  - The Stack v2 (16개 프로그래밍 언어)
  - StarCoder2 PR, Jupyter/Kaggle 노트북, GitHub 이슈, StackExchange
- **수학(Math)**: **3%**
  - FineMath3+, InfiWebMath3+

#### ? **2단계: 안정화 단계 (Stable Phase)**

**8T → 10T 토큰**  
더 **고품질의 수학 및 코드 데이터셋**을 도입하여 성능 향상을 유도하면서, 웹 커버리지도 유지합니다.

- **웹(Web)**: **75%** (12% 다국어)
- **코드(Code)**: **15%**
  - Stack-Edu 데이터 추가
- **수학(Math)**: **10%**
  - FineMath4+, InfiWebMath4+, MegaMath
  - (Qwen QA, Pro synthetic rewrites, 텍스트-코드 혼합 블록 포함)

#### ? **3단계: 감쇠 단계 (Decay Phase)**

**10T → 11.1T 토큰**  
최종 단계에서는 **수학 및 코드 데이터를 더욱 업샘플링(oversample)**하여 학습합니다.

- **웹(Web)**: **63%** (12% 다국어)
- **코드(Code)**: **24%**
  - 고품질 코드 데이터의 업샘플링
- **수학(Math)**: **13%**
  - 수학 데이터 업샘플링
  - OpenMathReasoning 등 **지시 따르기 및 추론 데이터셋** 추가

이러한 단계별 학습과 정교한 데이터 혼합 전략 덕분에, **기반(base) 모델만으로도 매우 경쟁력 있는 성능**을 달성할 수 있었습니다.  
평가 결과는 다음 섹션에서 더 자세히 설명됩니다.

> 정확한 데이터 가중치가 포함된 nanotron 학습 설정(config)은 [여기](<https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs>)에서 확인할 수 있으며,  
> 학습 로그 및 중간 체크포인트도 함께 공개될 예정입니다.

사전학습이 완료된 후에는, **장문 컨텍스트 처리 능력 및 추론 성능 향상**을 위한 **mid-training 단계**가 이어집니다.

### Mid-training (중간 학습 단계)

우리는 **긴 컨텍스트 적응(long context adaptation)**과 **추론 능력 강화(reasoning adaptation)** 과정을 통틀어 **“Mid-training”**이라 부릅니다. 이는 메인 사전학습(main pretraining)에 비해 훨씬 짧지만, 여전히 **일반화된 학습**으로 간주되며 두 영역에서 모델의 성능을 개선하는 데 목적이 있습니다. 먼저, **긴 컨텍스트 학습(long context training)**부터 살펴보겠습니다.

### ? Long Context 확장

![](/assets/images/posts/583/img_5.png)

메인 사전학습 이후, **추가로 1,000억(100B) 토큰**을 학습시켜 **컨텍스트 길이를 확장**하였습니다. 총 두 단계로 나뉘며, 각 단계는 500억(50B) 토큰 학습으로 구성됩니다:

1. **4k → 32k**
   - RoPE의 **theta 값**을 **1.5M**으로 증가
2. **32k → 64k**
   - RoPE **theta**를 **5M**으로 다시 증가

이 두 단계 모두에서 **수학, 코드, 추론 데이터**를 **업샘플링(oversampling)**하여 학습하였습니다.

> 어블레이션 실험 결과, 코드 저장소, 책, 긴 웹 페이지 등 **“특정한 long-context 데이터셋”을 추가로 업샘플링하는 것은 오히려 성능 향상에 기여하지 않음**을 확인했습니다. 대신, **기존 decay mixture에 기반한 장문 시퀀스 + 증가된 RoPE theta**만으로도 RULER 및 HELMET 벤치마크에서 **경쟁력 있는 장문 성능(최대 64k)**을 달성할 수 있었습니다.

이후 Qwen2.5와 마찬가지로, **YARN**을 활용해 학습한 컨텍스트 길이를 **초과하여 extrapolation**을 적용하였으며, **추론 시에는 최대 128k 토큰까지 처리 가능**합니다 (학습 시 최대 64k의 2배 확장).

### ? Reasoning Mid-training (추론 학습 단계)

모델의 컨텍스트 확장이 완료된 후, **추론 능력 향상**을 위한 **mid-training 단계**를 별도로 진행하였습니다.

이 단계는 Pretraining 및 후속 SFT(Post-training)와 달리,

- 특정 도메인(수학, 코드 등)에 특화하지 않고,
- **일반적인 추론 능력 자체를 강화**하는 것이 목적입니다.

#### ? 사용된 데이터셋:

- 총 **350억(35B) 토큰**
  - Open Thought의 **OpenThoughts3-1.2M**
  - NVIDIA의 **Llama-Nemotron-Post-Training-Dataset-v1.1** 중 **R1 reasoning trace** 포함된 일부 샘플

#### ? 학습 설정:

- **ChatML 템플릿** 및 **packing** 기법 사용  
  → 모델에 과도한 구조를 주지 않도록 주의
- 총 **4 epoch**, 약 **1,400억(140B) 토큰**에 해당
- 최종적으로 **이 Mid-training 체크포인트를 이후 SFT의 시작점으로 사용**

이처럼 SmolLM3는 **긴 컨텍스트**와 **추론 능력** 모두를 mid-training 단계를 통해 체계적으로 강화하였으며, 이는 이후의 지시 따르기(SFT) 및 평가 단계에서 그 효과가 반영됩니다.

### ? 사후 학습 (Post-training)

DeepSeek R1, Qwen3와 같은 **추론 특화 모델들**의 등장은, 모델이 **명시적인 추론(explicit reasoning)**을 수행할 수 있을 때 **강력한 성능**을 발휘할 수 있음을 보여주었습니다.

하지만 여전히 커뮤니티에는

- **추론 모드와 비추론(non-reasoning) 모드 모두를 지원하는 이중 지시 모델(dual instruction model)**을
- **공개 데이터셋 기반**으로 **재현 가능한 방식**으로 구축한 **완전 공개 레시피**가 부족한 상황입니다.

기존 접근법 대부분은 복잡한 **강화학습(RL)** 프로세스와 **비공개 데이터셋**에 의존하고 있어, 연구자들이 이를 재현하고 확장하기 어렵다는 한계가 있습니다.

### ? 우리의 접근법과 전체 설계 공개

이 섹션에서는 이러한 문제를 어떻게 해결했는지 설명하고, **SmolLM3의 dual instruction model을 구축하기 위한 전체 레시피**를 공유합니다. 우리는 다음과 같은 학습 파이프라인을 통해 **추론 모드와 비추론 모드 간의 성능 균형을 유지**합니다:

1. **Mid-training**: 일반적인 추론 능력을 학습
2. **Supervised Fine-tuning (SFT)**: **합성 데이터(synthetic data)**를 생성해 정제된 지시 학습 수행
3. **Anchored Preference Optimization (APO)**: 최근 등장한 **DPO(Direct Preference Optimization)의** 변형으로, 모델 정렬(alignment)을 수행

![](/assets/images/posts/583/img_6.png)

## ? Chat 템플릿 설계

학습 방법론에 들어가기 앞서, 사용자가 **이중 모드 모델과 어떻게 상호작용하는지**를 정의하는 **채팅 템플릿(chat template)**을 먼저 살펴보는 것이 중요합니다. 이 템플릿은 **추론 모드 / 비추론 모드 간의 전환**을 가능하게 해주며, 학습 데이터 형식뿐 아니라 **모델의 실제 동작에도 직접적인 영향을 미칩니다.**

### ? SmolLM3의 Chat Template 구조

- **사용자 제어 방식**:  
  시스템 프롬프트(system prompt)에 다음과 같은 플래그를 삽입해 모드를 제어할 수 있습니다.
  - /think → **추론 모드 활성화**
  - /no\_think → **비추론 모드 활성화**
- **비추론 모드 처리 방식**:  
  Qwen3와 유사하게, **모델 응답 내에 비어 있는 think 블록**을 미리 삽입(pre-fill)하여 **추론 없이 직접적인 답변을 유도**합니다.

### ?️ 툴 호출(툴 사용 기능) 지원

SmolLM3는 **툴 호출(tool calling)** 기능을 지원하며, 이를 위해 **툴 설명 텍스트**를 다음 두 가지 포맷으로 명확히 구분해 템플릿에 포함시킵니다:

- **XML Tools**
- **Python Tools**

→ 이러한 명확한 구분은 툴 정의 해석의 정확도를 높이는 데 효과적이었습니다.

### ? 템플릿 기본 구조

- **기본 System 메시지**: reasoning 모드별로 **기본 시스템 메시지(system message)**가 제공됨
- **메타데이터 섹션 포함**:
  - 날짜
  - knowledge cut-off 시점
  - 현재 reasoning 모드
- **유연한 오버라이드** 기능:  
  사용자가 직접 시스템 메시지를 입력하면 기본 메시지를 대체 가능  
  /system\_override 플래그를 사용하면 **메타데이터 섹션 제거** 가능  
  → 특정 목적에 따라 유연한 템플릿 구성이 가능함

이처럼 SmolLM3는 **완전 공개된 지시 따르기 모델 중 드물게**,

- **명시적인 reasoning과 간결한 direct answer**를 **모두 지원하는 이중 모드 구조**,
- **툴 사용 지원**,
- **사용자 제어가 가능한 chat 템플릿 설계**를 통해 현실적인 사용성과 재현 가능성을 함께 달성하였습니다.

### ? Supervised Finetuning (지도 미세조정)

**1400억(140B) 토큰의 일반 추론 데이터**로 mid-training을 수행한 이후, 이제 **지도 학습(Supervised Finetuning, SFT)** 단계로 넘어갑니다.

이 단계의 목표는,

- **추론 모드(reasoning)**와 **비추론 모드(non-reasoning)** 모두에 대해
- **수학, 코드, 일반 추론, 지시 따르기, 다국어 처리, 툴 호출** 능력을 통합하는 것입니다.

### ⚖️ 이중 모드 모델을 위한 데이터 밸런싱

**이중 모드(dual-mode)** 모델을 학습하려면,

- 두 모드 각각의 성능이 떨어지지 않도록
- **모든 도메인에서 균형 있는 데이터 혼합 구성(data mixture)**이 필수입니다.

우리는 SFT 훈련 전반에서 다음 다섯 가지 도메인의 성능을 추적하여 밸런스를 조정했습니다:

1. 수학 (Math)
2. 코드 (Code)
3. 일반 추론 (General Reasoning)
4. 지시 따르기 (Instruction Following)
5. 다국어 처리 (Multilinguality)

### ? 주요 도전과제: Reasoning 데이터 부족

추론 모드용 데이터셋을 구축하면서 가장 큰 어려움은

- **일부 도메인에서 reasoning trace(추론 경로)를 포함한 공개 데이터셋이 매우 부족**하다는 점이었습니다.

이를 해결하기 위해,

- 기존 **비추론 데이터셋**의 프롬프트를 활용해
- **Qwen3-32B 모델을 추론 모드로 프롬프트**하여
- **합성 reasoning 데이터**를 생성했습니다.

이 접근은 특히 다음 영역의 추론 성능 개선에 효과적이었습니다:

- 멀티턴 대화 (Multi-turn Conversations)
- 다국어
- 일상적 대화

![](/assets/images/posts/583/img_7.png)

### ? 최종 데이터 혼합 구성

수많은 어블레이션 실험을 통해,

- **reasoning vs. non-reasoning 간의 비율**
- **각 모드 내부 구성 데이터 비중**  
  을 최적화하였습니다.

최종 SFT 데이터 구성은 다음과 같습니다:

- **총 18억(1.8B) 토큰**
  - **비추론 모드**: **10억(1B)** 토큰
    - 12개의 비추론 데이터셋
  - **추론 모드**: **8억(0.8B)** 토큰
    - 10개의 reasoning trace 포함 데이터셋
- **훈련 방식**:
  - **BFD(Best-Fit Decreasing)** packing
  - **user turn 및 tool call 결과에 대해 loss masking** 적용
  - **총 4 에폭**, 약 **8억(8B) 토큰 분량**

### ? 완전 공개 예정

이 SFT 데이터 구성과 전체 학습 스크립트는 **커뮤니티가 직접 재현하고 확장할 수 있도록 완전히 공개**할 예정입니다.  
→ SmolLM3는 **재현성, 투명성, 실용성**이라는 세 가지 측면 모두에서 진정한 오픈 모델로 자리매김하고자 합니다.

### ? Anchored Preference Optimization (APO)를 통한 오프폴리시 모델 정렬

SFT(Supervised Fine-Tuning) 이후, 우리는 **모델 정렬(Model Alignment)**을 위해 한 차례 추가 학습을 수행했습니다.  
이때 사용된 방식은 다음과 같습니다:

- **비추론(non-reasoning) 모드**:
  - **Tulu3의 preference 데이터셋** 사용
- **추론(reasoning) 모드**:
  - **Qwen3-32B와 Qwen3-0.6B로 생성한 합성 선호쌍(synthetic preference pairs)** 사용
  - 비추론 데이터셋의 커버리지를 보완하기 위해, **thinking 모드에 대응하는 쌍(pair)을 새로 생성**

> 정렬 시에는 Qwen3-32B의 출력을 **"chosen"**, Qwen3-0.6B의 출력을 **"rejected"**로 설정하여  
> **Anchored Preference Optimization (APO)** 방식으로 학습을 진행했습니다.

![](/assets/images/posts/583/img_8.png)

**Anchored Preference Optimization (APO)**는 **Direct Preference Optimization (DPO)**의 변형 기법으로, **더 안정적인 최적화 목표(objective)**를 제공합니다.

### ? DPO에서의 보상 함수

DPO에서 사용되는 **보상 함수** r\_θ(x, y)는 다음을 계산합니다:

> **학습 중인 모델이 특정 응답 y를 생성할 확률**과 **학습 초기 시점의 참조 모델(reference model)**이 같은 응답을 생성할 확률 간의 **로그 비율(log-ratio)**

![](/assets/images/posts/583/img_9.png)

![](/assets/images/posts/583/img_10.png)

여기서 **β(베타)**는

> **최적화 중인 모델이 참조 모델(reference model)에 비해 얼마나 변화할 수 있는지를 조절하는 하이퍼파라미터**입니다.

![](/assets/images/posts/583/img_11.png)

![](/assets/images/posts/583/img_12.png)

APO 목적 함수는 **더 높은 학습 안정성**을 보여주었으며, 우리의 내부 어블레이션 실험에서도 **다운스트림 성능이 향상됨**을 확인할 수 있었습니다.

![](/assets/images/posts/583/img_13.png)

수학, 과학, 지시 따르기, 코딩, 대화, 다국어 작업 등 다양한 **다운스트림 평가에서는 성능 향상**이 나타났지만, **RULER와 같은 장문 컨텍스트 벤치마크(long context benchmark)**에서는 **성능 저하**가 관찰되었습니다.

이 성능 저하의 원인을 추적한 결과,

- **Reasoning Mid-training 단계**에서 **추론 능력에 초점을 맞춘 학습**이
- **장문 컨텍스트 처리 성능에 부정적인 영향을 미쳤음**을 확인했습니다.

또한,

- **APO 학습 데이터의 시퀀스 길이가 대부분 24k 토큰 이하**였기 때문에,
- reasoning 데이터셋의 **대다수가 장문 처리를 위한 길이를 충분히 제공하지 못한 점**도 영향을 미쳤습니다.

### ? 해결 방안: 모델 병합(Model Merging)

이러한 성능 저하를 완화하기 위해, 우리는 **모델 병합(model merging)**을 하나의 해결책으로 탐색하였습니다.

### ? 모델 병합 (Model Merging)

**모델 병합(Model Merging)**은

- 서로 다른 모델의 **강점을 결합**할 수 있으면서도
- **앙상블처럼 추가 연산 비용이 없고**,
- **별도의 재학습 없이도 활용 가능한 강력한 기법**입니다.

우리는 이를 위해 **MergeKit 라이브러리**를 사용했으며, MergeKit은 **선형(linear)** 및 **비선형(non-linear)** 병합 방식 등 다양한 방법을 지원합니다.

### ? 병합 전략(Recipe)

우리의 모델 병합 과정은 **두 단계로 구성**되어 있습니다:

1. **APO 체크포인트들을 결합해 "모델 수프(model soup)" 생성**
   - APO를 통해 정렬된 여러 버전의 모델들을 평균화하여 하나의 수프 모델로 만듭니다.
2. 이렇게 생성된 모델 수프를,
   - **장문 컨텍스트 성능이 뛰어난 Mid-training 체크포인트**와 병합
   - 병합 비율은 다음과 같음:
     - **APO 모델 수프: 0.9**
     - **Mid-training 체크포인트: 0.1**

이 **선형 병합(linear merge)** 구성은 가장 우수한 성능을 보여주었으며,

- **최대 128k 토큰까지의 문맥에서**
- **기존 base 모델 수준의 RULER 점수를 회복**할 수 있었습니다.

### ? 최종 결과 모델

이렇게 병합된 모델이 바로 **오늘 공개하는 체크포인트**입니다.

- 다양한 태스크에서의 **성능을 고르게 유지**하면서,
- 장문 컨텍스트와 추론 능력 간의 **트레이드오프를 효과적으로 극복**한 모델입니다.

이제,

- 이 모델과
- Base 모델의 **평가 결과(Evaluation Results)**를 살펴보겠습니다.

### ? 평가 (Evaluation)

우리는 SmolLM3의 **Base 모델**과 **Instruct 모델** 모두를 평가했으며, **Instruct 모델은 추론 모드와 비추론 모드 각각에 대해 평가**를 진행했습니다. 먼저, **Base 모델의 성능**부터 살펴보겠습니다.

![](/assets/images/posts/583/img_14.png)

## ? Base 모델 성능

아래 그래프는 SmolLM3가 **지식, 추론, 수학, 코딩 능력**을 평가하는 **12개의 주요 벤치마크**에서 기록한 **승률(Win Rate)**을 나타냅니다.

SmolLM3는:

- **동급 3B 모델들을 일관되게 능가**하고
- **Qwen3-4B**, **Gemma3-4B**와 같은 **4B 모델들과도 경쟁력 있는 성능**을 보여주었습니다.

![](/assets/images/posts/583/img_15.png)

### ? 사용된 평가 벤치마크:

- **지식/상식/추론**:
  - HellaSwag, ARC, Winogrande, CommonsenseQA
  - MMLU-CF, MMLU Pro CF, BoolQ
  - PIQA, OpenBookQA
- **수학/코딩**:
  - GSM8K, MATH
  - HumanEval+, MBPP+

### ? 주요 결과 요약

- **지식 및 추론 벤치마크** (HellaSwag, ARC, BoolQ)에서 **1위 또는 2위**를 기록  
  → **핵심 추론 능력에서 매우 강력한 성능**
- **수학 및 코딩 성능**  
  → 3B 클래스 내에서는 **최상위권 경쟁력**
- **RULER 64k 기준 장문 컨텍스트 평가**  
  → **긴 시퀀스 처리 능력 우수**

![](/assets/images/posts/583/img_16.png)

## ? 다국어(Multilingual) 성능

SmolLM3는 다음과 같은 **다국어 벤치마크**에서 **5개 주요 유럽어(영어, 프랑스어, 스페인어, 독일어, 이탈리아어)**에 대해 **강력한 성능을 입증**했습니다:

- Global MMLU
- MLMM HellaSwag
- Flores-200
- Belebele

이 벤치마크들은 **지식, 상식 추론, 문장이해, 번역** 등을 종합적으로 평가하며, SmolLM3는 **영어 이외 언어에서도 일관된 성능**을 유지합니다.

### ✅ 요약

- **SmolLM3 Base 모델**은
  - 지식, 추론, 수학, 코딩, 다국어 등 **다양한 도메인에서 매우 강력한 성능**을 보이며
  - 3B 모델 중에서는 독보적인 수준
  - 4B 모델과도 대등한 성능을 달성했습니다.

### ? 이중 모드 Instruct / Reasoning 모델 평가

SmolLM3는 **Instruct 모드**와 **Reasoning 모드**를 모두 지원하기 때문에, 두 모드 각각에서의 성능을 평가하고, **유사한 기능을 가진 모델들과 비교**할 필요가 있습니다.

## ? **No Thinking 평가** (비추론 모드 평가)

먼저, **비추론 모드(no\_think)**에서 SmolLM3의 성능을 다른 **3B급 비추론 모델**들과 비교하였습니다. 또한, **Qwen3의 reasoning 모델을 no\_think 모드로 작동시킨 결과**와도 비교했습니다.

? **결과 요약 (성능 차트 기준):**

- SmolLM3는 다음 모델들을 능가함:
  - **Llama3.2 3B Instruct**
  - **Qwen2.5 3B Instruct**
- 효율성 면에서는:
  - **Qwen3 1.7B reasoning 모델**보다 **확실한 성능 우위**
  - **Qwen3 4B 수준의 성능**에 가까우면서도, **연산 비용은 훨씬 낮음**

![](/assets/images/posts/583/img_17.png)

### ⚖️ Pareto 최적점에 위치한 Instruct 모델

이러한 결과를 종합하면, SmolLM3의 Instruct 모델은 **성능과 비용 간 trade-off에서 Pareto front(**

**Pareto front**는 **두 개 이상의 상충(trade-off)하는 목표 사이에서, 한쪽을 개선하면 다른 쪽은 반드시 악화되는 최적의 경계선**을 의미합니다.

**)에 위치**하며,

- **경량 모델로서의 효율성**과
- **범용 instruct 성능**을 모두 만족시킵니다.

이제, 다음으로는 **Reasoning 모드(추론 활성화)**에서의 성능을 살펴보겠습니다.

### ? **Thinking 모드 확장 평가** (추론 모드 성능)

SmolLM3의 **추론 모드(think 모드)**를 활성화하고 평가한 결과, **비추론 모드에 비해 대부분의 벤치마크에서 성능이 크게 향상**되는 것을 확인할 수 있었습니다.

![](/assets/images/posts/583/img_18.png)

⚖️ Qwen3-4B와의 비교

- Qwen3 4B는 여전히 **추론/비추론 모드 모두에서 가장 높은 점수**를 기록했지만,
- SmolLM3는 **3B 파라미터 클래스에서는 매우 경쟁력 있는 성능**을 보여주며,  
  특히:
  - **수학적 추론(Mathematical Reasoning)**
  - **복잡한 문제 해결(Complex Problem Solving)** 영역에서 두각을 나타냈습니다.

### ? SmolLM3의 이중 모드 장점

SmolLM3는 다음과 같은 **유연한 활용이 가능**합니다:

- **빠른 추론이 필요한 경우**: /no\_think 모드로 **속도 우선**
- **깊이 있는 분석이 필요한 경우**: /think 모드로 **정밀 추론 수행**

![](/assets/images/posts/583/img_19.png)

### ❓마지막 질문: **이 모델을 어떻게 사용할 수 있을까?**

→ 다음 섹션에서는 **SmolLM3를 실제로 어떻게 활용할 수 있는지** 자세히 안내합니다.

### ? SmolLM3 로컬 실행 방법 (How to run locally)

SmolLM3는 **transformers 라이브러리 v4.53.0 이상**에서 사용할 수 있도록 제공됩니다. 또한, **vLLM 최신 버전**에서도 실행 가능하며, 내부적으로 transformers를 백엔드로 사용합니다.

✅ 1. 라이브러리 설치 및 버전 업그레이드

```
pip install -U transformers
```

✅ 2. 모델 및 토크나이저 불러오기

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda"  # GPU 사용 시 "cuda", CPU만 사용할 경우 "cpu"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
```

✅ 3. 입력 프롬프트 준비 및 템플릿 적용

```
prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [
    {"role": "user", "content": prompt}
]

# chat 템플릿 적용
text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
```

✅ 4. 출력 생성 및 디코딩

```
# 출력 생성 (최대 32768 토큰 생성 가능)
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# 새로 생성된 토큰만 추출 및 디코딩
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### ? 권장 생성 파라미터

- temperature=0.6
- top\_p=0.95

이 파라미터를 model.generate()에 추가하면 보다 **다양하면서도 안정적인 생성 결과**를 얻을 수 있습니다.

```
model.generate(
    **model_inputs,
    max_new_tokens=32768,
    temperature=0.6,
    top_p=0.95,
)
```

이렇게 하면 SmolLM3를 로컬에서 손쉽게 실행하고 활용할 수 있습니다!

? 확장 추론 모드 활성화 및 비활성화

(SmolLM3의 /think, /no\_think 사용법)

SmolLM3는 기본적으로 **확장 추론 모드(extended thinking)**가 **활성화되어 있습니다**. 즉, 앞서 소개한 기본 코드 예시는 **추론 과정을 포함한 응답(reasoning trace)**을 생성합니다.

### ? 모드 전환 방법

#### ? 비추론 모드 (추론 비활성화: /no\_think 사용 예시)

```
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

#### ? 추론 모드 활성화 (기본 상태 또는 /think 사용)

- 위 코드에서 /no\_think를 **/think**로 바꾸면 **추론 활성화**
- 또는 **시스템 프롬프트를 생략**하면 기본적으로 /think가 적용됨

## ? 에이전트 활용 (Agentic Usage)

SmolLM3는 **툴 호출(tool calling)** 기능도 지원합니다! 사용자는 아래와 같이 **툴 리스트를 전달**하여 에이전트처럼 사용할 수 있습니다.

### ✨ 지원되는 툴 타입:

- xml\_tools: 일반적인 표준 툴 정의용
- python\_tools: Python 함수 스타일로 <code> 블록 호출 가능

?️ 툴 호출 예제

```
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

tools = [
    {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for"
                }
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Hello! How is the weather today in Copenhagen?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    enable_thinking=False,  # True도 가능, 상황에 따라 선택
    xml_tools=tools,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### ✅ 결론 (Conclusion)

우리는 **SmolLM3**를 공개합니다. 이 모델은 다음과 같은 특징을 가집니다:

- **소형 모델(3B)**이면서도
- **장문 컨텍스트(최대 128k)** 지원
- **다국어 지원**
- **추론 능력(reasoning)** 내장

뿐만 아니라, 다음을 **모두 공개**합니다:

- **모델 체크포인트**
- **전체 학습 레시피** (Pretraining, Mid-training, Post-training, Synthetic Data 생성)
- **데이터셋** (곧 공개 예정)

우리는 이 모델과 레시피가 **연구 커뮤니티에 실질적인 도움이 되기를 바라며**, 다른 연구자들이 이를 **개선하고 확장하는 기반이 되기를 희망합니다.**

### ? 리소스

- **양자화된 체크포인트 포함 모델 컬렉션**: (링크 미제공)
- **SmolLM GitHub (학습 구성 및 평가 코드)**:<https://github.com/huggingface/smollm>
- **HuggingFace 조직 페이지**:<https://huggingface.co/HuggingFaceTB>
