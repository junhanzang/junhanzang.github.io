---
title: "LLMs can see and hear without any training"
date: 2025-04-28 21:42:46
categories:
  - 인공지능
tags:
  - llms can see and hear without any training
---

<https://arxiv.org/abs/2501.18096>

[LLMs can see and hear without any training](https://arxiv.org/abs/2501.18096)

**초록**  
우리는 MILS(Multimodal Iterative LLM Solver)를 소개합니다. MILS는 놀라울 정도로 간단하면서도 학습이 필요 없는 접근 방식으로, 여러분이 선호하는 대형 언어 모델(LLM)에 멀티모달 기능을 부여할 수 있게 합니다. LLM의 본질적인 다단계 추론(multi-step reasoning) 능력을 활용하여, MILS는 LLM이 후보 출력을 생성하도록 유도하고, 각 후보에 점수를 매긴 뒤 이를 반복적으로 피드백하여 최종적으로 과제에 대한 해답을 생성합니다. 이 방법을 통해, 일반적으로 과제별 데이터에 특화된 모델 학습이 필요한 다양한 응용을 수행할 수 있습니다.  
특히, 우리는 emergent zero-shot 이미지, 비디오, 오디오 캡셔닝(emergent zero-shot image, video and audio captioning) 분야에서 새로운 최첨단(state-of-the-art) 성과를 달성했습니다. MILS는 미디어 생성(media generation)에도 자연스럽게 적용되어, 텍스트-투-이미지(text-to-image) 생성 품질을 향상시키기 위한 프롬프트 재작성(prompt rewrites)을 찾아내고, 스타일 전환(style transfer)을 위한 프롬프트 수정까지 수행할 수 있습니다!  
마지막으로, MILS는 그래디언트 없이 최적화하는 방식(gradient-free optimization approach)이기 때문에, 멀티모달 임베딩(multimodal embeddings)을 텍스트로 복원할 수도 있어, 크로스모달 산술(cross-modal arithmetic)과 같은 응용이 가능합니다.

![](/assets/images/posts/549/img.png)

**Figure 1 설명**  
우리가 제안하는 방법인 MILS는 이미지, 비디오, 오디오 캡셔닝부터 텍스트-투-이미지 생성 품질 향상, 스타일 전환과 같은 이미지 편집, 다양한 모달리티를 텍스트로 변환하여 수행하는 산술 연산에 이르기까지 다양한 응용을 가능하게 합니다. 이러한 모든 작업을, 과제별 학습이나 데이터 수집 없이, 순수하게 테스트 시간 최적화(test-time optimization)만으로 수행합니다!

**1. 서론**  
대형 언어 모델(LLMs)의 테스트 시점 추론 능력(test-time reasoning ability)은 어려운 과제를 해결하는 강력한 도구로 떠오르고 있습니다. 최근 OpenAI는 O1(OpenAI)을 소개했는데, 이는 테스트 시점 연산(test-time compute)을 활용하여 점진적으로 더 나은 결과를 얻기 위해 강화 학습(reinforcement learning)으로 훈련된 모델입니다. 특히 복잡한 수학 및 코딩 과제에서 두드러진 성과를 보였습니다. 추가 학습 없이도, LLM들은 체인-오브-생각(Chain-of-Thought, CoT) 추론 방식을 통해 테스트 시점 연산을 활용하여 인상적인 성능 향상을 보여주었으며, 사용자 질문에 답하기 위해 실행 계획(execution plan)을 전개하는 형태로 작동합니다(Wei et al., 2022; Kojima et al., 2022).

본 연구에서는, LLM이 지닌 이러한 본질적인 추론 능력을 활용하여, 학습 없이 멀티모달 이해(multimodal understanding) 및 생성(generation) 과제를 해결하는 방법을 제안합니다! 우리의 접근법인 **MILS(Multimodal Iterative LLM Solver)**는, LLM을 “**생성기(GENERATOR)**”로 사용하여 주어진 과제에 대한 후보 솔루션을 생성하고, 기성(multimodal) 모델을 “**채점기(SCORER)**”로 사용하여 각 후보 솔루션의 품질을 평가합니다.  
SCORER의 출력 결과는 GENERATOR로 다시 전달되어 피드백을 제공하고, 다음 단계에서 과제를 해결할 가능성이 더 높은 새로운 후보들을 생성하도록 돕습니다. 이 반복(iterative) 과정을 수렴할 때까지, 또는 정해진 횟수만큼 수행한 후, 최종적으로 과제에 대한 출력을 생성하게 됩니다.  
우리는 이 단순한 방법이 놀랍도록 강력하고 범용적이며, 다양한 과제와 모달리티에 걸쳐 잘 작동한다는 사실을 발견했습니다. 다양한 조합의 GENERATOR와 SCORER를 사용함으로써, MILS는 멀티모달 캡셔닝(captioning), 생성(generation), 편집(editing), 멀티모달 산술(multimodal arithmetic) 등 다양한 과제를 해결할 수 있습니다.

대부분의 기존 연구들은 이러한 과제들을 위해 해당 과제에 맞게 선별된 데이터로 훈련된 특수한 모델들을 사용합니다. 예를 들어, 제로샷 이미지 캡셔닝(zero-shot image captioning) 모델들도 여전히 이미지-캡션 쌍 데이터에 대해 학습되는 경우가 많습니다. 반면, **MILS**는 이러한 별도의 학습이 전혀 필요 없으며, 자연적으로(emergent) 제로샷 능력을 발휘합니다.  
예를 들어 이미지 캡셔닝의 경우, MILS는 표준 Llama (Dubey et al., 2024) LLM을 **GENERATOR**로 사용하고, CLIP (Radford et al., 2021)을 **SCORER**로 사용합니다. 주목할 점은, CLIP이 이미지-텍스트 데이터로 학습되긴 했지만, 일반적인 캡셔닝 모델들이 사용하는 정제된 이미지-캡션 데이터로 학습된 것은 아니라는 것입니다. 대부분의 비전-언어 모델(vision-language models)은 CLIP을 초기화 용도로만 사용하며, 이후에는 캡셔닝 데이터에 대해 추가 학습(post-training)을 필요로 합니다.  
따라서, 기존 모델들은 테스트 시점에서 새로운 데이터 분포에 대해 제로샷 일반화(zero-shot generalization)를 보일 수 있지만, **MILS는 새로운 과제인 "캡셔닝" 자체에 대해 emergent한 제로샷 일반화**를 보여줍니다.

또한, 캡셔닝 데이터 없이 캡션을 생성하는 접근 방식(Salewski et al., 2023; Tewel et al., 2022; Shaharabany et al., 2023; Zeng et al., 2024)도 일부 존재하지만, 이들은 특정 모달리티와 과제에만 한정되어 있습니다. 대부분 이러한 방법들은 CLIP의 그래디언트(gradient)를 활용하여 다음 토큰 예측을 유도하는데, 이는 캡셔닝 작업에만 국한되는 한계가 있습니다.  
반면, MILS는 **GENERATOR**와 **SCORER** 모듈만 교체함으로써 새로운 과제나 모달리티에도 자연스럽게 일반화(generalization)할 수 있습니다. 예를 들어, 단순히 LLM과 텍스트-투-이미지(Text-to-Image, T2I) 모델을 연결하여 만든 GENERATOR는, LLM을 "프롬프트 리라이팅(prompt rewriting)" 도구로 활용하여 기존 최첨단 T2I 모델의 성능을 향상시킬 수 있습니다. 이는 기존 접근법에서는 제공되지 않던 능력입니다.

본 연구에서는 MILS의 적용 가능성을 시각적 및 비시각적 세 가지 모달리티(이미지, 비디오, 오디오)와 세 가지 과제(캡셔닝, 생성, 편집)에서 보여줍니다. 추가로, MILS는 그래디언트가 필요 없는(gradient-free) 접근법이기 때문에, 멀티모달 임베딩(multimodal embeddings)을 이산 텍스트(discrete text)로 복원하는 데 사용할 수 있음을 입증합니다.  
이는 기존 연구(Kazemi et al., 2024)에서 그래디언트 기반 방법을 사용해 임베딩을 연속적 공간(continuous spaces, 예: 이미지)으로 복원했던 것과 대조됩니다. MILS의 이러한 능력은 멀티모달 샘플을 텍스트로 복원하고, 이를 결합하여 다시 생성하는 방식을 통해, **멀티모달 산술(multimodal arithmetic)** 과 같은 새로운 응용을 가능하게 합니다.  
우리는 Figure 1에서 이러한 기능들의 일부를 시각화하여 제시합니다.

**2. 관련 연구 (Related Work)**  
멀티모달 임베딩 공간(multimodal embedding spaces)은 일반적으로 인터넷에서 수집한 대규모 멀티모달 페어 데이터(주로 이미지와 텍스트)를 통해 학습됩니다. 각 모달리티에 대해 인코더(encoders)를 학습시키는데, 이때 쌍(pairwise) 간 유사도(similarity)를 최대화하는 목표(Objective)를 사용합니다(Radford et al., 2021; Ilharco et al., 2021; Zhai et al., 2023; Li et al., 2023a).  
이러한 모델들은 텍스트와 추가 모달리티를 짝지은 데이터(Wang et al., 2023; Guzhov et al., 2022)나, 임베딩 공간 내 다른 모달리티와 연결된 데이터(Girdhar et al., 2023; Gong et al., 2023)를 수집하여 추가적인 모달리티로 확장될 수 있습니다.  
이러한 임베딩들은 제로샷 인식(zero-shot recognition)(Radford et al., 2021; Girdhar et al., 2023), 오픈월드 객체 탐지(open-world object detection)(Zhou et al., 2022), 이미지 생성(image generation)(Ramesh et al., 2022) 등 다양한 응용을 가능하게 했습니다.  
우리는 이러한 임베딩을 활용하여 모달리티 간 유사도 점수(similarity score)를 계산하고, 이를 최적화(optimization)에 활용하여 원래 시각 및 청각 능력이 없는(즉, 멀티모달 입력을 처리할 수 없는) LLM에 멀티모달 기능을 부여합니다.

생성 모델(Generative models)은 최근 제로샷(zero-shot)으로 새로운 과제에 일반화할 수 있는 능력 덕분에 인기를 얻고 있습니다.  
LLMs(Dubey et al., 2024; Jiang et al., 2023; Team et al., 2024)은 텍스트와 같은 이산(discrete) 입력을 처리하기 위한 대표적인 모델로 자리잡았습니다.  
이들은 방대한 데이터 코퍼스(corpora)로 대규모 학습을 수행한 후, 인간 피드백을 포함한 고품질 데이터로 지침 튜닝(instruction tuning)을 거쳐 다양한 과제에서 강력한 성능을 발휘합니다.  
또한 체인 오브 싱킹(chain-of-thought prompting)(Wei et al., 2022; Kojima et al., 2022; Menon et al., 2024)과 최근 LLM의 추론 능력을 강화하는 훈련(OpenAI)을 통해, 복잡한 수학 및 코딩 과제에서도 더욱 뛰어난 성능을 보이고 있습니다.  
그러나 지침 튜닝은 목표 과제와 모달리티에 맞추어 LLM을 학습 또는 미세조정(finetuning)하는 과정을 필요로 합니다.  
반면 **MILS**는 학습 없이 **추론 시간(inference-time)** 동안 최적화를 수행하는 방식입니다.

최근 연구에서는 LLM의 추론 능력을 반복적으로(iteratively) 활용하여 최적화(optimization)(Yang et al., 2023) 및 생성(generation)(Mañas et al., 2024) 과제를 해결하려는 시도도 있었습니다. 그러나 이러한 방법들은 시각적 과제(visual tasks)에는 평가되지 않았거나(Yang et al., 2023), emergent한 제로샷 능력을 보여주지는 못했습니다(Mañas et al., 2024).

또 다른 범주의 생성 모델로는 연속적(continuous) 도메인, 특히 이미지(Rombach et al., 2022; Betker et al., 2023; Dai et al., 2023; Saharia et al., 2022)나 비디오(Polyak et al., 2024; Girdhar et al., 2024; Ho et al., 2022; Blattmann et al., 2023)에서 주로 사용되는 디퓨전(diffusion)(Nichol & Dhariwal, 2021) 또는 플로우 매칭(flow-matching)(Lipman et al., 2022) 기반 생성 모델들이 있습니다.  
이러한 모델들은 미디어 생성(media generation) 능력을 극적으로 향상시켰으며, 최근에는 LLM을 활용해 학습 데이터 캡셔닝 품질을 높이거나, 추론 시점 프롬프트 재작성(prompt rewrites)에도 사용되고 있습니다(Betker et al., 2023; Polyak et al., 2024).

제로샷 멀티모달 이해(zero-shot multimodal understanding)는 두 가지 형태로 연구되고 있습니다. 하나는 **데이터 분포 간 제로샷**(zero-shot across data distributions), 다른 하나는 **emergent 제로샷**(emergent zero-shot)(Girdhar et al., 2023)입니다.  
후자는 모델이 단순히 새로운 데이터에만 일반화하는 것이 아니라, **완전히 새로운 과제**에도 일반화하는 경우를 의미합니다.  
멀티모달 확장 버전의 대표적인 LLM들(Dubey et al., 2024; Agrawal et al., 2024; Li et al., 2023b)은 대부분 전자의 경우에 해당합니다. 이들은 보통 테스트 시점에 주어지는 데이터 유형에 대해 미리 학습되거나 튜닝되어 있습니다.  
본 연구의 초점은 후자에 있으며, 우리는 MILS가 테스트 시점에 **완전히 새로운 과제**에 대해 일반화할 수 있음을 보여줍니다.

이전 연구들(Tewel et al., 2022; Zeng et al., 2023; 2024; Salewski et al., 2023; Shaharabany et al., 2023)도 이러한 설정을 시도한 바 있으나, 이는 특정 모달리티에 대해 특수화된 기법을 사용하여 제한적으로 수행되었습니다.  
반면, **MILS**는 다양한 모달리티에 대해 이해(understanding) 및 생성(generation) 과제를 모두 자연스럽게 일반화할 수 있습니다.

**3. MILS**  
이제 **MILS**를 사용하여 멀티모달 과제를 해결하는 우리의 간단한 접근법을 설명합니다.  
MILS는 학습이 필요 없는(training-free) 방법이기 때문에, 테스트 샘플(test sample)만 입력으로 받습니다. MILS는 두 가지 핵심 모듈에 의존하는데, 이를 각각 **GENERATOR**와 **SCORER**라고 부릅니다.  
이름에서 알 수 있듯이, **GENERATOR**는 주어진 과제에 대한 후보 솔루션(candidate solutions)을 생성하고, **SCORER**는 이 후보들을 평가하여 다시 GENERATOR로 피드백을 보냅니다. 이 과정을 통해 더 나은 후보 세트를 생성하게 됩니다.  
특정 과제에서는, 초기 후보 세트에 대한 점수를 사용하여 부트스트랩(bootstrapping)할 수도 있습니다.  
이 최적화 과정은 수렴(convergence)하거나, 정해진 횟수만큼 반복(iterations)한 뒤 종료되며, 최종적으로 과제에 대한 해답을 생성합니다.  
Figure 2는 이 전체 과정을 도식화한 것입니다.

![](/assets/images/posts/549/img_1.png)

**Figure 2 설명**  
MILS는 **GENERATOR**와 **SCORER**라는 두 가지 핵심 모듈을 활용하여 멀티모달 과제를 해결합니다.  
**GENERATOR**는 여러 개의 텍스트 후보(candidates)를 생성하는데, 예를 들어 이미지 캡셔닝(image captioning)에서는 캡션(captions), 텍스트-투-이미지(Text-to-Image, T2I) 과제에서는 프롬프트(prompts)를 생성합니다.  
각 후보들은 **SCORER**에 의해 점수화(scoring)되고, 그 결과가 피드백으로 다시 GENERATOR에 전달되어 다음 후보군을 생성하는 데 사용됩니다.  
이 과정을 반복하여 최종적으로 입력 테스트 샘플(test sample)에 대한 최종 출력을 만들어냅니다.

**GENERATOR**  
**GENERATOR**의 목표는 주어진 과제를 해결할 수 있는 후보 출력(Candidates, C)을 생성하는 것입니다.  
입력으로는 과제 설명을 담은 텍스트(T)와, 이전 최적화 단계에서 SCORER로부터 받은 점수(S, 있을 경우)를 함께 받습니다.  
GENERATOR는 이 입력 신호를 활용해 다음 후보 세트를 생성합니다.  
일반적으로 **GENERATOR**는 텍스트를 입력으로 받아 이를 기반으로 추론할 수 있는 LLM을 이용해 모델링됩니다. 그러나 출력은 텍스트에 한정되지 않습니다.  
생성된 후보들은 다른 모달리티를 생성하는 모델을 호출하는 프롬프트로 사용될 수 있습니다. 예를 들어, 텍스트-투-이미지(Text-to-Image, T2I) 모델인 Emu(Dai et al., 2023)를 이용해 이미지를 생성할 수도 있습니다.  
또한, 일부 GENERATOR는 테스트 샘플(test sample) 자체를 입력으로 사용할 수도 있습니다. 예를 들어 이미지 편집(image editing)이나 스타일 전환(stylization) 과제에서는 이러한 방식이 사용될 수 있습니다.

**SCORER**  
**SCORER**의 목표는, GENERATOR가 생성한 후보(Candidates, C)에 대해 스칼라 점수(scalar score) S ∈ ℝ를 계산하는 것입니다.  
SCORER는 입력으로 테스트 샘플(test sample)과 후보군 C를 받아 이 둘을 비교합니다.  
SCORER는 다양한 방식으로 구현할 수 있습니다.  
예를 들어, 두 이미지의 텍스처(texture)를 비교하는 저수준(low-level) 이미지 처리 함수가 될 수도 있고, CLIP(Radford et al., 2021; Ilharco et al., 2021)과 같은 학습된 머신러닝 모델이 될 수도 있습니다.  
SCORER는 모든 후보를 점수에 따라 정렬한 후, 상위 K개(top-K candidates)와 해당 점수를 반환합니다.  
또한, GENERATOR의 입력 용량(컨텍스트 길이, context length)에 따라 전체 점수 목록을 반환할 수도 있고, ε-탐욕(ϵ-greedy) 전략을 사용해 점수가 낮은 후보 일부를 포함할 수도 있습니다.  
초기 실험에서는, 단순한 탐욕적(top-K) 선택이 가장 좋은 성능을 보였기 때문에, 본 연구에서는 이 방식을 사용합니다.  
SCORER의 출력은 텍스트 형태(T)로 포맷되어 다시 GENERATOR로 전달됩니다.

**최적화 과정(Optimization process)**  
MILS는 SCORER의 비용 함수(cost function) 하에서 최적의 생성물(candidate generation, C)을 탐색합니다.  
이 최적화 과정은 N 스텝 동안 수행되거나, 수렴(convergence)할 때까지 반복됩니다.  
수렴은 후보군 C가 연속된 스텝 사이에서 유사해지는 것으로 정의할 수 있습니다.  
과제(task)에 따라, 최적화는 초기 후보 세트(initial candidate set)를 부트스트랩(bootstrapping)하여 시작할 수도 있습니다.  
예를 들어 이미지 캡셔닝의 경우, GENERATOR가 생성한 다양한 가능한 캡션 후보군을 초기 세트로 사용할 수 있습니다.  
반면, T2I(Text-to-Image)와 같은 과제에서는 이러한 초기 세트가 필요 없습니다.

---

정리하면:

- **MILS에서 사용하는 LLM 자체는 멀티모달 모델이 아닙니다.**  
  → 이 LLM은 **텍스트 입력만** 처리할 수 있고, **텍스트 출력만** 생성합니다. (예: LLaMA, GPT 계열처럼 일반 텍스트 LLM)
- 그런데 MILS는 **멀티모달 과제(이미지 캡셔닝, 이미지 생성, 오디오 설명 등)** 를 풀기 위해 필요할 때 **외부 멀티모달 도구**를 함께 사용합니다.
- 예를 들어:
  - 이미지 생성(Text-to-Image, T2I)을 하고 싶으면,  
    → LLM이 프롬프트(prompt)를 텍스트로 작성하고,  
    → 그 프롬프트를 **외부 T2I 모델(예: Emu, Stable Diffusion 같은 모델)** 에 넘겨서 이미지를 생성합니다.
  - 반대로, 이미지를 평가(Scoring)할 때도  
    → 생성한 이미지와 목표 이미지/조건을 비교하기 위해 **CLIP** 같은 멀티모달 모델을 사용합니다.

**요약**  
MILS 안에서는:

- **LLM** → 텍스트 생성만 담당 (후보 생성, 프롬프트 수정 등)
- **멀티모달 모델 (T2I, CLIP 등)** → 이미지 생성이나 이미지-텍스트 유사도 평가를 담당  
  → 둘을 **반복적으로 연결**하면서, 멀티모달 과제를 "학습 없이" 해결하는 구조입니다.

![](/assets/images/posts/549/img_2.png)

---
