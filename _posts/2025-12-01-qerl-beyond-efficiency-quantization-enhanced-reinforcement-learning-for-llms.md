---
title: "QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs"
date: 2025-12-01 23:45:26
categories:
  - 인공지능
---

<https://arxiv.org/abs/2510.11696>

[QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs

We propose QeRL, a Quantization-enhanced Reinforcement Learning framework for large language models (LLMs). While RL is essential for LLMs' reasoning capabilities, it is resource-intensive, requiring substantial GPU memory and long rollout durations. QeRL

arxiv.org](https://arxiv.org/abs/2510.11696)

**초록(Abstract)**  
본 논문에서는 대규모 언어 모델(LLM)을 위한 **QeRL(Quantization-enhanced Reinforcement Learning)** 프레임워크를 제안한다. RL은 LLM의 추론 능력을 향상시키는 데 필수적이지만, 많은 GPU 메모리와 긴 rollout 시간이 필요해 자원 소모가 크다. QeRL은 **NVFP4 양자화**와 **Low-Rank Adaptation(LoRA)**을 결합해 RL의 rollout 단계를 가속화하고 메모리 사용량을 줄인다. 효율성뿐만 아니라, 본 연구는 양자화 노이즈가 정책 엔트로피(policy entropy)를 증가시켜 탐색 성능을 향상시키고 RL 과정에서 더 나은 전략을 찾도록 돕는다는 사실을 보여준다. 탐색 효율을 더욱 최적화하기 위해, QeRL은 학습 중 노이즈를 동적으로 조절하는 **Adaptive Quantization Noise(AQN)** 메커니즘도 도입한다.

실험 결과, QeRL은 rollout 단계에서 **1.5배 이상의 속도 향상**을 달성했다. 또한 본 프레임워크는 **단일 H100 80GB GPU에서 32B LLM의 RL 학습을 가능하게 한 최초의 기법**이며, RL 전체 학습 과정에서도 전반적인 속도 향상을 제공한다. 더불어 QeRL은 16-bit LoRA와 QLoRA보다 더 빠른 보상 상승 속도와 더 높은 최종 정확도를 보였으며, 7B 모델 기준 GSM8K(90.8%)와 MATH 500(77.4%) 등 수학 벤치마크에서 **full-parameter fine-tuning과 동등한 성능**을 달성했다. 이러한 결과는 QeRL이 LLM RL 학습을 위한 효율적이고 효과적인 프레임워크임을 입증한다.

![](/assets/images/posts/606/img.png)

**그림 1:** Qwen2.5-7B-Instruct에서 QeRL의 rollout 속도 및 정확도. QeRL은 RL rollout과 end-to-end 학습(batch=8)에서 더 빠른 속도를 달성하며, vanilla LoRA와 QLoRA보다 우수한 성능을 보이고, 수학 벤치마크에 대해 full-parameter RL과 유사한 정확도를 제공한다.

**1. 서론(Introduction)**  
다단계 추론 능력은 대규모 언어 모델(LLM)이 이론적 문제 해결부터 실제 의사결정에 이르기까지 복잡한 작업을 처리하는 데 핵심적이다(Sui et al., 2025; Xu et al., 2025; Chu et al., 2025; Yang et al., 2021). 지도 미세조정(Supervised Fine-Tuning, SFT)은 명시적 추론 과정을 모델이 모방하도록 학습시켜 추론 성능을 향상하는 일반적인 방법이다(Huang et al., 2024d; Min et al., 2024). 그러나 이러한 접근은 실제 추론을 유도하기보다는 단순 모방을 강화하는 위험이 있다. 반면, 강화학습(RL)은 검증 가능한 보상 신호를 기반으로 적응적 학습을 수행하게 하여, 모델이 다양한 추론 경로를 탐색하고 더 견고한 해법을 스스로 찾아내도록 돕는다(Lambert et al., 2024; DeepSeek-AI, 2025; Chen et al., 2025a).

LLM의 추론 능력 향상을 위해 RL은 효과적이지만 자원 소모가 매우 크다. 예를 들어 GRPO(Shao et al., 2024)에서는 정책 모델과 기준(reference) 모델 등 여러 모델을 동시에 실행해야 하므로 GPU 메모리 사용량이 크게 증가한다. 여기에 더해 추론 특화 LLM의 거대한 규모(DeepSeek-AI, 2025)는 메모리 요구량을 더욱 악화시킨다. 또한 RL 학습은 rollout, 보상 계산, logit 평가, 그래디언트 업데이트 등 여러 단계를 거치므로 학습 속도가 상당히 느리다. 특히 rollout 단계는 긴 시퀀스를 반복적으로 샘플링하고 처리해야 하므로 매우 비용이 크다(Yu et al., 2025). 또한 RL 특유의 **샘플 비효율성(sample inefficiency)**(Hassani et al., 2024) 역시 전체 비용을 증가시키는 요인이다.

![](/assets/images/posts/606/img_1.png)

**그림 2:** QeRL의 개념도.  
(a) LoRA 기반 RL: 학습 가능한 파라미터 수는 감소시키지만, rollout 병목은 해결하지 못한다.  
(b) QLoRA 기반 RL: NF4 양자화를 LoRA와 결합하지만 NF4는 LoRA보다 느리다.  
(c) QeRL: NVFP4 양자화와 LoRA를 결합하여 메모리를 절감하고 RL 속도를 향상시키며, 적응형 양자화 노이즈(AQN)를 통해 full-parameter 미세조정과 동등한 성능을 달성한다. AQN은 지수 스케줄러를 통해 양자화 노이즈를 동적으로 조정하며 탐색 성능을 향상시킨다.

LLM에서 RL 효율을 높이는 문제에는 여러 난제가 존재한다. 한 가지 접근법으로 Tina(Wang et al., 2025)가 있는데, 이는 Low-Rank Adaptation(LoRA) (Hu et al., 2022)와 같은 **파라미터 효율적 미세조정(PEFT)** 기법을 활용해 학습해야 할 파라미터 수를 줄인다. 그러나 SFT에서의 LoRA와 동일하게(Chen et al., 2024b), 이러한 방법은 **rollout 속도가 느리다는 근본적인 병목을 해결하지 못한다**. 또 다른 접근으로 FlashRL(Liu et al., 2025a)은 **양자화된 rollout 모델**을 사용해 연산 비용을 줄인다. 하지만 rollout 모델과 logit 모델의 정밀도가 일치하지 않으면(예: 8비트 vs 16비트), 불일치를 보정하기 위해 **중요도 샘플링(importance sampling)**이 필요하며, 이 과정에서 8비트 모델과 16비트 모델을 동시에 메모리에 유지해야 하므로 메모리 사용량이 증가한다. 이러한 한계를 극복하기 위해, 우리는 **중복 모델 없이 더 낮은 비트의 양자화**에 초점을 맞춘다.

추가적으로, RL에서 QLoRA(Dettmers et al., 2023a)를 사용할 경우 rollout 속도가 **1.5~2배 느려지며**, 효율성이 크게 저하된다. 이는 QLoRA가 사용하는 **NormalFloat 4-bit(NF4)**의 구조적 특성 때문인데, NF4는 행렬 곱셈 전에 lookup table을 통해 부동소수점 값으로 압축을 풀어야 하므로 계산 병목이 발생한다.

NF4의 이러한 한계를 해결하기 위한 자연스러운 대안은 더 고성능의 양자화를 사용하는 것이다. 하지만 일반적인 양자화 방식은 **고정적이고 결정적인 노이즈**를 도입하며, 이는 RL 학습 후반 단계에서 도움이 되지 않는다. 이를 피하기 위해, 우리는 양자화 노이즈가 **정밀하게 제어된다면 오히려 RL에 이점을 줄 수 있음**을 발견했다(그림 3). 양자화 노이즈는 정책 엔트로피(policy entropy)를 증가시키며, 이는 RL에서 파라미터 노이즈가 탐색을 촉진하는 효과(Plappert et al., 2017; Pang and Jiang, 2021)와 유사하게 작용하여 더 나은 전략을 찾도록 돕는다(Cui et al., 2025). 우리의 실험 결과, 적절히 설계된 노이즈 전략은 양자화된 LLM이 이러한 특성을 활용해 **메모리 사용량을 줄이면서도 보상 곡선을 개선**할 수 있음을 보여준다. 이는 SFT 환경에서의 기존 연구(Dettmers et al., 2023a; Guo et al., 2023)와는 정반대의 결과로, RL에서는 **제어 가능한 양자화 노이즈가 탐색을 강화**하여 16-bit LoRA보다 효율성과 성능 모두에서 개선을 가능하게 한다는 점을 시사한다.

우리는 LLM의 추론 작업을 위한 양자화 기반 RL 프레임워크인 **QeRL**을 제안한다. 그림 2에서 보이듯, QeRL은 LLM 가중치에 **NVFP4 양자화**를 적용하며, rollout과 prefilling 단계 모두에 **Marlin 기반(Frantar et al., 2024)** 방식을 통합한다. 이 설계는 정확도 손실 없이 rollout과 prefilling을 가속하며, LoRA 레이어를 통해 gradient 역전파가 가능하도록 구성된다. 또한 고정적 양자화 노이즈 문제를 해결하기 위해 **Adaptive Quantization Noise(AQN)**를 도입한다. AQN은 학습 중에 채널 단위의 랜덤 노이즈를 주입하며, 지수 스케줄러(exponential scheduler)를 이용해 탐색 노이즈 크기를 동적으로 조정한다. 추가로, 노이즈 벡터를 layer normalization에 병합하는 **noise-sharing 전략**을 도입해, 노이즈 주입에 필요한 추가 파라미터를 0으로 유지한다.

QeRL은 vanilla LoRA에 비해 rollout이 더 빠르고 보상 성장률도 높다. 예를 들어 그림 1에서 보이듯, QeRL은 QLoRA와 vanilla LoRA보다 rollout과 prefilling 속도 모두에서 빠르며, Qwen2.5-7B-Instruct 모델 기준으로 **GSM8K 점수 90.8**을 기록해 **16-bit LoRA와 QLoRA를 넘어서는 성능**을 달성했다. MATH 500에서도 full-parameter fine-tuning과 동등한 정확도를 보였다. 또한 QeRL은 QLoRA 대비 **약 1.8배 빠른 end-to-end 학습 속도**를 달성했으며, 단일 H100 80GB GPU에서 GRPO를 사용한 **32B 모델 RL 학습**까지 가능함을 입증했다.

![](/assets/images/posts/606/img_2.png)

**그림 3:** RL 탐색에서 양자화의 역할 확장.  
양자화 노이즈는 초기 정책 엔트로피를 증가시켜 탐색을 촉진하며, 이로 인해 보상 상승 속도가 빨라진다.

**2. 예비 지식(Preliminary)**  
**모델 양자화(Model Quantization)**

![](/assets/images/posts/606/img_3.png)

![](/assets/images/posts/606/img_4.png)

**3. 방법(Method)**  
우리의 실험 결과, **양자화된 LLM은 RL에서 탐색을 크게 향상**시킬 수 있음을 확인했다. 양자화된 모델에 파라미터 효율적 미세조정(PEFT)을 적용하면 학습 자원 소비가 줄어들 뿐만 아니라, 보상 성장률과 평가 점수 측면에서 vanilla LoRA보다 우수한 성능을 보였다(그림 2). 이는 SFT 환경에서 양자화가 학습 효율을 저하시킨다는 기존 관점(Dettmers et al., 2023a; Guo et al., 2023)을 뒤흔드는 결과다. 특히, 우리는 양자화 오차가 네트워크에서 랜덤 노이즈와 유사하게 작용한다는 점을 관찰했다(Plappert et al., 2017; Eberhard et al., 2023; Osband et al., 2016). 이러한 양자화 노이즈는 정책 엔트로피를 증가시켜 RL에서 가능한 행동 또는 토큰 공간을 더 넓게 탐색하도록 유도하며(그림 3), 결과적으로 더 나은 탐색 전략을 만들어낸다.

### **3.1 QeRL의 학습 프레임워크(Training Framework of QeRL)**

QeRL은 GRPO(Shao et al., 2024), DAPO(Yu et al., 2025)와 같은 **최신 LLM 정책 최적화 알고리즘(policy optimization algorithms)**을 기반으로 한다.

![](/assets/images/posts/606/img_5.png)

**Dynamic Sampling Policy Optimization(DAPO)**(Yu et al., 2025)은  
• 더 큰 상한 클리핑 값(clipping upper-bound)이 **엔트로피 붕괴(entropy collapse)**를 방지하는 데 도움이 된다는 점을 제안한다.  
• 또한 token-level 정책 그래디언트 손실을 활용하는 개선을 포함한다.

DAPO에서는 식 (3)의 KL 패널티를 제거하여 RL 탐색 상한을 없애고, rollout 과정에서 **더 다양한 토큰을 시도할 수 있도록 탐색 범위**를 넓힌다.

### **3.2 양자화는 탐색을 촉진한다(Quantization Encourages Exploration)**

양자화가 RL을 어떻게 향상시키는지 이해하기 위해, 우리는 양자화가 모델의 샘플링 행동에 미치는 영향을 분석했다. 우리의 핵심 발견은 다음과 같다.

> **양자화가 도입하는 노이즈는 암묵적인 탐색 메커니즘으로 작용한다.**

이는 파라미터 공간 또는 행동 공간에 명시적으로 노이즈를 주입하는 기존 기법들(Plappert et al., 2017; Eberhard et al., 2023; Fortunato et al., 2018; Liu et al., 2025b)과 유사한 역할을 한다.

![](/assets/images/posts/606/img_6.png)

**그림 4:** 학습 보상 성능.  
위쪽 그림은 DAPO에서의 학습 보상 곡선을, 아래쪽 그림은 GRPO에서의 보상 곡선을 나타낸다. 초기 학습 단계에서는 MXFP4가 더 높은 보상을 보이지만, 최종적으로는 NVFP4가 더 높은 보상으로 수렴한다. LoRA의 랭크(rank)는 32로 설정하였다.

---



![](/assets/images/posts/606/img_7.png)
---

### **양자화는 샘플링 엔트로피를 향상시킨다**

우리는 GSM8K(Cobbe et al., 2021)에서 FP4 기반의 세 가지 양자화 포맷(NVFP4, MXFP4, NF4)의 동작을 분석하였다.

![](/assets/images/posts/606/img_8.png)

**그림 5:** RL 엔트로피 비교.

Qwen2.5-7B-Instruct(Team, 2024)에 대한 실험 결과, 매우 흥미로운 사실이 나타났다.  
**PEFT 기반의 RL을 적용할 때, 4비트로 양자화된 모델이 16비트 모델보다 일관되게 더 우수한 성능을 보인다는 점이다.**

이 우위는 두 가지 핵심 지표에서 두드러진다:

1. 학습 중 보상 수렴 속도가 훨씬 빠르며
2. 조정된 평가 점수(adjusted evaluation score)가 더 높다.

그림 4에서 볼 수 있듯이, 4비트 모델들의 보상 곡선은 16비트 모델 대비 훨씬 가파른 상승 추세를 보이며, DAPO와 GRPO 모두에서 full-parameter fine-tuning과 유사한 수렴 패턴을 나타낸다. 또한 NVFP4와 MXFP4는 NF4보다 더 좋은 보상 성장을 보였다.

![](/assets/images/posts/606/img_9.png)

이 샘플링 엔트로피의 증가는 강화학습에서 탐색을 촉진하는 데 중요한 역할을 한다(Cheng et al., 2025; Eysenbach & Levine, 2021). 모델이 단일 “최적” 토큰에 과도하게 확신(overconfidence)하는 것을 억제하고, 더 넓은 범위의 합리적 후보 토큰에 의미 있는 확률을 부여하도록 만든다(그림 3). 다른 모델들의 엔트로피 수치는 부록 H에서 제공한다.

![](/assets/images/posts/606/img_10.png)

### **양자화 노이즈의 한계**

양자화 오차의 주요 한계는 **결정적(deterministic)**이라는 점이다. 이는 RL에서 요구되는 **동적 탐색–활용 균형(exploration-exploitation trade-off)**과 맞지 않는다.

전통적인 RL의 stochastic noise(Plappert et al., 2017; Osband et al., 2016)는:

- 매 단계마다 새롭게 샘플링되며
- 완전히 독립적으로 적용되지만

양자화 노이즈는 학습 내내 정적으로 유지된다.  
즉, 탐색이 더 필요하거나 덜 필요한 특정 학습 시점에 맞춰 조정될 수 있는 **적응성(adaptivity)**이 부족하다.

### **3.3 파라미터 공간에서의 적응형 양자화 노이즈(Adaptive Quantization Noise)**

정적 양자화 노이즈를 **동적으로 변화하는 탐색 메커니즘**으로 전환하기 위해, 우리는 **적응형 양자화 노이즈(AQN, Adaptive Quantization Noise)** 기법을 도입한다.  
AQN의 핵심 개념은 다음과 같다:

> **본래 정적으로 존재하던 양자화 노이즈에 아주 작은 구조적 변형(structured modulation)을 주어, 이를 시간에 따라 변화하는 탐색 노이즈로 만드는 것.**

이를 위해 우리는 고성능 양자화 포맷인 **NVFP4**를 활용한다.

![](/assets/images/posts/606/img_11.png)

![](/assets/images/posts/606/img_12.png)

![](/assets/images/posts/606/img_13.png)

![](/assets/images/posts/606/img_14.png)

![](/assets/images/posts/606/img_15.png)

![](/assets/images/posts/606/img_16.png)

![](/assets/images/posts/606/img_17.png)

![](/assets/images/posts/606/img_18.png)

## **4 실험(Experiment)**

### **4.1 실험 설정(Experiment Settings)**

**RL 학습 설정**  
우리는 DAPO(Yu et al., 2025)와 GRPO(Shao et al., 2024)를 사용하여 두 가지 대표적인 수학적 추론 데이터셋에서 실험을 수행했다: **GSM8K**(Cobbe et al., 2021)와 **BigMath**(Albalak et al., 2025).  
GSM8K는 총 7,500개의 샘플로 구성되며 generation number는 8이고, BigMath는 122,000개의 샘플과 generation number 16을 갖는다. 두 데이터셋 모두 난이도 레벨 3~5에 해당하는 중·고난도 문제들로 구성된다.

- GSM8K: 3B 및 7B 모델 학습
- BigMath: 7B, 14B, 32B 모델 학습
  - 7B/14B 모델: 레벨 3~5 문제 사용
  - 32B 모델: 더 어려운 레벨 4~5 문제만 사용

![](/assets/images/posts/606/img_19.png)

속도 측정(speedup) 실험은 **단일 H100 GPU**에서 수행되었고, 최종 평가 모델은 대규모 데이터 효율성을 위해 **8× H100 GPU**로 학습되었다. QeRL의 상세 하이퍼파라미터 및 배포 구성은 Appendix E와 Appendix F에 제시한다.

**백본 모델(Backbone Models)**  
우리는 Qwen2.5(Team, 2024) 시리즈를 사용하며, 추가적인 수학 데이터로 미세조정되지 않은 기본 모델을 사용했다.  
weight-only quantization에는 **AWQ(Lin et al., 2024)**를 적용하여 MXFP4와 NVFP4 포맷을 생성했다.

- Calibration dataset:  
  256개의 시퀀스(각 2048 tokens),  
  OpenThoughts-114k(Guha et al., 2025)에서 샘플링

weight-only 양자화된 모델은 NVIDIA H100 GPU에서 **Marlin 커널(Frantar et al., 2024)**을 사용한 가속 추론이 가능하다.  
NF4 quantization은 Dettmers et al.(2023a)의 기본 설정을 따른다.

**평가 벤치마크 및 메트릭(Evaluation Benchmarks and Metrics)**  
평가는 다음과 같은 널리 사용되는 수학적 추론 벤치마크를 기준으로 수행했다:

- **GSM8K** (Cobbe et al., 2021)
- **MATH500** (Lightman et al., 2023)
- **AIME 2024/2025** (Li et al., 2024)
- **AMC 23** (Li et al., 2024)

추론 시 설정은 다음과 같다:

- Temperature = 0.6
- Completion length = 4096
- top-p sampling: p=0.95

각 데이터셋은 여러 번 테스트되며, 보고되는 성능은 주로 **Pass@1(1회 샘플 기준 정확도)**의 평균값이다.

![](/assets/images/posts/606/img_20.png)

![](/assets/images/posts/606/img_21.png)

### **표 1: GSM8K에서의 Qwen2.5 성능**

- GRPO 알고리즘을 사용하여 3B와 7B 모델을 GSM8K 데이터셋에서 학습하였다.
- “Full”은 **full-parameter training**을 의미한다.
- “W#”는 **가중치의 비트 폭(bit-width)** 및 데이터 포맷을 의미한다.
- + 및 –는 기본 BF16 모델 대비 얼마나 성능이 향상되거나 감소했는지를 나타낸다.

![](/assets/images/posts/606/img_22.png)

![](/assets/images/posts/606/img_23.png)

![](/assets/images/posts/606/img_24.png)

## **표 2: 네 가지 벤치마크에 대한 성능 비교**

- DAPO 알고리즘을 사용하여 Qwen2.5-7B/14B/32B Instruction 모델을 BigMath 데이터셋에서 학습했다.
- “Full”은 full-parameter training을 의미한다.

![](/assets/images/posts/606/img_25.png)

**그림 7:** 7B 및 14B 모델의 학습 보상 곡선.

![](/assets/images/posts/606/img_26.png)

**그림 8:** 3B 및 7B 모델에서 AQN의 ablation 실험 결과.

## **4.2 실험 결과(Experiment Results)**

### **추론 성능(Reasoning Performance)**

표 1에서 보이듯, 우리는 GRPO를 사용해 3B 및 7B 모델을 GSM8K에서 학습한 결과를 보고한다.  
양자화된 모델은 BF16 모델 대비 초기 성능 저하가 존재하지만, **PEFT와 RL을 결합하여 3B 모델을 학습한 경우**, NVFP4와 AQN을 함께 적용하면 **59.4 → 83.7**로 성능이 크게 향상되었으며, 이는 **16비트 LoRA(76.1)**를 초월하고 full-parameter 학습 대비 단 **0.7점 차이**만을 보인다.

7B 모델에서도 동일한 경향이 관찰되며, 우리 기법은 **16비트 LoRA 대비 1.7포인트 성능 향상**을 달성한다.  
또한 QLoRA와 비교할 때, 본 접근법은

- **3B 모델에서 +7.6포인트**,
- **7B 모델에서 +5.8포인트**  
  평균 정확도 향상을 보여준다.

표 2는 DAPO를 사용하여 BigMath 데이터셋에서 학습된 7B, 14B, 32B 모델의 결과를 제시한다.  
모든 데이터셋에서 QeRL은 **16비트 LoRA로 학습한 모델의 성능과 동등하거나 그 이상**을 안정적으로 달성한다.

특히 QeRL은

- **full-parameter training의 약 1%만 학습**하며,
- **vanilla LoRA 대비 GPU 메모리 사용량은 40%–50% 수준**에 불과하다.

7B 모델의 경우, 양자화 모델의 평균 점수 25.7에서 QeRL 적용 후 36.4까지 상승해, vanilla LoRA의 35.7을 넘어섰다.

14B 및 32B 모델에서도 동일한 경향이 관찰되며, QeRL은 모든 벤치마크에서 일관적으로 vanilla LoRA를 능가한다. 이 결과는 **양자화가 RL 학습을 강화한다는 본 논문의 주장**을 강하게 뒷받침한다.

특히 AMC 23 데이터셋에서는  
**14B QeRL 모델이 57.5점을 기록하여 full-parameter 학습(55.0)을 초과하는 성과를 달성했다.**

![](/assets/images/posts/606/img_27.png)

**그림 9:** 노이즈 스케줄러 비교.

![](/assets/images/posts/606/img_28.png)

**그림 10:** LoRA rank ablation 실험.

### **Reward Visualization**

Sec.3.2에서는 GRPO와 DAPO 환경에서 **양자화 LoRA(quantized LoRA)**, **vanilla LoRA**, **full-parameter training**의 정확도 기반 보상(reward)을 비교하였다.  
그림 8은 난도가 높은 BigMath 데이터셋에서 7B 및 14B 모델의 보상 곡선을 보여준다.

특히 **QeRL은 200 스텝 이내에 빠르게 보상이 증가**하는 반면,  
vanilla LoRA는 개선이 나타나기까지 **500 스텝 이상(Appendix H 참고)**이 필요하다.

이 결과는 양자화된 LLM이 내재적으로 포함하는 노이즈가 RL에서 탐색 성능을 향상시키고,  
더 빠른 보상 상승과 더 높은 보상 목표에 도달하도록 돕는다는 본 논문의 주장을 뒷받침한다.

### **Noise Decay Schedule**

그림 10은 3B 모델을 대상으로 네 가지 노이즈 감소 스케줄의 성능을 비교한다:

- Linear decay
- Exponential decay
- Cosine decay
- Logarithmic decay

초기 단계에서는 네 스케줄 간 차이가 거의 없지만,  
**exponential decay는 후반에 노이즈를 더 낮은 수준까지 안정적으로 감소시켜 더 일관된 성능 향상**을 보인다.  
각 스케줄링 곡선은 Appendix H에 제공한다.

표 3: 7B 및 14B 모델의 메모리 절감 및 속도 향상

![](/assets/images/posts/606/img_29.png)

**표 3 설명:**  
7B 및 14B 모델에 대해 **메모리 절감량과 GRPO 기준 end-to-end 학습 속도 향상**을 보고한 결과이다.  
각 입력 길이는 256 tokens이며, 최대 생성 길이는 2048 tokens이다.  
다른 모델에 대한 추가 실험 결과는 Appendix J에서 확인할 수 있다.

## **AQN Ablation**

학습 전체에 걸쳐 기본 양자화 노이즈만 사용하는 방식은 RL에서의 탐색을 제한한다. 이를 해결하기 위해 본 논문에서는 **AQN(Adaptive Quantization Noise)**을 도입한다. 그림 8에서 확인할 수 있듯이, 기본 양자화 노이즈로 시작한 뒤 학습 후반 단계에서 주기적으로 추가 노이즈를 주입하면 보상 곡선이 더욱 안정적으로 증가한다. 특히 **보상이 수렴 단계에 가까워질 때**, AQN은 효과적으로 탐색 공간을 확장하여 보상을 한 단계 더 끌어올린다.

## **LoRA Rank Ablation**

그림 10은 QeRL을 적용한 3B 모델에서 LoRA rank(16, 32, 64, 128)에 따른 보상 곡선을 비교한 결과이다.  
네 설정 모두 유사한 추세와 보상 성장을 보였으며, **rank 16이 조금 더 빠르게 수렴하면서 비용 효율적인 선택**임을 확인할 수 있다.

![](/assets/images/posts/606/img_30.png)

**그림 11:** 14B 및 32B 모델의 rollout 처리량(throughput). 설정은 표 7(batch=1)과 동일하다.

## **4.3 메모리 절감 및 속도 향상**

표 3은 NVIDIA H100-80GB GPU(NVIDIA, 2023) 단일 장비에서 수행한 실험을 기반으로, 양자화 모델의 크기와 end-to-end RL 학습 속도 향상을 비교한다.

7B와 14B 모델 모두에서:

- **QLoRA (NF4)** 와
- **QeRL (NVFP4 + Marlin kernel(Frantar et al., 2024))**

은 메모리 사용량을 크게 줄여, 모델 크기가 16비트 대비 **약 25%~30% 수준**으로 감소한다.

그러나 NF4의 생성 속도 제한(Egashira et al., 2024) 때문에, QLoRA는 배치 크기 전반에서 **0.7×~0.8×로 속도가 오히려 감소**한다.

반면 QeRL은 **1.2×~1.5×** 속도 향상을 달성하며, 특히 긴 reasoning sequence 생성에 강한 이점을 보인다.

이러한 효율성은 **rollout 길이가 길어질수록 더 중요**해지며, RL 환경에서 QeRL의 장점이 더욱 두드러진다.

또한 속도 측정은 **학습 초기 30 step 동안** 수행된 것으로, 이 시기에는 출력 토큰 길이가 상대적으로 짧다.  
학습이 진행되어 생성 길이가 늘어날수록 **QeRL의 속도 이점은 더 크게 확대**된다.

따라서 QeRL은 메모리 효율성과 학습 속도 모두에서 강점을 지니며,  
특히 **대규모 rollout이 필요한 RL 워크플로우**에서 매우 효과적인 방법이다.

그림 11에서는 LoRA rank에 따른 rollout 성능을 보여주며, 14B와 32B 모델 모두에서 QeRL이 **2× 이상의 속도 향상**을 달성함을 확인할 수 있다.  
다른 모델과 설정에 대한 추가 비교는 Appendix J에 수록되어 있다.

## **5 결론(Conclusion)**

본 논문에서는 LLM에서의 강화학습을 위해 **NVFP4 정밀도 양자화**와 **LoRA 미세조정을 결합한 QeRL**이라는 효율적인 학습 프레임워크를 제안했다.  
본 프레임워크는 **양자화가 RL에서 탐색을 강화할 수 있다**는 새로운 관찰에 기반하며, 이는 기존 SFT 연구에서 보고된 결과들과는 대조적이다.

양자화된 LLM은

- 기존의 16비트 LoRA 학습을 능가할 뿐 아니라
- full-parameter fine-tuning에 근접한 성능까지 달성하였다.

정적 양자화 노이즈의 한계를 해결하기 위해 **AQN 메커니즘**을 도입하여,  
학습 단계에 따라 노이즈를 동적으로 조절함으로써 RL의 안정성과 성능을 향상시켰다.

광범위한 실험 결과, QeRL은 다양한 모델 크기에서

- **정확도**,
- **16비트 LoRA 대비 성능**,
- **QLoRA 대비 성능**  
  모두에서 일관된 개선을 보였다.

또한 NVFP4 커널의 지원을 통해, QeRL은 end-to-end RL 학습에서 약 **1.5× 속도 향상**과 함께  
메모리 사용량을 크게 절감하여 실용성을 더욱 높였다.

## **Appendix A 윤리 성명(Ethics Statement)**

본 연구는 학계에서 이미 구축되고 검증된 **공개(open-source) 데이터셋만을 사용**한다.  
새로운 텍스트, 영상, 오디오 데이터는 생성하거나 사용하지 않았다.  
활용된 모든 데이터셋은 **연구 목적에 한정**되어 있으며, 어떠한 상업적 목적에도 사용되지 않았다.

## **Appendix B 재현성(Reproducibility) 성명**

연구 공동체가 본 연구를 재현할 수 있도록, 이 프로젝트는 **오픈소스 소프트웨어로 공개될 예정**이다.  
방법론은 Sec.3에서 상세하게 설명되어 있으며, Sec.4.1과 Appendix E에서는 **모든 하이퍼파라미터 설정을 포함한 전체 학습 프로토콜과 구현 세부 사항**을 제공한다.

## **Appendix C 대형 언어모델(LLM) 사용 내역**

본 논문을 작성하는 과정에서, 우리는 GPT-5(OpenAI, 2025)를 포함한 대형 언어모델을 **문장 및 단락 수준의 문장 표현, 문법, 흐름을 다듬는 목적으로만** 사용하였다.  
이 도구는 **아이디어 생성, 실험 설계, 결론 도출**에는 일절 사용되지 않았다.

모든 기술적 내용, 방법론, 해석은 **저자들이 직접 작성하고 검증한 것**이며, LLM의 도움을 받지 않았다.  
사실 오류나 인용 누락의 위험을 최소화하기 위해, LLM이 편집한 모든 문장은 사람에 의해 재검토되었고, 모든 참고문헌은 원 출처와 대조하여 확인했다.  
저자들은 본 논문의 **정확성과 연구 윤리적 완결성에 대한 모든 책임을** 전적으로 부담한다.

# **Appendix D 관련 연구(Related Work)**

## **강화학습 기반 LLM 훈련 (Reinforcement Learning for LLMs)**

최근 연구들은 RL을 활용해 LLM의 추론 능력을 향상시키는 데 집중하고 있다(Min et al., 2024; Chu et al., 2025).  
DeepSeekMath(Shao et al., 2024)는 수학 중심의 데이터로 추가 사전학습을 수행하고 GRPO(Group Relative Policy Optimization)(Shao et al., 2024)를 도입하여 수학적 추론 능력을 개선하였다.

이를 기반으로 DeepSeek-R1(DeepSeek-AI, 2025)은 **RL만으로도 강력한 추론 능력을 이끌어낼 수 있음**을 보이며, 대규모 학습을 통해 상용 모델 수준의 성능에 도달했다.

이와 보완적으로, DAPO(Yu et al., 2025)는 **최적화 단계를 분리한(decoupled) 오픈소스 RL 프레임워크**를 제시해 간소화된 학습 파이프라인으로 경쟁력 있는 성능을 달성하였다.

GSPO(Zheng et al., 2025)는 시퀀스 수준의 최적화를 통해 RL 훈련의 안정성과 분산 감소를 달성하였으며, 대규모 mixture-of-experts 모델에서도 효과적임을 입증했다.

HybridFlow(Sheng et al., 2025)는 **혼합 제어 흐름(hybrid control flow)**과 **3D-HybridEngine**을 갖춘 유연한 RLHF 프레임워크를 도입했다.

이러한 연구 흐름은 RL을 통한 LLM 추론 능력 향상에 있어 지속적인 진전이 이루어지고 있음을 보여준다.

## **LLM 양자화 (Quantization for LLMs)**

양자화는 대규모 언어모델의 파라미터 정밀도를 줄여 모델을 압축하고 효율성을 향상시키는 핵심 기술이다.  
가장 일반적인 방식인 사후훈련 양자화(Post-Training Quantization, PTQ)(Dettmers et al., 2022; Frantar et al., 2022; Xiao et al., 2023; Shao et al., 2023; Lin et al., 2024)는 **재훈련 없이도** 사전 학습된 모델을 비용 효율적으로 변환할 수 있다.

최근 연구들은 성능을 유지하면서도 초저비트 양자화로 확장되고 있다(Huang et al., 2024c; Dettmers et al., 2023b; Shang et al., 2023; Huang et al., 2024b; Liao & Monz, 2024; Tseng et al., 2024; Huang et al., 2024a). 또한 QAT(Quantization Aware Training)의 견고성을 개선하는 연구들도 등장하고 있다(Liu et al., 2023; Chen et al., 2024a).

새로운 정밀도 포맷도 적극 개발되고 있는데, NF4(Dettmers et al., 2023a), FP4(Tseng et al., 2025; Chmiel et al., 2025), MXFP4(Chmiel et al., 2025)는 정확한 가중치 표현을 가능하게 하며, 높은 압축률에도 성능 저하를 최소화하거나 개선하기도 한다.

NVFP4(NVIDIA, 2024)는 NVIDIA Blackwell GPU 아키텍처와 함께 도입된 혁신적인 **4비트 부동소수점 포맷**으로, 기존의 초저비트 “micro FP” 포맷 개념을 확장하여 더욱 유연하고 실용적인 정밀도 선택지를 제공한다(Zhang et al., 2025b; Castro et al., 2025; Lee et al., 2024).

## **효율적 파인튜닝 (Efficient Fine-tuning)**

효율적 파인튜닝은 LLM을 적은 연산 비용으로 새로운 과제에 적응시키는 데 필수적이다.  
LoRA(Hu et al., 2022)는 동결된 가중치 행렬에 저랭크 어댑터를 삽입하는 방식으로 이 분야를 개척했다.

DoRA(Liu et al., 2024)는 업데이트를 방향과 크기 성분으로 분해해 저랭크 제약을 완화하고 안정성을 개선했다.  
QLoRA(Dettmers et al., 2023a)는 LoRA와 4비트 양자화를 결합하여 자원 사용을 추가로 감소시켰고, LongLoRA(Chen et al., 2024b)는 긴 문맥을 위한 파인튜닝 기법을 제안했다.  
Tina(Wang et al., 2025)는 작은 규모의 모델도 LoRA 기반 RL을 통해 추론 능력을 향상시킬 수 있음을 보였다.

LoRA 계열(Hu et al., 2022) 외에도 다양한 효율적 파인튜닝 방법들이 있다:

- Prompt Tuning
- Prefix Tuning
- IA3
- BitFit
- Fisher-masked tuning
- Input-tuning

(Lester et al., 2021; Li & Liang, 2021; Liu et al., 2022; Zaken et al., 2022; Sung et al., 2021; An et al., 2022; Guo et al., 2023)

이러한 연구들은 실제 LLM 적용에서 효율적 파인튜닝의 중요성을 강하게 보여준다.

# **Appendix E 실험 하이퍼파라미터(Experiment Hyperparameters)**

## **학습 데이터 및 보상 함수(Training Data and Reward Function)**

본 연구에서는 추론 능력 평가에 널리 사용되는 **Qwen2.5-3B-Instruct**, **Qwen2.5-7B-Instruct**, **Qwen2.5-14B-Instruct**, **Qwen2.5-32B-Instruct** 모델을 학습하였다.  
여타 연구들이 수학 특화 모델을 활용하는 것과 달리, 우리는 **범용(base) 모델로부터 학습 성능을 직접 평가하는 것**을 목표로 한다.

또한 QeRL은 Qwen3 시리즈 등 **다른 모델 계열에도 자연스럽게 적용 가능**하다.

- **GSM8K 데이터셋**:  
  GRPO를 사용해 Qwen2.5-3B-Instruct 및 Qwen2.5-7B-Instruct 모델을 주로 학습
- **BigMath 데이터셋**:  
  DAPO를 사용해 Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct 모델 학습

학습 데이터 난이도 선택 기준은 다음과 같다.

- 7B 및 14B 모델: **중·고난도(레벨 3–5)** 데이터 사용
- 32B 모델: **고난도(레벨 4–5)** 데이터만 사용

문제 프롬프트에는 다음의 문장을 suffix로 추가한다:

**“Solve the following math problem step by step.”**

그리고 추론 과정(reasoning)과 최종 답안(answer)은 각각  
<think> </think>, <answer> </answer>  
태그로 감싸는 구조를 사용한다.

예시 형식:

```
<think> reasoning process here </think>
<answer> answer here </answer>
```

또는

```
<think> ... </think>
<answer> ... </answer>
```

## **RL Training Configuration**

GRPO와 DAPO 모두에서, 우리는 **엔트로피 보상(entropy loss)**이나 **KL 보상(KL loss)**을 사용하지 않고 **표 4의 하이퍼파라미터**를 적용하였다. 4비트 학습에서는 학습률을 **1e-5**로 설정한다.

그러나 **BF16 기반 LoRA 모델은 학습 안정성이 매우 약하기 때문에**, 학습률을 **5e-6보다 높게 설정할 수 없으며**, 그 이상으로 설정할 경우 **학습 후반부에서 모델이 붕괴(collapse)**된다.

![](/assets/images/posts/606/img_31.png)

![](/assets/images/posts/606/img_32.png)

![](/assets/images/posts/606/img_33.png)

![](/assets/images/posts/606/img_34.png)

![](/assets/images/posts/606/img_35.png)

![](/assets/images/posts/606/img_36.png)

![](/assets/images/posts/606/img_37.png)

![](/assets/images/posts/606/img_38.png)

# **Appendix H 추가 학습 실험(Additional Experiments of Training)**

![](/assets/images/posts/606/img_39.png)

**그림 12:** 7B 모델의 학습 보상 곡선.

![](/assets/images/posts/606/img_40.png)

**그림 13:** 32B 모델의 학습 보상 곡선.

## **Different Model의 학습 보상 비교**

그림 12와 그림 13은 복잡한 추론 데이터셋에서 QeRL과 16비트 LoRA 학습 성능을 비교한 결과를 보여준다.

그림 12에서는 BigMath 데이터셋(난이도 3–5)에서 학습한 **Qwen2.5-7B-Instruct** 모델의 보상 곡선을 제시하며, 이는 그림 8의 확장 비교 실험이다.  
양자화 모델에서 QeRL이 제공하는 **탐색력 증가** 덕분에, **약 200 스텝 이후 보상이 급격히 상승**하는 반면,  
16비트 LoRA는 비슷한 수준에 도달하기 위해 **500 스텝 이상**이 필요하다.

그림 13은 가장 높은 난이도(레벨 4–5)의 데이터를 사용해 학습한 **Qwen2.5-32B-Instruct** 모델의 학습 보상을 보여준다.  
32B 모델에서는 3B·7B·14B 모델처럼 두 방식의 차이가 크게 벌어지지는 않지만,  
**여전히 QeRL이 LoRA보다 일관적으로 더 높은 보상을 달성**한다.

## **Entropy 추가 실험**

![](/assets/images/posts/606/img_41.png)

**그림 14:** RL 스텝에 따른 엔트로피 변화 곡선.

그림 5의 확장 실험으로서, 그림 14는 **Qwen2.5-14B-Instruct** 모델의 다양한 학습 스텝에서 엔트로피 변화를 나타낸다.  
QeRL에서의 엔트로피 값은 RL 과정 전반에서 LoRA보다 **항상 더 높게 유지**되며,  
특히 **학습 초반 단계에서 더 뚜렷한 차이**를 보인다.

이는 QeRL이 RL 학습에서 **탐색을 강화한다는 명확한 증거**이다.  
엔트로피가 높다는 것은 모델이 더 넓은 해 공간을 탐색하고 있다는 의미이며,  
양자화를 통한 탐색력 증가가 모델이 복잡한 환경을 더 효과적으로 탐색하고 최적화에 도움을 준다는 점을 시사한다.

이 실험 결과는 양자화가 **탐색–활용(exploration–exploitation) 균형을 향상**시키는 역할을 한다는 본 논문의 주장을 뒷받침한다.

## **Noise Scheduler**

![](/assets/images/posts/606/img_42.png)

**그림 15:** 서로 다른 노이즈 스케줄러의 노이즈 변화 곡선.

그림 15는 본 연구에서 사용한 노이즈 스케줄러의 변화를 보여준다.  
4가지 서로 다른 decay 방식이 시각화되어 있다:

- Linear decay
- Exponential decay
- Cosine decay
- Logarithmic decay

스케줄러는 총 10개의 스테이지로 노이즈 크기를 조절해 학습을 유도한다.

- **Linear decay**: 각 스테이지마다 균일하게 노이즈 감소
- **Exponential decay**: 학습 초기에 노이즈를 빠르게 줄이고 후반에는 작은 노이즈 유지
- **Cosine decay**: 코사인 곡선을 따라 부드럽게 감소
- **Logarithmic decay**: 초기 급격히 감소 후 후반에 안정

이 중, 본 연구에서는 **exponential decay**를 선택했다.  
그 이유는 학습 후반에 노이즈 크기를 **더 안정적이고 작은 수준으로 유지**해  
보상 곡선이 더 **안정적이고 높은 값**으로 수렴하는 것을 확인했기 때문이다.

노이즈 레벨을 정교하게 조절할 수 있는 이러한 스케줄링 전략은  
학습 과정에서 **탐색과 수렴의 균형을 맞추는 데 핵심적 역할**을 한다.

# **Appendix I 추가 Ablation 연구(Additional Ablation Study)**

![](/assets/images/posts/606/img_43.png)

**그림 16:** QeRL에서의 learning rate ablation (Qwen2.5-7B-Instruct).

![](/assets/images/posts/606/img_44.png)

**그림 17:** LoRA에서의 learning rate ablation (Qwen2.5-7B-Instruct).

## **Learning Rate Ablation**

우리는 학습률 변화가 \*\*양자화 모델(QeRL)\*\*과 \*\*16비트 모델(LoRA)\*\*의 성능에 어떤 영향을 미치는지 분석하였다.

그림 16과 그림 17에서 확인할 수 있듯이:

- 학습률을 **5e-6**처럼 작게 설정한 경우,  
  QeRL은 LoRA보다 **소폭 더 높은 보상**을 달성하며,  
  보상 값은 **약 0.95**에 도달한다.
- 학습률을 **3e-5**로 높이면,  
  어댑터의 업데이트 크기가 증가하면서 **보상이 더 빠르게 상승**하고  
  **모델 수렴 속도도 빨라진다**.

하지만:

- **16비트 LoRA 모델**에서는 업데이트 크기가 지나치게 커져  
  모델 불안정성이 증가하고,  
  \*\*학습이 붕괴(collapse)\*\*하는 경우가 잦다.

반면에:

- **QeRL은 NVFP4 양자화 노이즈 덕분에 높은 학습률에서도 매우 높은 안정성**을 보인다.  
  양자화로 인한 노이즈가 일종의 완충 작용을 하여 업데이트를 안정화시키기 때문이다.

이로 인해 QeRL은 **큰 학습률에서도 안정적인 학습을 유지**할 수 있으며,  
보상 상승 속도는 **16비트 모델보다 거의 2배 빠르다**.

이 실험은 QeRL이 특히 **고학습률(high-learning-rate)** 환경에서  
확실한 \*\*적응성(adaptability)\*\*과 \*\*효율성(efficiency)\*\*을 제공한다는 점을 강하게 입증한다.

# **Appendix J 추가 효율성 실험(More Efficiency Experiments)**

![](/assets/images/posts/606/img_45.png)

## **표 5: Qwen2.5-3B-Instruct 모델의 메모리 절감 및 속도 향상**

이 표는 배치 크기 2와 8에서 rollout 단계의 처리량(tokens/s)을 보여준다.  
각 입력 길이는 256 토큰, 최대 생성 길이는 2048 토큰이다.  
“W#”는 weight 데이터 포맷, “BS#”는 배치 크기, “E2E”는 GRPO 학습의 end-to-end 속도를 의미한다.  
“GC”는 gradient checkpointing을 의미한다.

![](/assets/images/posts/606/img_46.png)

## **표 6: Qwen2.5-7B-Instruct 모델의 메모리 절감 및 속도 향상**

조건은 표 5와 동일하다.

![](/assets/images/posts/606/img_47.png)

## **표 7: Qwen2.5-14B-Instruct 모델의 메모리 절감 및 속도 향상**

(표 내용 동일하게 유지)

![](/assets/images/posts/606/img_48.png)

## **표 8: Qwen2.5-32B-Instruct 모델의 메모리 절감 및 속도 향상**

(표 내용 동일하게 유지)

![](/assets/images/posts/606/img_49.png)

## **표 9: LoRA rank 변화에 따른 rollout 단계 처리량 비교**

vLLM 엔진에서 rank별 tokens/s를 측정했으며, 설정은 표 7과 동일하고 batch size는 1이다.

## **설명**

표 5, 6, 7, 8은 Qwen2.5-3B, 7B, 14B, 32B 모델을 대상으로  
배치 크기 2와 8에서 추가적인 속도 벤치마크를 제공한다.

### **작은 모델(3B, 7B)**

- 최대 속도를 얻기 위해 **gradient checkpointing**이나 **Liger loss** 같은 메모리 절감 기법을 사용하지 않았다.

### **큰 모델(14B, 32B)**

- 모델 크기가 크고,  
  RL 학습 과정에서 gradient 기반 importance sampling 때문에 계산량이 급증하므로  
  **gradient checkpointing을 활성화하여** 계산 효율을 높였다.
- 작은 GPU 메모리를 사용할 경우  
  **checkpointing 활성화가 필수적**이며,  
  속도가 다소 느려질 수 있다.

### **NVFP4 기반 rollout 속도 향상**

- Marlin kernel 최적화(NVIDIA Blackwell, 2024) 덕분에  
  NVFP4는 rollout 단계에서 **1.0× ~ 2.0×의 속도 향상**을 보였다.
- 모델이 클수록 더 큰 속도 향상이 나타나며,  
  **32B 모델에서는 최대 2.0× 개선**이 관찰된다.
- 이는 NVFP4 정밀도가 **대규모 모델일수록 더 큰 이점**을 제공한다는 것을 시사한다.

### **E2E RL 효율 평가**

- GRPO 한 스텝(latency)은  
  rollout 생성, log-prob 계산, 파라미터 업데이트를 포함한 전체 wall-clock 시간을 의미한다.
- batch = 2, 8 기준으로 비교했으며  
  입력 길이 256, 최대 생성 길이 2048로 고정하였다.
- 공정한 비교를 위해,  
  BF16과 NVFP4 모두 동일한 GPU 메모리 예산을 설정하였다:
  - 3B: 0.20
  - 7B: 0.30
  - 14B: 0.45
  - 32B: 0.40 (단일 GPU 학습 가능하도록 조정)

이 조건에서, E2E latency 감소는 rollout 단계의 속도 향상 경향을 그대로 반영하며,  
**모델이 클수록 더 큰 가속 이점**이 나타난다.  
특히 Qwen2.5-14B-Instruct 모델에서 가장 큰 효율 향상이 관찰되었다.

### **LoRA rank 변화 실험(표 9)**

- NVFP4는 **모든 rank에서 16bit 대비 더 빠른 rollout 속도**를 유지했다.
- rank가 커질수록 BF16과 NVFP4 모두 속도가 감소하지만,  
  NVFP4는 항상 우위를 유지한다.
- 이는 NVFP4가 **작은 모델부터 큰 모델까지, 다양한 adapter 설정에서도 안정적 효율성**을 제공함을 보여준다.

종합적으로, NVFP4는 고급 kernel과 결합될 때  
**추론 효율을 극대화할 수 있는 강력한 포맷**임을 확인할 수 있다.

# **Appendix K 한계점 분석(Limitation Analysis)**

우리는 QeRL이 기존 16bit LoRA RL 학습보다 우수한 성능을 달성하고,  
또한 16bit full-parameter RL 학습에 필적하는 정확도를 유지하면서도  
학습 속도는 **2배 이상 빠르다**는 점을 보여주었다.

그러나 다음과 같은 한계가 존재한다.

### **1. 대규모 모델(70B 이상)에 대한 검증 부족**

- 현재 실험은 3B~32B 범위에서 수행되었으며,  
  RL은 SFT보다 훨씬 많은 계산 자원을 요구하기 때문에  
  **70B 이상의 초대규모 모델에서 같은 성능을 재현할 수 있는지**는  
  향후 연구가 필요하다.

### **2. 벤치마크 다양성 부족**

- 우리는 GSM8K, MATH500, AIME24/25, AMC23 등  
  대표적인 reasoning 벤치마크에서는 폭넓게 검증했지만,  
  다음 영역에서는 평가가 이루어지지 않았다.
  - 코드 관련 벤치마크
  - 일반 자연어 처리(NLU/NLG) 벤치마크
  - reasoning 외 도메인

다만 QeRL은 다양한 데이터 유형과 과제에  
자연스럽게 적용 가능하므로,  
커뮤니티가 더 넓은 범위의 데이터를 사용해 연구를 확장하기를 기대한다.
