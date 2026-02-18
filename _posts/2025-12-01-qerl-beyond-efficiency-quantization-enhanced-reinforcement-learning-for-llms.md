---
title: "QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs"
date: 2025-12-01 23:45:26
categories:
  - 인공지능
---

<https://arxiv.org/abs/2510.11696>

[QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs](https://arxiv.org/abs/2510.11696)

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
