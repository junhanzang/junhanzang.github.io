---
title: "DeepSeek-V3 Technical Report"
date: 2025-02-14 21:44:01
categories:
  - 인공지능
tags:
  - deepseek-v3 technical report
---

<https://arxiv.org/abs/2412.19437>

[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

**초록**  
우리는 토큰당 37B가 활성화되고 총 671B 파라미터를 갖춘 강력한 Mixture-of-Experts(MoE) 언어 모델인 DeepSeek-V3를 소개한다. 효율적인 추론과 비용 효율적인 학습을 위해 DeepSeek-V3는 Multi-head Latent Attention(MLA) 및 DeepSeekMoE 아키텍처를 채택했으며, 이는 DeepSeek-V2에서 철저히 검증되었다. 또한 DeepSeek-V3는 부가 손실(auxiliary loss) 없이 로드 밸런싱을 달성하는 전략을 처음 도입하고, 더 강력한 성능을 위해 다중 토큰 예측(multi-token prediction) 학습 목표를 설정하였다. 우리는 DeepSeek-V3를 14.8조 개의 다양하고 고품질 토큰에 대해 사전 학습한 뒤, 지도 학습(Supervised Fine-Tuning) 및 강화 학습(Reinforcement Learning)을 거쳐 모델의 역량을 극대화하였다. 종합 평가 결과, DeepSeek-V3는 다른 오픈소스 모델들을 능가하며, 선도적인 폐쇄형 모델들과 견줄 만한 성능을 보여준다. 우수한 성능에도 불구하고 DeepSeek-V3는 전체 학습에 단 2.788M H800 GPU 시간이 필요하다. 더욱이 학습 과정이 매우 안정적이어서, 전체 학습 기간 동안 되돌릴 수 없는 손실 스파이크가 발생하거나 롤백 조치가 필요했던 적이 전혀 없었다. 모델 체크포인트는 <https://github.com/deepseek-ai/DeepSeek-V3>에서 확인할 수 있다.

![](/assets/images/posts/509/img.png)

**Figure 1:** DeepSeek-V3 및 유사 모델들의 벤치마크 성능.

![](/assets/images/posts/509/img_1.png)

**1 서론**  
최근 몇 년간 대규모 언어 모델(LLM)은 급속도로 발전하며 (OpenAI, 2024a; Anthropic, 2024; Google, 2024), 범용 인공지능(Artificial General Intelligence, AGI)에 한 걸음씩 다가가고 있다. 폐쇄형 모델뿐 아니라 DeepSeek 계열(DeepSeek-AI, 2024b, c; Guo et al., 2024; DeepSeek-AI, 2024a), LLaMA 계열(Touvron et al., 2023a, b; AI@Meta, 2024a, b), Qwen 계열(Qwen, 2023, 2024a, 2024b), Mistral 계열(Jiang et al., 2023; Mistral, 2024) 등 여러 오픈소스 모델들 역시 빠르게 발전하여, 이들 폐쇄형 모델들과의 격차를 좁히고자 노력하고 있다. 이러한 오픈소스 모델 역량의 한계를 한층 더 확장하기 위해, 우리는 모델을 대규모로 확장하여 토큰당 37B가 활성화되고 총 671B 파라미터를 갖춘 대규모 Mixture-of-Experts(MoE) 모델인 DeepSeek-V3를 소개한다.

미래지향적 관점에서 우리는 항상 강력한 모델 성능과 경제적인 비용의 균형을 추구한다. 이에 따라 DeepSeek-V3는 추론 효율을 위해 Multi-head Latent Attention(MLA)(DeepSeek-AI, 2024c)을, 그리고 비용 효율적인 학습을 위해 DeepSeekMoE(Dai et al., 2024)를 여전히 채택한다. 이 두 아키텍처는 DeepSeek-V2(DeepSeek-AI, 2024c)에서 검증되어, 효율적인 학습과 추론을 유지하면서도 견고한 모델 성능을 제공함이 입증되었다. 또한 기본 아키텍처 외에도 모델 역량을 더욱 향상하기 위해 두 가지 추가 전략을 구현하였다. 첫째, DeepSeek-V3는 부가 손실(auxiliary loss)을 사용하지 않고 로드 밸런싱을 달성하는 전략(Wang et al., 2024a)을 새롭게 도입하여, 로드 밸런싱을 촉진하는 과정에서 발생할 수 있는 모델 성능 저하를 최소화한다. 둘째, DeepSeek-V3는 멀티 토큰 예측(multi-token prediction) 학습 목표를 채택하여 전체적인 평가 지표에서 모델 성능을 향상시킨다는 사실을 확인하였다.

효율적인 학습을 달성하기 위해, 우리는 FP8 혼합 정밀도(mixed precision) 학습을 지원하고 학습 프레임워크에 대한 종합적인 최적화를 구현하였다. 저정밀도 학습은 (Kalamkar et al., 2019; Narang et al., 2017; Peng et al., 2023b; Dettmers et al., 2022) 효율적 학습을 위한 유망한 해법으로 대두되고 있으며, 이는 하드웨어 기술 발전(Micikevicius et al., 2022; Luo et al., 2024; Rouhani et al., 2023a)과 맞물려 빠르게 발전하고 있다. 본 논문에서는 FP8 혼합 정밀도 학습 프레임워크를 도입하고, 초대형 규모 모델에서 FP8 학습이 실제로 가능하고 효과적임을 처음으로 검증하였다. FP8 계산 및 저장을 지원함으로써 학습 속도 가속과 GPU 메모리 사용량 감소 효과를 모두 얻었다. 학습 프레임워크 측면에서는 파이프라인 병렬화를 효율적으로 수행하기 위해 DualPipe 알고리즘을 설계하였고, 이를 통해 파이프라인 버블을 줄이고 대부분의 통신을 학습 과정 중에 연산과 겹치도록(overlap) 처리한다. 이런 겹침으로 인해 모델 규모가 더 커져도 일정한 연산 대 통신 비율을 유지하기만 하면, 여러 노드에 걸쳐 파인 그레이닝(fine-grained)된 전문가들을 배치하더라도 거의 0에 가까운 all-to-all 통신 오버헤드로 학습을 진행할 수 있다. 또한 우리는 InfiniBand(IB)와 NVLink 대역폭을 최대한 활용하기 위해 효율적인 노드 간 all-to-all 통신 커널을 개발하였다. 아울러 메모리 사용량을 꼼꼼하게 최적화하여, 비용이 많이 드는 텐서 병렬성(tensor parallelism)을 사용하지 않고도 DeepSeek-V3를 학습할 수 있도록 하였다. 이로써 학습 효율을 크게 끌어올릴 수 있었다.

사전 학습 단계에서는 14.8조 개의 다양하고 높은 품질의 토큰을 학습에 활용하였다. 학습 과정은 매우 안정적으로 진행되어, 전체 훈련 기간 동안 되돌릴 수 없는 수준의 손실(spike)이 발생하거나 롤백을 수행해야 하는 상황이 한 번도 없었다. 이후 DeepSeek-V3에 대해 두 단계로 맥락 길이(context length)를 확장하였다. 첫 번째 단계에서 최대 맥락 길이를 32K로 확장하고, 두 번째 단계에서 이를 128K까지 늘렸다. 그 뒤, DeepSeek-V3의 베이스(base) 모델을 대상으로 한 지도 학습(Supervised Fine-Tuning, SFT)과 강화 학습(RL) 단계를 진행하여 사람의 선호도와 더욱 정교하게 정렬(alignment)시키고 모델의 잠재력을 극대화하였다. 이 과정에서 DeepSeek-R1 계열 모델로부터 추론 능력을 증류(distillation)하는 한편, 모델 정확도와 생성 길이 간 균형도 신중히 유지하였다.

우리는 DeepSeek-V3를 다양한 벤치마크에서 평가하였다. 경제적인 학습 비용에도 불구하고, 종합 평가 결과 DeepSeek-V3-Base는 코드와 수학 영역에서 특히 강력한 성능을 보이며 현재 공개된 베이스 모델 중 가장 우수한 성능을 보여준다. Chat 버전 또한 다른 오픈소스 모델들을 상회하고, GPT-4o 및 Claude-3.5-Sonnet과 같은 선도적인 폐쇄형 모델들과의 성능 격차를 상당히 좁혀준다.

![](/assets/images/posts/509/img_2.png)

**Table 1:** DeepSeek-V3의 전체 학습 비용 추정. H800 GPU 대여 비용을 GPU 시간당 $2로 가정.

마지막으로, Table 1에 정리한 바와 같이 알고리즘·프레임워크·하드웨어를 함께 최적화한 결과, DeepSeek-V3의 학습 비용은 매우 경제적이다. 사전 학습 단계에서 매 1조 토큰당 약 18만 H800 GPU 시간이 필요하며, 이는 2048대의 H800 GPU를 운용할 경우 약 3.7일 만에 처리 가능하다. 따라서 전체 사전 학습(14.8조 토큰)을 완료하는 데 약 2,664K GPU 시간이 소요되어 두 달이 채 걸리지 않는다. 이후 맥락 길이 확장에 119K GPU 시간, 후속 포스트 트레이닝에 5K GPU 시간이 추가로 들며, 결과적으로 DeepSeek-V3 전체 학습에는 총 2.788M GPU 시간이 소요된다. H800 GPU 대여 비용을 시간당 $2로 가정하면, 총 학습 비용은 약 $5.576M에 불과하다. 단, 이는 DeepSeek-V3의 공식 학습에 필요한 비용만을 산정한 것이며, 아키텍처·알고리즘·데이터 관련 사전 연구나 소규모 실험에 소요된 비용은 포함되지 않았다.

### 주요 기여

#### 아키텍처: 혁신적 로드 밸런싱 전략 및 학습 목표

- DeepSeek-V2의 효율적인 아키텍처를 기반으로, 부가 손실(auxiliary loss)을 배제한 로드 밸런싱 전략을 처음으로 도입하여, 로드 밸런싱을 촉진하는 과정에서 발생할 수 있는 성능 저하를 최소화한다.
- 멀티 토큰 예측(Multi-Token Prediction, MTP) 학습 목표를 적용해 모델 성능이 향상됨을 확인하였으며, 이는 추론 속도를 높이는 speculative decoding에도 활용 가능하다.

#### 사전 학습: 궁극의 학습 효율 추구

- FP8 혼합 정밀도 학습 프레임워크를 설계하고, 초대형 모델 규모에서 FP8 학습이 실제로 가능하고 효과적임을 처음으로 검증하였다.
- 알고리즘·프레임워크·하드웨어를 공동으로 설계하여, 노드 간 MoE 학습에서 발생하는 통신 병목을 극복하고 연산-통신 겹침을 극대화하였다. 이를 통해 학습 효율을 크게 높이고 비용을 절감하여, 추가 오버헤드 없이 모델 규모를 더욱 확장할 수 있게 되었다.
- H800 GPU 시간 2.664M만으로 14.8조 토큰에 대한 사전 학습을 완료함으로써, 현재 공개된 모델 중 가장 강력한 오픈소스 베이스 모델을 생산하였다. 사전 학습 이후 단계인 맥락 길이 확장과 포스트 트레이닝에는 약 0.1M GPU 시간만 필요하다.

#### 포스트 트레이닝: DeepSeek-R1로부터의 지식 증류

- 길이가 긴 Chain-of-Thought(CoT)를 사용하는 모델인 DeepSeek R1 계열로부터 추론 능력을 일반 LLM(특히 DeepSeek-V3)에 증류하는 새로운 방식을 제안하였다. 이 파이프라인을 통해 R1의 검증과 반추(reflection) 패턴을 DeepSeek-V3에 자연스럽게 접목함으로써 추론 성능을 크게 향상시키면서도, DeepSeek-V3의 출력 스타일과 길이를 필요에 따라 제어할 수 있게 하였다.

### 핵심 평가 결과 요약

- **지식(한국어·영어 등):**
  1. MMLU, MMLU-Pro, GPQA와 같은 교육 분야 벤치마크에서 DeepSeek-V3는 MMLU 88.5, MMLU-Pro 75.9, GPQA 59.1 점수를 기록하며 다른 오픈소스 모델을 능가한다. 동시에 GPT-4o 및 Claude-Sonnet-3.5와 유사한 수준의 성능을 보이며, 이 영역에서 오픈소스와 폐쇄형 모델 간 격차를 좁힌다.
  2. 사실성 평가(factuality) 벤치마크인 SimpleQA, Chinese SimpleQA에서도 오픈소스 모델 중 가장 뛰어난 성능을 보여준다. 영어 사실성(SimpleQA) 측면에서는 GPT-4o와 Claude-Sonnet-3.5에 약간 뒤처지나, 중국어 사실성(Chinese SimpleQA)에서는 이들을 능가하며 중국어 팩트 지식 측면에서의 우수성을 드러낸다.
- **코드, 수학, 추론:**
  1. 수학 관련 벤치마크에서, DeepSeek-V3는 모든 비-장문 CoT(non-long-CoT) 오픈소스·폐쇄형 모델 중 최상위 성능을 기록한다. 특히 MATH-500과 같은 특정 벤치마크에서 o1-preview 모델을 상회하며, 강력한 수학적 추론 역량을 보여준다.
  2. 코딩 관련 태스크(예: LiveCodeBench)에서도 DeepSeek-V3는 경쟁 모델 중 가장 높은 점수를 획득하며, 이 분야를 선도하는 모델로 자리매김한다. 엔지니어링 관련 태스크에서도 Claude-Sonnet-3.5보다 약간 뒤처지긴 하지만, 다른 모델들과 비교했을 때는 상당히 앞서는 성능을 발휘하여 다양한 기술적 벤치마크에서 경쟁력을 입증한다.

이하 본문에서는 먼저 DeepSeek-V3 모델 아키텍처를 상세히 설명한다(2장). 이후 우리의 컴퓨팅 클러스터, 학습 프레임워크, FP8 학습 지원, 추론 배포 전략, 차세대 하드웨어 설계에 대한 제언 등 인프라 전반을 소개한다(3장). 다음으로 사전 학습 과정—데이터 구성, 하이퍼파라미터, 맥락 길이 확장 기법, 평가 및 논의—을 다룬다(4장). 이어 포스트 트레이닝 단계인 지도 학습(SFT)과 강화 학습(RL), 이와 관련된 평가와 논의에 대해 논한다(5장). 마지막으로 DeepSeek-V3의 한계점을 짚고 향후 연구 방향을 제시하며 본 논문을 마무리한다(6장).

**2 아키텍처**  
본 장에서는 먼저 효율적인 추론을 위한 Multi-head Latent Attention(MLA)(DeepSeek-AI, 2024c)와 경제적인 학습을 위한 DeepSeekMoE(Dai et al., 2024)로 구성된 DeepSeek-V3의 기본 아키텍처를 소개한다. 이후 다양한 평가 지표에서 모델 성능을 향상시키는 것으로 확인된 Multi-Token Prediction(MTP) 학습 목표에 대해 설명한다. 명시적으로 언급되지 않은 기타 세부 사항은 DeepSeek-V2(DeepSeek-AI, 2024c)와 동일하다.

![](/assets/images/posts/509/img_3.png)

**Figure 2:** DeepSeek-V3의 기본 아키텍처를 그림으로 나타낸다. DeepSeek-V2와 마찬가지로, 효율적인 추론을 위해 MLA를, 경제적인 학습을 위해 DeepSeekMoE를 채택하였다.

### 2.1 기본 아키텍처

DeepSeek-V3의 기본 아키텍처는 여전히 Transformer(Vaswani et al., 2017) 프레임워크 내에 있다. 효율적인 추론과 경제적인 학습을 위해, DeepSeek-V3는 DeepSeek-V2에서 철저히 검증된 MLA와 DeepSeekMoE를 채택한다. DeepSeek-V2와 비교했을 때 가장 큰 차이는, DeepSeekMoE에 대해 부가 손실 없이(auxiliary-loss-free) 로드 밸런싱을 달성할 수 있는 전략(Wang et al., 2024a)을 추가 도입하여, 로드 밸런싱 과정에서 초래되는 성능 저하를 줄였다는 점이다. 그림 2는 DeepSeek-V3의 기본 아키텍처를 나타내며, 이하 본 절에서는 MLA와 DeepSeekMoE에 대한 세부 내용을 간략히 검토한다.

#### 2.1.1 Multi-Head Latent Attention

![](/assets/images/posts/509/img_4.png)

---

![](/assets/images/posts/509/tfile.svg)

---
