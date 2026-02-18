---
title: "STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis"
date: 2025-06-17 17:51:12
categories:
  - 인공지능
tags:
  - starflow
---

<https://arxiv.org/abs/2506.06276>

[STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis](https://arxiv.org/abs/2506.06276)

![](/assets/images/posts/572/img.png)

**그림 1**: 다양한 종횡비로 생성된 고해상도 텍스트 조건 샘플들. 모두 3.8B 파라미터 규모의 STARFlow 모델에서 생성되었으며, 시각화의 편의를 위해 해상도가 조정되어 있음.

**초록(Abstract)**

우리는 고해상도 이미지 합성에서 강력한 성능을 보이는, 정규화 흐름(normalizing flows)에 기반한 확장 가능한 생성 모델인 **STARFlow**를 소개한다. STARFlow의 핵심 구성 요소는 **TARFlow(Transformer Autoregressive Flow)**로, 이는 정규화 흐름과 Autoregressive Transformer 아키텍처를 결합한 구조이며 최근 이미지 모델링 분야에서 인상적인 성과를 보인 바 있다.

이 연구에서는 먼저 TARFlow가 연속 확률 분포를 모델링하는 데 있어 **이론적으로 보편적(universal)**이라는 사실을 입증한다. 이 기반 위에 우리는 모델의 확장성을 크게 높여주는 일련의 아키텍처 및 알고리즘 혁신들을 도입하였다:

1. **딥-셸로우(Deep-Shallow) 구조**: 대부분의 모델 용량을 담당하는 깊은 Transformer 블록과, 계산량은 적지만 무시할 수 없는 기여를 하는 얕은 Transformer 블록 몇 개를 조합함.
2. **사전학습된 오토인코더의 잠재 공간(latent space)에서의 학습**: 픽셀을 직접 모델링하는 것보다 훨씬 효과적인 방식임을 입증함.
3. **샘플 품질을 크게 향상시키는 새로운 가이던스 알고리즘**.

무엇보다도, STARFlow는 **단일 end-to-end 정규화 흐름 모델**로 유지되며, 이는 **이산화(discretization) 없이 연속 공간에서 정확한 최대우도 학습**을 가능하게 한다.

STARFlow는 클래스 조건 및 텍스트 조건 이미지 생성 양쪽 모두에서 경쟁력 있는 결과를 달성하며, 생성된 샘플 품질은 최신 확산 모델(diffusion model)에 근접한다. 우리가 알기로, 이 연구는 이 규모와 해상도에서 정규화 흐름의 성공적인 적용을 처음으로 입증한 사례이다.

### 1. 서론

최근 몇 년간 고해상도 텍스트-이미지 생성 모델링 분야는 눈부신 발전을 이루었으며, 최첨단 기법들은 주로 두 가지 뚜렷한 범주로 나뉩니다. 한편으로는, 연속 공간(continuous space)에서 작동하는 **확산 모델(diffusion models)**(Ho et al., 2020; Rombach et al., 2022; Peebles & Xie, 2023; Esser et al., 2024)이 이미지 품질 측면에서 새로운 기준을 제시해왔습니다. 하지만 이러한 모델은 반복적인 디노이징(denoising) 과정을 필요로 하므로, 학습과 추론 모두에서 계산 비용이 매우 높다는 단점이 있습니다. 다른 한편으로는, **오토레그레시브(autoregressive) 방식의 이미지 생성 방법**(Yu et al., 2022; Sun et al., 2024; Tian et al., 2024)이 있으며, 이는 대형 언어 모델(LLMs, Brown et al., 2020; Dubey et al., 2024)의 성공에서 영감을 받아, 양자화(quantization)를 통해 이산 공간(discrete space)에서 이미지를 모델링함으로써 이러한 비효율성을 회피합니다. 그러나 양자화는 심각한 제약을 초래할 수 있으며, 결과적으로 이미지의 충실도(fidelity)를 저해할 수 있습니다. 최근에는 오토레그레시브 기법을 **연속 공간에 직접 적용하는 하이브리드 모델**(Li et al., 2024; Gu et al., 2024b; Fan et al., 2024)에 대한 탐색이 유망한 흐름으로 등장했습니다. 하지만 두 패러다임(확산과 오토레그레시브)의 본질적으로 다른 특성은 효과적인 통합에 있어 추가적인 복잡성을 유발합니다.

이 논문에서는 또 다른 모델링 접근 방식인 **정규화 흐름(Normalizing Flows, NFs)**(Rezende & Mohamed, 2015; Dinh et al., 2016)에 주목합니다. 이는 **우도 기반(likelihood-based)** 생성 모델 계열로, 최근 생성형 AI의 급류 속에서는 상대적으로 적은 관심을 받아왔습니다. 우리는 먼저 강력한 Transformer 아키텍처와 오토레그레시브 흐름(Autoregressive Flows, AFs)(Kingma et al., 2016; Papamakarios et al., 2017)을 결합한 최근 모델인 **TARFlow**(Zhai et al., 2024)를 분석하는 데서 출발합니다. TARFlow는 정규화 흐름(NF)이 유효한 모델링 프레임워크임을 보여주는 유망한 결과를 제시하지만, 이것이 확산 모델이나 이산 오토레그레시브 모델들과 비교해 확장 가능한 방법으로 작동할 수 있는지는 아직 불분명합니다. 이를 위해 우리는 **STARFlow**라는 생성 모델 계열을 제안합니다. 이 모델은 **정규화 흐름 기반 모델이 고해상도 및 대규모 이미지 모델링에도 성공적으로 일반화될 수 있음**을 최초로 입증합니다. 우리는 먼저 다중 블록 AF가 연속 분포를 모델링함에 있어 **보편성(universality)**을 지닌다는 이론적 통찰을 제공하며, 이를 기반으로 **새로운 Deep–Shallow 아키텍처**를 제안합니다. 실험 결과, 플로우 개수와 각 플로우에 할당된 Transformer의 깊이 및 너비와 같은 아키텍처 구성은 모델 성능에 결정적인 역할을 하는 것으로 나타났습니다. TARFlow에서는 모든 플로우에 균등한 깊이를 배분했지만, 우리는 대부분의 파라미터를 첫 번째 AF 블록(사전 분포에 가장 가까운)에 집중시키고, 이후 몇 개의 얕지만 의미 있는 블록을 추가하는 **비균등 구조(skewed architecture)**가 더 효과적임을 발견했습니다. 중요하게도, STARFlow는 **여전히 독립적인 end-to-end 정규화 흐름 모델**로서, 연속 공간에서 양자화 없이 **최대우도 학습**을 지원합니다. 우리는 픽셀 공간이 아닌, **사전학습된 오토인코더의 잠재 공간(latent space)**에서 AF를 학습하며, 이는 고해상도 입력을 보다 효과적으로 모델링할 수 있게 해줍니다. 이는 직관적이지만 중요한 관찰로, 실험에서도 직접 픽셀 공간에서 학습할 때보다 훨씬 우수한 성능을 보였습니다. TARFlow와 유사하게, **노이즈 주입(noise injection)**은 여전히 중요합니다. 우리는 디코더를 파인튜닝하며 **노이즈가 섞인 latent**에서 학습하고, 동시에 샘플링 파이프라인을 단순화했습니다. 또한, AF에 대한 **classifier-free guidance (CFG)** 알고리즘을 더 원리적인 방식으로 재검토하고, **새로운 가이던스 알고리즘**을 제안함으로써 특히 높은 guidance weight에서 **텍스트-이미지 생성의 품질을 크게 향상**시켰습니다.

이러한 혁신을 통해, 우리는 NF 기반 모델이 **대규모 고해상도 이미지 생성에 성공적으로 적용된 첫 사례**를 제시합니다. 우리의 접근 방식은 기존 확산 기반 또는 오토레그레시브 기반 방법에 대한 **확장 가능하고 효율적인 대안**을 제공하며, 클래스 조건 이미지 생성과 대규모 텍스트-이미지 합성 모두에서 **경쟁력 있는 성능**을 보입니다. 또한, STARFlow는 매우 유연한 프레임워크로, **이미지 인페인팅(inpainting)**이나 **지시 기반 이미지 편집(instruction-based editing)**과 같은 다양한 설정에도 **간단한 파인튜닝을 통해 적용이 가능함**을 실험을 통해 보여줍니다.

### 2. 사전 지식

#### 2.1 정규화 흐름(Normalizing Flows)

본 논문에서는 **정규화 흐름(Normalizing Flows, NFs)**(Rezende & Mohamed, 2015; Dinh et al., 2014, 2016)을 **변수 변환 공식(change of variable formula)**을 따르는 **우도 기반(likelihood-based)** 모델의 한 종류로 다룹니다.

![](/assets/images/posts/572/img_1.png)

![](/assets/images/posts/572/img_2.png)

![](/assets/images/posts/572/img_3.png)

---

말은 어렵게했지만 결국 diffusion대신에 autoregressive한 transformer 썼다.

### ? 배경 비교:

- **Diffusion 모델들**: 고품질 이미지를 생성하지만,
  - 수십~수백 번의 반복적 디노이징이 필요해서
  - **학습과 추론 비용이 큼**
- **기존 Autoregressive 모델들 (예: LLM 기반 이미지 생성)**:
  - 효율적이지만
  - \*\*이산 공간(quantized space)\*\*에서 작동 → 이미지 품질 저하 우려
- **TARFlow / STARFlow (제안 모델)**:
  - 연속 공간(continuous space)에서 **autoregressive Transformer + normalizing flow**를 사용
  - **Diffusion처럼 정밀하게**
  - **Autoregressive처럼 효율적으로**

---
