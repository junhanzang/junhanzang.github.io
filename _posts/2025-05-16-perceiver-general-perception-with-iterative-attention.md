---
title: "Perceiver: General Perception with Iterative Attention"
date: 2025-05-16 16:14:54
categories:
  - 인공지능
tags:
  - perceiver
---

<https://arxiv.org/abs/2103.03206>

[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

MLA 개념의 최초 제안

**초록**  
생물학적 시스템은 시각, 청각, 촉각, 고유 감각(proprioception) 등 다양한 양식(modality)으로부터의 고차원 입력을 동시에 처리함으로써 세상을 인지합니다. 반면, 딥러닝에서 사용하는 인지(perception) 모델들은 보통 개별 모달리티에 맞춰 설계되며, 대부분 기존 시각 모델들이 활용하는 지역 격자 구조(local grid structure)와 같은 도메인 특화 가정(domain-specific assumption)에 의존합니다. 이러한 사전 지식(prior)은 유용한 귀납적 편향(inductive bias)을 제공하지만, 동시에 모델을 특정 모달리티에 묶어두는 제약이 됩니다.

본 논문에서는 **Perceiver**라는 모델을 소개합니다. 이 모델은 Transformer 구조를 기반으로 하여 입력 간 관계에 대한 구조적 가정을 거의 하지 않으며, 동시에 ConvNet처럼 수십만 개의 입력을 처리할 수 있도록 확장 가능합니다. Perceiver는 비대칭적 주의(attention) 메커니즘을 활용해, 입력을 반복적으로 압축된 잠재 표현(latent bottleneck)으로 정제(distill)함으로써 매우 큰 입력도 처리할 수 있도록 설계되었습니다.

우리는 이 아키텍처가 다양한 모달리티(이미지, 포인트 클라우드, 오디오, 비디오, 비디오+오디오)에 걸친 분류 작업에서 강력한 특화 모델들과 경쟁하거나 이를 능가함을 보여줍니다. Perceiver는 2D 컨볼루션을 사용하지 않고도 50,000개의 픽셀에 직접 주의를 기울임으로써 ImageNet에서 ResNet-50 및 ViT 수준의 성능을 달성합니다. 또한 AudioSet을 포함한 모든 모달리티에서도 경쟁력 있는 성능을 보입니다.

**키워드:**  
Perceiver, Transformer, Attention, Cross-attention, Image Transformer, Vision Transformer, 멀티모달(Multimodal), ImageNet, Permuted ImageNet, AudioSet, ModelNet

### 1. 서론 (Introduction)

초기 시각 처리에서의 공간적 지역성(spatial locality)과 같은 귀납적 편향(inductive bias)은 인지(perception) 모델의 학습 효율을 획기적으로 높이는 데 명백히 유용하며, 그 가치가 잘 알려져 있습니다. 하지만 점점 더 많은 대규모 데이터셋이 사용 가능해지고 있는 오늘날, 이러한 편향을 모델 안에 강하게 구조화된 형태로 주입하는 것이 과연 옳은 선택일까요? 아니면 가능한 한 많은 유연성을 확보하고, 데이터 자체가 말하도록 두는 것이 더 나은 접근일까요? (LeCun et al., 2015)

강한 구조적 사전(architectural prior)의 문제점 중 하나는 그것이 종종 **모달리티 특화(modality-specific)** 라는 점입니다. 예를 들어, 입력이 단일 이미지라고 가정하면 우리는 그 2차원 격자 구조(2D grid structure)를 이용해 2D 컨볼루션 연산에 기반한 효율적인 아키텍처를 설계할 수 있습니다. 하지만 입력이 스테레오 이미지 쌍으로 바뀌면, 두 센서의 픽셀을 어떻게 함께 처리할 것인지 결정해야 합니다. 예컨대, early fusion을 사용할지 late fusion을 사용할지(Karpathy et al., 2014), 또는 피처를 더할지(concatenate) 혹은 합할지(sum) 선택해야 합니다. 오디오로 전환하면 2D 격자 구조의 장점은 더 이상 명확하지 않으며, 대신 1D 컨볼루션이나 LSTM(Hochreiter & Schmidhuber, 1997; Graves et al., 2013)과 같은 다른 종류의 모델이 더 적합할 수 있습니다. 포인트 클라우드(point cloud)를 처리하고자 한다면(예: 라이다 센서를 탑재한 자율주행차), 저해상도의 고정 격자에 최적화된 기존 모델에 의존할 수 없습니다. 요컨대, 기존 도구를 사용하면 입력이 바뀔 때마다 아키텍처를 새로 설계해야 하는 문제가 있습니다.

![](/assets/images/posts/562/img.png)

**[그림 1 캡션]**  
Perceiver는 시각, 비디오, 오디오, 포인트 클라우드 및 멀티모달 입력 등 고차원 입력을 처리할 수 있도록 설계된 주의(attention) 기반 아키텍처입니다. 도메인 특화 가정을 하지 않으면서도 범용적인 확장성을 갖추고 있습니다. Perceiver는 고차원 입력 바이트 배열을 고정 차원의 잠재 병목(latent bottleneck)으로 투영하기 위해 교차 주의(cross-attention) 모듈을 사용합니다 (입력 인덱스 수 M이 잠재 인덱스 수 N보다 훨씬 큼). 이후 Transformer 스타일의 self-attention 블록을 잠재 공간에서 깊게 쌓아 입력을 처리합니다. Perceiver는 cross-attention과 latent self-attention을 교차 적용하여 입력 바이트 배열을 반복적으로 주의(attend)합니다.

본 논문에서는 Perceiver라는 모델을 소개합니다. 이 모델은 다양한 모달리티 구성(configurations)을 단일 Transformer 기반 아키텍처로 처리할 수 있도록 설계되었습니다. Transformer(Vaswani et al., 2017)는 입력에 대해 거의 가정을 하지 않으면서도 매우 유연한 구조적 블록이지만, 입력 수가 증가할수록 메모리와 연산 측면에서 **제곱 비율(quadratic)**로 확장되는 문제가 있습니다. 최근 연구에서는 이미지에 Transformer를 성공적으로 적용했지만, 대부분 픽셀의 격자 구조를 활용해 연산량을 줄였습니다. 예를 들어, 먼저 2D 컨볼루션을 적용하거나(Dosovitskiy et al., 2021; Touvron et al., 2020), 행과 열로 인자를 분해하거나(Ho et al., 2019; Child et al., 2019), 매우 강한 다운샘플링을 수행하는 방식(Chen et al., 2020a)입니다. 이에 반해, 우리는 입력 구성이 어떻게 되든 유연하게 처리할 수 있으면서도 고차원 입력을 효과적으로 다룰 수 있는 새로운 메커니즘을 제안합니다.

우리의 핵심 아이디어는 **입력이 반드시 통과해야 하는 작은 수의 잠재 유닛(latent unit)**을 도입해 **주의 병목(attention bottleneck)**을 형성하는 것입니다 (그림 1 참고). 이를 통해 Transformer의 전통적인 all-to-all attention에서 발생하는 제곱 확장을 제거하고, 네트워크 깊이를 입력 크기와 독립적으로 설계할 수 있어 매우 깊은 모델 구성이 가능해집니다. Perceiver는 입력에 반복적으로 주의를 기울임으로써, 제한된 용량을 가장 관련성 높은 입력에 집중시킬 수 있으며, 각 단계의 결과를 기반으로 정보를 점진적으로 축적합니다.

하지만 많은 모달리티에서는 **공간적/시간적 정보**가 핵심이며, 멀티모달 상황에서는 입력이 어떤 모달리티에서 온 것인지 구분하는 것도 매우 중요합니다. 명시적인 구조가 없는 우리의 아키텍처에서는 모든 입력 요소(예: 픽셀 하나하나나 오디오 샘플 등)에 **위치 및 모달리티 특화 특성(position- and modality-specific features)**을 연관시켜 이를 보완할 수 있습니다. 이들은 학습되거나, 고해상도 푸리에 특성(high-fidelity Fourier features; Mildenhall et al., 2020; Tancik et al., 2020; Vaswani et al., 2017)을 통해 구성할 수 있습니다. 이는 생물학적 신경망에서 특정 유닛의 활동을 의미적 또는 공간적 위치와 연관시키는 **지형 및 감각 간 지도(topographic and cross-sensory maps)** 전략과 유사한 방식입니다 (Kandel et al., 2012, 21장).

우리는 Perceiver가 다음과 같은 다양한 작업에서 강력한 모델들과 경쟁할 수 있음을 보여줍니다:

- **ImageNet 분류**에서 ResNet-50 및 ViT와 유사한 성능 달성
- **AudioSet 소리 이벤트 분류**에서 오디오, 비디오 또는 멀티모달 입력으로 경쟁력 있는 성능 달성
- **ModelNet-40 포인트 클라우드 분류**에서도 기존 접근법들과 비교해 우수한 성능 확보

### 2. 관련 연구 (Related Work)

![](/assets/images/posts/562/img_1.png)

**[그림 2 캡션]**  
우리는 Perceiver 아키텍처를 ImageNet(Deng et al., 2009)의 이미지(왼쪽), AudioSet(Gemmeke et al., 2017)의 비디오 및 오디오(단일 모달과 멀티모달 모두 고려, 중앙), 그리고 ModelNet40(Wu et al., 2015)의 3D 포인트 클라우드(오른쪽)에서 학습시켰습니다. 다양한 입력 데이터를 처리함에 있어 모델 구조를 본질적으로 변경할 필요가 없습니다.

**ConvNet과의 비교**  
ConvNet(Fukushima, 1980; LeCun et al., 1998; Cireşan et al., 2011; Krizhevsky et al., 2012)은 지난 10년 동안 인지(perceptual) 작업에서 뛰어난 성능과 확장성 덕분에 지배적인 아키텍처로 자리 잡아왔습니다. ConvNet은 2D 컨볼루션을 통해 가중치를 공유하고 각 유닛의 연산을 지역적인 2D 이웃으로 제한함으로써, 고해상도 이미지를 상대적으로 적은 파라미터와 연산량으로 처리할 수 있습니다. 그러나 앞서 언급했듯이, ConvNet은 다양한 신호를 결합할 때 유연성이 부족합니다. 이는 언어 분야에서 Transformer(Vaswani et al., 2017)와 같은 주의(attention) 기반 모델이 지배적인 것과 대조적입니다.

**효율적인 Attention 아키텍처**  
Transformer는 매우 유연한 구조를 갖지만, 입력 크기에 따라 모든 self-attention 레이어가 동일한 수의 입력을 가지므로 메모리와 연산 비용이 **제곱 비율(quadratic)**로 증가하는 단점이 있습니다. 그럼에도 불구하고, self-attention은 빠르게 시각 인지 분야로 확산되고 있으며, 이는 이미지(Bello et al., 2019; Cordonnier et al., 2020; Srinivas et al., 2021)와 비디오(Wang et al., 2018; Girdhar et al., 2019) 모델에서 부분적으로 적용되는 형태로 나타납니다.

입력이 너무 큰 도메인에서도 Transformer를 적용할 수 있도록, 입력 크기를 줄이는 다양한 전략이 제안되었습니다. 예를 들어, 입력을 다운샘플링하거나(Chen et al., 2020a), 사전 처리로 컨볼루션을 사용하는 방식(Wu et al., 2020)입니다. ViT (Vision Transformer; Dosovitskiy et al., 2021)는 이러한 전략을 따릅니다. 이 모델은 2D 컨볼루션 계층(해당 논문에서는 “평탄화된 패치의 선형 투영”이라 지칭)을 통해 입력을 약 200개 정도로 줄인 뒤, BERT(Devlin et al., 2019)에서 사용하는 class token과 함께 Transformer에 입력합니다. ViT는 ImageNet에서 인상적인 성능을 보이지만, 이러한 사전 처리 전략은 입력이 **격자 형태(grid-like)**를 가지는 이미지 기반 도메인으로 제한된다는 한계가 있습니다.

**Transformer 구조 내부의 효율 개선 연구**  
Transformer의 self-attention 모듈 내부를 개선하여 효율성을 높이려는 시도도 여럿 있었습니다 (자세한 내용은 부록 A 섹션 참고). 이 중 Perceiver와 가장 유사한 연구는 **Set Transformer (Lee et al., 2019)**입니다. Set Transformer는 cross-attention을 통해 대규모 입력 배열을 더 작은 배열로 투영하며, 이를 통해 연산량을 줄이거나 입력을 타깃 출력 형태로 변환할 수 있습니다(예: 입력 집합을 로짓으로 매핑). Perceiver 역시 이와 유사하게, **보조적인 저차원 배열을 활용한 cross-attention**을 통해 attention 복잡도를 입력 크기에 대해 **선형(linear)**으로 낮춥니다.

비슷한 맥락에서 Linformer(Wang et al., 2020b)는 cross-attention 없이도, key와 value를 입력보다 작은 크기의 배열로 투영하여 self-attention의 계산 복잡도를 선형으로 낮추는 방법을 제시합니다. 하지만 Perceiver는 단순히 선형 복잡도를 달성하는 것뿐 아니라, **네트워크의 깊이를 입력 크기와 분리(decouple)**한다는 점에서 차별점을 가집니다. 이는 단순한 확장성 이상의 효과를 만들어내며, 다양한 도메인에서 높은 성능을 위한 매우 깊은 아키텍처 구성을 가능하게 해줍니다 (자세한 내용은 3절에서 논의). Perceiver와 Set Transformer 및 관련 모델과의 관계는 부록 A에서 더 자세히 다룹니다.

**멀티모달 아키텍처**  
현재 멀티모달 처리 방식은 보통 각 모달리티별로 별도의 feature extractor를 사용하는 구조입니다(Kaiser et al., 2017; Arandjelovic & Zisserman, 2018; Wang et al., 2020c; Chen et al., 2020b; Alayrac et al., 2020; Lee et al., 2020; Xiao et al., 2020). 예를 들어, 오디오 스펙트로그램이나 원시 오디오 파형을 이미지와 단순히 결합(concatenate)한 뒤 ConvNet에 넣는 것은 일반적으로 합리적이지 않습니다.

이 접근법은 **언제 모달리티를 융합할지** 등 다양한 아키텍처 선택지를 필요로 하며, 각 애플리케이션마다 이를 재조정해야 하는 불편이 있습니다. 그 결과, 비전에서 최적화된 아키텍처는 모든 도메인에 적용될 수 없으며, 포인트 클라우드(Qi et al., 2017; Guo et al., 2020)처럼 특수한 도메인을 위해 별도의 모델이 개발되었습니다.

반면, **Perceiver는 처음부터 다양한 입력 모달리티를 매우 유연하게 처리할 수 있도록 설계**되었습니다. 특히 이미지나 오디오처럼 **대역폭이 높은 입력(high-bandwidth inputs)**도 별다른 구조 변경 없이 처리할 수 있습니다 (그림 2 참조).

### 3. 방법 (Methods)

#### 3.1 Perceiver 아키텍처

**개요 (Overview)**  
우리는 Perceiver 아키텍처를 두 가지 구성 요소로 설계합니다:  
(i) **크로스 어텐션 모듈(cross-attention module)** – 바이트 배열(예: 픽셀 배열)과 잠재 배열(latent array)을 입력받아 잠재 배열로 매핑  
(ii) **Transformer 타워(Transformer tower)** – 잠재 배열을 입력받아 또 다른 잠재 배열로 매핑

바이트 배열의 크기는 입력 데이터에 의해 결정되며 일반적으로 매우 큽니다 (예: 224 해상도의 ImageNet 이미지는 50,176개의 픽셀을 가짐). 반면 잠재 배열의 크기는 하이퍼파라미터로, 일반적으로 훨씬 작습니다 (예: ImageNet에서는 512개의 latent를 사용).

우리 모델은 cross-attention 모듈과 Transformer를 번갈아가며 적용합니다. 즉, 고차원 바이트 배열을 낮은 차원의 **어텐션 병목(attention bottleneck)**을 통해 투영한 후, 깊은 Transformer로 처리하고, 그 결과 표현을 이용해 다시 입력을 질의(query)하는 방식입니다. 이 모델은 잠재 위치(latent position)를 클러스터 중심으로 사용하여 **입력을 end-to-end 방식으로 군집화**하는 것으로도 볼 수 있으며, 이는 비대칭적인 cross-attention을 통해 수행됩니다.

Transformer 타워 간의 파라미터를 공유하고(선택적으로), 첫 번째를 제외한 cross-attention 모듈들 간에도 파라미터를 공유하기 때문에, 이 모델은 시간 차원이 아니라 깊이 차원에서 펼쳐진 **순환 신경망(RNN)**으로 해석될 수 있습니다. Perceiver의 모든 attention 모듈은 **비인과적(non-causal)**이며 마스크를 사용하지 않습니다. 아키텍처 구성은 그림 1에 나와 있습니다.

**cross-attention을 통한 제곱 복잡도의 제어**  
Perceiver는 입력 구조에 대한 가정이 거의 없으면서도 강력한 성능을 보여주는 **attention 기반** 아키텍처를 중심으로 설계되었습니다. 주요 과제는 매우 크고 일반적인 입력에서도 **attention 연산이 확장 가능하도록** 만드는 것입니다. Cross-attention과 Transformer 모듈은 모두 QKV (query-key-value) attention 구조에 기반합니다 (Graves et al., 2014; Weston et al., 2015; Bahdanau et al., 2015). QKV attention은 일반적으로 MLP를 사용하는 query, key, value 네트워크를 각각의 입력 요소에 적용해, 동일한 인덱스 차원(또는 시퀀스 길이) M을 유지하는 세 개의 배열을 생성합니다.

Transformer의 주요 병목은 QKV self-attention이 입력 차원 M에 대해 **O(M²)** 복잡도를 가진다는 점입니다. 예를 들어 224×224 이미지에서는 M = 50,176입니다. 오디오의 경우도 유사하게, 1초 길이의 오디오는 약 50,000개의 샘플로 구성되므로 마찬가지 문제가 발생합니다. 멀티모달 입력에서는 이 문제가 더욱 심각해집니다.

이 문제를 피하기 위해 기존 연구들은 QKV attention을 직접 입력 픽셀 배열에 적용하지 않고(자세한 내용은 2절과 부록 A 참조), 사전 처리 등을 사용합니다. 반면, 우리는 attention 연산에 **비대칭성(asymmetry)**을 도입함으로써 이를 해결합니다.

보다 구체적으로,

![](/assets/images/posts/562/img_2.png)

일 때, 기본 QKV 연산 softmax(QKᵀ)V는 두 번의 큰 행렬 곱셈을 포함하므로 **O(M²)** 복잡도를 갖습니다. (채널 차원 C와 D는 M에 비해 작기 때문에 무시합니다.)

이를 해결하기 위해 우리는 **Q를 학습된 잠재 배열(latent array)**로부터 투영하고, **K와 V는 입력 바이트 배열에서 투영**하도록 비대칭 구조를 도입합니다. 잠재 배열의 인덱스 차원 N은 M보다 훨씬 작으며, 하이퍼파라미터입니다. 이렇게 하면 cross-attention 연산의 복잡도는 **O(MN)**이 됩니다.

**잠재 Transformer를 통한 깊이 분리(uncoupling)**  
cross-attention 모듈의 출력은 Q의 입력과 동일한 형태를 가집니다. 즉, cross-attention은 **병목 구조(bottleneck)**를 유도합니다. 우리는 이 병목 구조를 활용하여 **잠재 공간(latent space)**에서 깊고 표현력 높은 Transformer를 구성합니다. 이 경우 연산 비용은 **O(N²)**로 매우 저렴해집니다.

이 설계를 통해 Perceiver는 입력 크기와 무관하게 매우 깊은 Transformer를 사용할 수 있게 되며, 이는 domain-specific한 가정 없이도 가능합니다. 바이트 기반 Transformer는 L 레이어일 때 **O(LM²)**의 복잡도를 가지지만, latent 기반 Transformer는 **O(LN²)** 복잡도를 가지므로, N ≪ M인 상황에서는 매우 유리합니다.

이 결과로 Perceiver의 전체 복잡도는 **O(MN + LN²)**가 되며, 이는 중요한 설계 포인트입니다. 입력 크기와 네트워크 깊이를 분리함으로써, **입력 크기와 무관하게 더 많은 Transformer 블록을 추가할 수 있게** 되어 대규모 데이터에 적합한 모델 구성이 가능해집니다. 예를 들어, ImageNet 실험에서는 48개의 latent Transformer 블록을 사용하였으며, 이는 입력 크기와 깊이가 결합되어 있는 구조에서는 현실적으로 불가능합니다 (표 5 참고).

우리의 latent Transformer는 GPT-2 아키텍처(Radford et al., 2019)를 기반으로 하며, 이는 Transformer의 디코더(Vaswani et al., 2017)를 기반으로 합니다. 실험에서는 N ≤ 1024의 값을 사용하였으며, 이는 자연어 처리에서 사용되는 모델들과 유사한 입력 크기입니다. 잠재 배열은 **학습된 위치 인코딩(position encoding)**으로 초기화됩니다 (Gehring et al., 2017, 자세한 내용은 부록 C 참고).

**반복적 cross-attention 및 weight sharing**  
잠재 배열의 크기를 통해 픽셀을 직접 모델링하고 깊은 Transformer를 구성할 수 있지만, 병목 구조의 강도는 입력 신호에서 필요한 모든 정보를 완전히 포착하는 데 제한이 될 수 있습니다. 이를 완화하기 위해 Perceiver는 **여러 개의 cross-attention 레이어를 반복적으로 구성**할 수 있습니다. 이는 잠재 배열이 입력 이미지에서 필요한 정보를 점진적으로 추출할 수 있도록 합니다.

이 방식은 정보가 많은 cross-attention(연산량 많음)과 반복적인 latent self-attention(연산량 적음) 간의 **균형 조정**을 가능하게 합니다. 부록 표 6에서 보이듯이, cross-attention 레이어를 더 많이 사용할수록 성능은 향상되지만, 입력 크기에 선형적으로 비례하는 연산량이 증가합니다.

마지막으로, 반복적인 구조 덕분에 cross-attention 및 latent Transformer 블록 간에 **가중치 공유(weight sharing)**를 통해 파라미터 효율을 크게 높일 수 있습니다. cross-attend가 하나뿐일 경우, latent self-attention 블록들만 공유해도 충분합니다. ImageNet 실험에서는 가중치 공유를 통해 파라미터 수를 약 10배 줄였으며, 이는 과적합을 줄이고 검증 성능을 향상시켰습니다.

결과적으로, Perceiver는 **cross-attention 입력 투영**, **병목화된 잠재 표현**, 그리고 **Transformer 기반 반복 구조**를 갖춘 RNN과 같은 형태의 모델로 해석될 수 있습니다. 이러한 가중치 공유 기법은 기존 Transformer 연구(Dehghani et al., 2019; Lan et al., 2020)에서도 유사한 목적으로 활용된 바 있습니다.

---

![](/assets/images/posts/562/img_3.png)

![](/assets/images/posts/562/img_4.png)

50,176명의 사람들이 있고, 512명의 요약 담당자(latents)가 각각 중요한 사람들 이야기를 들은 다음 요약하는 방식입니다.  
사람 수(입력 크기)는 줄지 않지만, 중요한 정보는 요약자들이 담아서 **소수의 벡터로 압축**하는 구조인 거죠.

Cross-Attention 단계에서 줄어듭니다.

```
(입력) Byte array (M×C)   ← M = 224×224 = 50,176
            ↓
      Cross Attention (Q=latent, K,V=input)    ← 바로 이 단계가 복잡도 O(MN)
            ↓
     Latent Transformer (self-attention)       ← 복잡도 O(N²), N ≪ M
            ↓
      Cross Attention (다시 입력에서 정보 뽑음)
            ↓
     Latent Transformer
            ↓
            ...
            ↓
      평균화 (Average)
            ↓
          Logits
```

![](/assets/images/posts/562/img_5.png)

![](/assets/images/posts/562/img_6.png)

![](/assets/images/posts/562/img_7.png)

![](/assets/images/posts/562/img_8.png)

![](/assets/images/posts/562/img_9.png)

![](/assets/images/posts/562/img_10.png)

![](/assets/images/posts/562/img_11.png)

![](/assets/images/posts/562/img_12.png)

![](/assets/images/posts/562/img_13.png)

![](/assets/images/posts/562/img_14.png)

---
