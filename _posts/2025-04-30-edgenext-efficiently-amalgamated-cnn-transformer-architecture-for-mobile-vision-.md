---
title: "EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications"
date: 2025-04-30 12:25:06
categories:
  - 인공지능
tags:
  - edgenext
---

<https://arxiv.org/abs/2206.10589>

[EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

**초록**  
†  
\*동등 기여

점점 더 높은 정확도를 달성하기 위해, 대규모이면서 복잡한 신경망이 주로 개발되고 있습니다. 하지만 이러한 모델들은 높은 연산 자원을 요구하기 때문에 엣지(Edge) 디바이스에 배포하기 어렵습니다. 여러 응용 분야에서 유용하게 활용될 수 있는, 자원 효율적인 범용 네트워크를 구축하는 것은 매우 중요한 과제입니다. 본 연구에서는 CNN과 Transformer 모델 각각의 강점을 효과적으로 결합하고자 하여, 새로운 효율적인 하이브리드 아키텍처인 **EdgeNeXt**를 제안합니다.  
특히 EdgeNeXt에서는, 입력 텐서를 여러 채널 그룹으로 나눈 후, 채널 차원에 대해 깊이별 합성곱(Depth-wise Convolution)과 자기 주의(Self-attention)를 함께 활용하여 수용 영역(Receptive Field)을 암묵적으로 확장하고 다중 스케일 특성(Multi-scale Features)을 인코딩하는 **분할 깊이별 전치 주의(Split Depth-wise Transpose Attention, STDA) 인코더**를 도입합니다.  
분류(Classification), 탐지(Detection), 분할(Segmentation) 작업에 대해 광범위한 실험을 수행한 결과, 제안한 접근 방식이 상대적으로 적은 연산량으로도 최신(state-of-the-art) 방법들을 능가함을 확인하였습니다.  
1.3M 파라미터를 가진 EdgeNeXt 모델은 ImageNet-1K에서 71.2%의 Top-1 정확도를 달성하여, MobileViT 대비 절대 2.2% 높은 성능을 기록했으며, 연산량(FLOPs)은 28% 감소했습니다. 또한, 5.6M 파라미터를 가진 EdgeNeXt 모델은 ImageNet-1K에서 79.4%의 Top-1 정확도를 달성했습니다.  
코드와 모델은 <https://t.ly/_Vu9>에서 확인할 수 있습니다.

**키워드**: 엣지 디바이스(Edge devices), 하이브리드 모델(Hybrid model), 합성곱 신경망(Convolutional neural network), 자기 주의(Self-attention), 트랜스포머(Transformers), 이미지 분류(Image classification), 객체 탐지(Object detection), 분할(Segmentation)

**1. 서론**  
합성곱 신경망(Convolutional Neural Networks, CNNs)과 최근 도입된 비전 트랜스포머(Vision Transformers, ViTs)는 객체 인식, 탐지, 분할과 같은 여러 주요 컴퓨터 비전 과제에서 최신 성능(State-of-the-Art, SOTA)을 크게 발전시켜 왔습니다【37, 20】.  
일반적인 추세는 점점 더 높은 정확도를 추구하기 위해 네트워크 아키텍처를 더욱 깊고 복잡하게 만드는 것입니다. 그러나 대부분의 기존 CNN 및 ViT 기반 아키텍처는 정확도를 높이는 데 집중하는 반면, **모델 크기와 속도와 같은 계산 효율성(computational efficiency)** 측면은 간과하고 있습니다. 이는 모바일 플랫폼과 같이 자원이 제한된 디바이스에서 운용할 때 매우 중요한 요소입니다. 실제 로봇 공학이나 자율주행차와 같은 여러 응용 분야에서는, **정확도가 높으면서도 낮은 지연 시간**을 갖는 인식 프로세스가 모바일 플랫폼 상에서 요구됩니다.

대부분의 기존 접근법은 자원이 제한된 모바일 플랫폼에서 속도와 정확도 간의 균형을 이루기 위해, 신중하게 설계된 효율적인 합성곱 변형(variants of convolutions)을 활용합니다【19, 28, 36】. 이외에도 일부 연구【16, 39】는 하드웨어 인지 신경망 아키텍처 탐색(Hardware-aware Neural Architecture Search, NAS)을 통해 모바일 디바이스용으로 낮은 지연 시간과 높은 정확도를 동시에 갖춘 모델을 구축하고자 합니다.  
이러한 경량 CNN들은 학습이 용이하고 이미지의 로컬 세부 정보를 효율적으로 인코딩할 수 있다는 장점이 있지만, **픽셀 간의 전역 상호작용(Global Interaction)을 명시적으로 모델링하지는 못하는** 한계가 있습니다.

![](/assets/images/posts/550/img.png)

**참고**  
**그림 1:** 본 논문에서 제안하는 EdgeNeXt 모델과 최신 비전 트랜스포머(ViTs) 및 하이브리드 아키텍처 설계들과의 비교.  
x축은 곱셈-덧셈(MAdd) 연산량을, y축은 ImageNet-1K Top-1 분류 정확도를 나타냅니다. 각 점에는 해당 모델의 파라미터 수가 표시되어 있습니다.  
제안한 EdgeNeXt는 최근 접근법들보다 **더 나은 연산량 대비 정확도** 트레이드오프를 보여줍니다.

비전 트랜스포머(ViTs)【8】는 **자기 주의(Self-attention)** 메커니즘을 도입함으로써 전역 상호작용을 명시적으로 모델링할 수 있도록 했지만, 이로 인해 **추론 속도가 느려지는** 단점이 생겼습니다【24】. 이는 모바일 비전 애플리케이션을 위한 경량 ViT 변형을 설계할 때 중요한 도전 과제가 됩니다.

대부분 기존 연구들은 CNN 기반 설계를 통해 효율적인 모델을 개발해 왔습니다. 하지만 CNN의 합성곱 연산은 두 가지 주요 한계를 가지고 있습니다:  
(1) **로컬 수용 영역(Local Receptive Field)** 만을 다루기 때문에 전역 문맥(Global Context)을 모델링할 수 없습니다.  
(2) 학습된 가중치가 추론 시점에서는 고정되어 있어, 입력 콘텐츠에 따라 적응하는 데 유연성이 부족합니다.  
트랜스포머를 활용하면 이러한 문제를 완화할 수 있으나, 트랜스포머는 대개 **연산량이 매우 크다는** 단점이 있습니다.  
최근 일부 연구【46, 29】에서는 CNN과 ViT의 강점을 결합하여 모바일 비전 과제를 위한 경량 아키텍처를 설계하는 시도를 했지만, 이들 접근법은 주로 **파라미터 최적화에 집중**하면서도, **곱셈-덧셈 연산(MAdds)이 많아** 고속 추론에는 적합하지 않은 문제를 안고 있습니다. 특히 자기 주의 블록은 입력 크기에 대해 **복잡도가 이차(quadratic)**이기 때문에【29】, 네트워크 내에 여러 개의 주의 블록이 있을 경우 이 문제는 더욱 심각해집니다.  
따라서 우리는 **모델 크기, 파라미터 수, MAdds 모두가 작아야** 자원 제한 디바이스에 적합한 CNN과 ViT의 통합 아키텍처를 설계할 수 있다고 주장합니다(그림 1 참조).

**공헌(Contributions)**  
본 논문에서는, 모델 크기, 파라미터 수, MAdds 측면에서 효율적이면서도 모바일 비전 과제에서 높은 정확도를 보이는 새로운 경량 아키텍처 **EdgeNeXt**를 제안합니다.  
구체적으로, CNN의 제한된 수용 영역 문제를 해결하기 위해 **분할 깊이별 전치 주의(Split Depth-wise Transpose Attention, SDTA) 인코더**를 도입하여, 추가적인 파라미터나 MAdds 증가 없이 로컬 및 전역 표현(Local and Global Representations)을 효과적으로 학습할 수 있도록 합니다.

제안한 EdgeNeXt 아키텍처는 이미지 분류, 객체 탐지, 시맨틱 분할 등 다양한 과제에서, 최신 모바일 네트워크들 대비 **정확도와 지연 시간(latency)** 양 측면 모두에서 우수한 성능을 보입니다.

- 5.6M 파라미터와 1.3G MAdds를 가진 EdgeNeXt 백본은 ImageNet-1K에서 **79.4% Top-1 분류 정확도**를 달성했으며, 이는 최근에 소개된 MobileViT【29】보다 높은 정확도를 기록하면서도 **MAdds는 35% 적게** 요구합니다.
- 객체 탐지 및 시맨틱 분할 과제에서도, EdgeNeXt는 기존에 발표된 모든 경량 모델들과 비교해 **더 적은 MAdds**와 **비슷한 파라미터 수**로 **더 높은 mAP(탐지 성능) 및 mIOU(분할 성능)** 를 달성했습니다.

**2. 관련 연구**  
최근 몇 년간, 모바일 비전 작업을 위한 **경량 하드웨어 효율적인 합성곱 신경망(CNN)** 설계가 활발히 연구되어 왔습니다. 현재의 주요 방법들은 **저전력 엣지 디바이스에 적합한 효율적인 합성곱 구조** 설계에 집중하고 있습니다【19, 17】.  
이 중에서도 **MobileNet**【17】은 **Depth-wise Separable Convolutions**【5】을 사용하는 가장 널리 쓰이는 아키텍처입니다.  
반면, **ShuffleNet**【47】은 **채널 셔플링(Channel Shuffling)**과 **저비용 그룹 합성곱(Group Convolutions)**을 사용합니다.  
**MobileNetV2**【36】는 **선형 병목 구조를 갖는 Inverted Residual Block**을 도입하여 다양한 비전 작업에서 유망한 성능을 보여줍니다.  
**ESPNetv2**【31】는 네트워크 복잡도 증가 없이 수용 영역(Receptive Field)을 확장하기 위해 **Depth-wise Dilated Convolutions**를 활용합니다.  
또한, **하드웨어 인지형 신경망 아키텍처 탐색(Hardware-aware NAS)**도 모바일 디바이스에서 **속도와 정확도 간의 균형**을 찾기 위한 방법으로 활발히 연구되었습니다【16, 39】.  
이러한 CNN 기반 경량 모델들은 모바일 디바이스에서 학습과 추론 속도가 빠르다는 장점이 있지만, **픽셀 간 전역 상호작용(Global Interaction)을 모델링하지 못한다는 한계**가 있어 정확도 측면에서 제약을 가집니다.

최근 **Dosovitskiy et al.**【8】는 **자기 주의(Self-Attention)** 메커니즘【41】을 기반으로 한 **Vision Transformer(ViT)** 아키텍처를 비전 작업에 도입했습니다. 이 아키텍처는 **대규모 사전 학습 데이터셋(JFT-300M 등)**, **광범위한 데이터 증강**, **긴 학습 스케줄**을 통해 경쟁력 있는 성능을 달성하였습니다.  
이후 **DeiT**【40】는 **지식 증류 토큰(Distillation Token)**을 추가하고, **ImageNet-1K**【35】 데이터셋만으로 학습이 가능한 구조를 제안했습니다.  
이후 다양한 **ViT 변형 및 하이브리드 아키텍처**들이 등장하였으며, **이미지 도메인에 특화된 유도 편향(inductive bias)**를 ViT에 추가함으로써 여러 비전 작업에서 성능 향상을 달성하였습니다【38, 9, 42, 48, 11】.

ViT 모델은 다양한 시각 인식 작업에서 **경쟁력 있는 결과**를 보여주고 있습니다【8, 24】.  
하지만 **다중 헤드 자기 주의(Multi-head Self-Attention, MHA)**의 높은 연산 비용 때문에, **자원이 제한된 엣지 디바이스에서는 배포가 어렵습니다**.  
이에 따라 최근에는 CNN과 Transformer의 장점을 결합한 **모바일 비전 작업을 위한 경량 하이브리드 네트워크** 설계가 시도되고 있습니다.  
예를 들어, **MobileFormer**【4】는 **MobileNetV2**【36】와 **ViT**【8】의 병렬 브랜치를 사용하고, **로컬-글로벌 상호작용**을 위한 브릿지 구조를 도입합니다.  
**Mehta et al.**【29】는 Transformer를 합성곱처럼 간주하고, **로컬-글로벌 문맥 결합(Local-global image context fusion)**을 위한 **MobileViT 블록**을 제안합니다.  
이 접근은 **비슷한 파라미터 규모 하에서 기존 경량 CNN 및 ViT를 능가하는 성능**을 이미지 분류 과제에서 달성하였습니다.

하지만 **MobileViT**【29】는 파라미터 및 지연 시간 최적화에 중점을 두었음에도, **MHA는 여전히 효율성의 주요 병목**입니다.  
특히, **입력 크기에 대해 이차 복잡도(quadratic complexity)**를 갖는 MHA 구조는, **MobileViT-S 모델 내에 9개의 Attention 블록**이 존재함에 따라 MAdds 및 추론 시간 측면에서 큰 부담을 줍니다.

이에 본 논문에서는, **파라미터 수와 MAdds 측면 모두 효율적이며**, 모바일 비전 작업에서 **높은 정확도**를 달성할 수 있는 새로운 경량 아키텍처를 설계하고자 합니다.  
제안한 아키텍처인 **EdgeNeXt**는 최근 소개된 CNN 기반 방법인 **ConvNeXt**【25】에 기반합니다. ConvNeXt는 **ResNet**【14】 아키텍처를 **ViT의 설계 철학에 맞게 현대화**한 모델입니다.

EdgeNeXt에서는 **Depth-wise Convolution**과 **Transpose Attention**을 **적응형 커널 크기**와 결합한 **SDTA 블록**을 도입하여, 효율적인 방식으로 로컬-글로벌 정보를 학습합니다.  
이로써, 정확도와 속도 간의 최적의 트레이드오프를 달성합니다.

**3. EdgeNeXt**  
본 연구의 주요 목적은 **저전력 엣지 디바이스를 위한 ViT와 CNN의 장점을 효과적으로 결합한 경량 하이브리드 설계**를 개발하는 것입니다.  
ViT 기반 모델(예: MobileViT【29】)의 연산 부담은 주로 **자기 주의(Self-Attention)** 연산에서 발생합니다.  
우리 모델의 Attention 블록은 MobileViT와 달리 입력의 공간 차원이 아니라 **채널 차원에 대해 Attention을 적용**하며, 이에 따라 연산 복잡도는 공간 차원에 대해 **선형 복잡도**인 **?(N·d²)**를 가집니다. 여기서 **N**은 패치 수, **d**는 특성(채널) 차원입니다.

또한, **우리는 MobileViT에서 사용된 9개의 Attention 블록보다 적은 수(3개)**의 Attention 블록만으로도 더 나은 성능을 달성할 수 있음을 보입니다.  
이러한 방식으로 제안하는 프레임워크는 **적은 MAdds로 전역 표현(Global Representations)을 모델링할 수 있으며**, 이는 엣지 디바이스에서 **낮은 지연 시간 추론(Low-latency inference)**을 보장하는 데 필수적인 조건입니다.

제안 아키텍처를 정당화하기 위해, 다음 두 가지 핵심 특성을 소개합니다:

**a) 전역 정보를 효율적으로 인코딩**  
Self-Attention의 핵심 장점은 **전역 표현(Global Representations)**을 학습하는 능력이며, 이는 비전 작업에서 매우 중요합니다.  
우리는 이러한 장점을 효율적으로 활용하기 위해, **공간 차원 대신 채널 차원에 대해 Attention을 적용하는 Cross-Covariance Attention**을 사용합니다.  
이로 인해 Self-Attention의 복잡도는 **토큰 수에 대해 이차(quadratic)**에서 **선형(linear)**으로 감소하며, 동시에 전역 정보를 **암묵적으로 효과적으로 인코딩**할 수 있습니다.

**b) 적응형 커널 크기(Adaptive Kernel Sizes)**  
큰 커널을 사용하는 합성곱은 수용 영역을 넓히는 데 유용하지만, **파라미터 수와 FLOPs가 커널 크기에 따라 이차적으로 증가**하기 때문에 매우 비용이 큽니다.  
이러한 큰 커널을 네트워크 전체에 적용하는 것은 **비효율적이고 최적이 아닙니다**.  
이를 해결하기 위해, **다단계 네트워크 계층 내에서 서로 다른 커널 크기를 사용하는 적응형 메커니즘**을 제안합니다.

CNN의 계층 구조에서 영감을 받아, 네트워크의 **초기 단계에서는 작은 커널**, **후기 단계에서는 큰 커널**을 사용하는 구조를 설계합니다.  
초기 단계는 일반적으로 **저수준 특징(Low-level features)**을 추출하며, 작은 커널이 이에 적합합니다.  
반면, 후기 단계에서는 **고수준 특징(High-level features)**을 포착하기 위해 **큰 커널**이 필요합니다【45】.

다음으로 아키텍처의 세부사항을 설명합니다.

![](/assets/images/posts/550/img_1.png)

**참고**  
**그림 2:**  
**상단:** 전체 프레임워크는 단계별(stage-wise) 구조를 따릅니다.

- **1단계:** 입력 이미지를 **4×4 스트라이드 합성곱(strided convolution)**을 통해 **1/4 해상도**로 다운샘플링하고, 이어서 **3개의 3×3 합성곱 인코더**를 적용합니다.
- **2~4단계:** 각 단계의 시작 부분에서 **2×2 스트라이드 합성곱**으로 다운샘플링하며, 이어서 **N×N 합성곱**과 **SDTA(Split Depth-wise Transpose Attention) 인코더**를 사용합니다.

**하단:** 왼쪽은 Conv. 인코더, 오른쪽은 SDTA 인코더의 설계를 보여줍니다.

- **Conv. 인코더:** 공간 혼합을 위해 **N×N Depth-wise Convolution**, 채널 혼합을 위해 **두 개의 Pointwise Convolution**을 사용합니다.
- **SDTA 인코더:** 입력 텐서를 **B개의 채널 그룹**으로 나누고, 각 그룹에 대해 **3×3 Depth-wise Convolution**을 적용해 다중 스케일 공간 혼합을 수행합니다.
- 브랜치 간 **스킵 연결(Skip Connections)**은 네트워크의 수용 영역을 확장합니다.
- **B₃, B₄ 브랜치**는 각각 3단계, 4단계에서 점진적으로 활성화되어, 깊은 계층일수록 더 넓은 수용 영역을 갖게 합니다.
- SDTA 블록 내부에서는 **Transpose Attention** 후 **경량 MLP**를 적용하며, 이는 **입력 이미지에 대해 선형 복잡도**를 가집니다.

### **전체 아키텍처(Overall Architecture)**

그림 2는 제안된 **EdgeNeXt** 아키텍처의 전체 개요를 보여줍니다. 주요 구성 요소는 두 가지입니다:  
(1) **적응형 N×N 합성곱 인코더(Conv. encoder)**  
(2) **분할 깊이별 전치 주의(SDTA) 인코더**

EdgeNeXt는 **ConvNeXt**【25】의 설계 원칙을 기반으로 하며, 네트워크를 네 개의 단계로 나누어 **4개의 스케일에서 계층적 특징(Hierarchical Features)**을 추출합니다.

입력 이미지(크기 H×W×3)는 네트워크의 시작에서 **패치화(Patchify) 스템 레이어**를 거치며, 이는 **4×4 비중첩 합성곱(Non-overlapping Convolution)**과 **LayerNorm**으로 구성되어,  
**(H/4)×(W/4)×C₁** 크기의 특징 맵을 생성합니다. 이후 **3×3 Conv. 인코더**를 통해 로컬 특징을 추출합니다.

**두 번째 단계(stage 2)**는 **2×2 스트라이드 합성곱**으로 구현된 다운샘플링 레이어로 시작하며, 이는 공간 해상도를 절반으로 줄이고 채널 수를 늘려  
**(H/8)×(W/8)×C₂** 크기의 특징 맵을 생성합니다. 이어서 **두 개의 연속된 5×5 Conv. 인코더**가 적용됩니다.

**Positional Encoding (PE)**은 **두 번째 단계에서 SDTA 블록 전에만 한 번 추가**됩니다.  
우리는 **객체 탐지 및 분할과 같은 조밀한 예측(dense prediction)** 작업에서 PE가 민감하게 작용함을 관찰하였고, 모든 단계에 PE를 추가하면 네트워크 지연 시간(latency)이 증가함을 확인하였습니다.  
따라서 **공간 위치 정보를 인코딩하기 위해 단 한 번만 PE를 적용**합니다.

그 후 특징 맵은 세 번째 및 네 번째 단계로 전달되며, 각각  
**(H/16)×(W/16)×C₃**,  
**(H/32)×(W/32)×C₄**  
크기의 특징을 생성합니다.

### **합성곱 인코더(Convolution Encoder)**

이 블록은 **적응형 커널 크기를 사용하는 Depth-wise Separable Convolution**으로 구성됩니다.  
다음의 두 레이어로 정의됩니다:

1. **적응형 N×N Depth-wise Convolution**
   - 단계별로 커널 크기를 다르게 적용합니다:  
     1단계: 3×3, 2단계: 5×5, 3단계: 7×7, 4단계: 9×9
2. **두 개의 Point-wise Convolution**
   - 지역 표현(Local Representation)을 강화하며,
   - **Layer Normalization (LN)**【2】과 **GELU 활성화 함수**【15】를 통해 **비선형 특징 매핑**을 수행합니다.
   - 마지막으로 **스킵 연결(Skip Connection)**을 추가하여 네트워크 계층 간 정보 흐름을 강화합니다.

이 블록은 전반적으로 ConvNeXt 블록과 유사하나, **커널 크기가 단계에 따라 동적으로 변화**한다는 차이점이 있습니다.  
실험 결과(Table 8 참조), **Conv. 인코더에서의 적응형 커널 크기가 정적 커널 크기보다 성능이 우수함**을 확인했습니다.

Conv. 인코더는 다음과 같은 수식으로 표현됩니다:

**xᵢ₊₁ = xᵢ + LinearG(Linear(LN(Dw(xᵢ))))** … (1)

여기서:

- **xᵢ**: 입력 특징 맵 (크기 H×W×C)
- **Dw**: k×k Depth-wise Convolution
- **LN**: 정규화 레이어 (LayerNorm)
- **Linear**: Point-wise Convolution
- **LinearG**: GELU를 뒤따르는 Point-wise Convolution
- **xᵢ₊₁**: Conv. 인코더의 출력 특징 맵

### **SDTA 인코더**

제안하는 **분할 깊이별 전치 주의(Split Depth-wise Transpose Attention, SDTA)** 인코더는 두 가지 주요 구성 요소로 이루어져 있습니다:

1. 첫 번째 구성 요소는 입력 이미지 내의 다양한 공간 수준(spatial levels)을 인코딩함으로써 **적응형 다중 스케일 특징 표현(adaptive multi-scale feature representation)**을 학습합니다.
2. 두 번째 구성 요소는 **전역 이미지 표현(global image representations)**을 **암묵적으로 인코딩**합니다.

SDTA 인코더의 첫 번째 부분은 **Res2Net**【12】에서 영감을 받았으며, **계층적 표현(hierarchical representation)**을 하나의 블록으로 통합하는 **다중 스케일 처리 방식(multi-scale processing)**을 채택합니다.  
이로 인해 출력 특징 표현의 공간 수용 영역(spatial receptive field)이 보다 **유연하고 적응적으로 확장**됩니다.

Res2Net과는 달리, SDTA 인코더의 첫 번째 블록은 **1×1 포인트와이즈 합성곱(pointwise convolution)**을 사용하지 않음으로써, **파라미터 수와 MAdds를 제한한 경량 네트워크 구조**를 유지합니다.  
또한, **각 단계(stage)별로 적응형으로 하위 집합(subset)의 수를 설정**하여, 유연하고 효과적인 특징 인코딩이 가능하도록 합니다.

우리의 SDTA 인코더에서는 입력 텐서 **H×W×C**를 **s개의 하위 집합(subset)**으로 나누며, 각 subset은 **?ᵢ**로 표기되고, 공간 크기는 동일하지만 채널 수는 **C/s**입니다. 여기서 **i ∈ {1, 2, …, s}**, **C**는 전체 채널 수입니다.

- 첫 번째 subset을 제외한 나머지 subset들은 모두 **3×3 Depth-wise Convolution (?ᵢ)**에 통과되며, 결과는 **?ᵢ**로 표기됩니다.
- 이때, **?ᵢ**는 이전 출력인 **?ᵢ₋₁**를 현재 입력 **?ᵢ**에 더한 뒤 convolution을 수행합니다.
- subset의 수 **s**는 현재 단계 번호 **t ∈ {2, 3, 4}**에 따라 **동적으로 조정(adaptive)**됩니다.

위 내용을 수식으로 표현하면 다음과 같습니다:

![](/assets/images/posts/550/img_2.png)

**각 Depth-wise 연산 ?ᵢ**는, 그림 2의 SDTA 인코더에서 보이듯, **이전 모든 split {?ⱼ, j ≤ i}의 특징 맵 출력**을 받아서 처리합니다.

앞서 언급한 바와 같이, **트랜스포머의 Self-Attention 레이어는 연산량(MAdds)과 지연(latency)이 크기 때문에 엣지 디바이스에서 비전 작업에 적용하기엔 부담이 큽니다**.  
이 문제를 해결하고 전역 문맥(Global Context)을 효율적으로 인코딩하기 위해, 우리는 SDTA 인코더에서 **Transpose된 Query와 Key 기반의 Attention 특징 맵(transposed query and key attention feature maps)**을 사용합니다【1】.

이 연산은 **공간 차원(spatial dimension)**이 아닌 **채널 차원(channel dimension)**을 따라 **다중 헤드 자기 주의(MSA)**의 내적(dot-product)을 수행함으로써 **선형 복잡도(linear complexity)**를 가집니다.  
이를 통해 채널 간 **교차 공분산(cross-covariance)**을 계산할 수 있으며, 이는 **전역 표현(global representations)**에 대한 **암묵적인 정보를 담은 attention 특징 맵**을 생성합니다.

정규화된 텐서 **?** (크기: **H×W×C**)가 주어졌을 때, 세 개의 선형 레이어를 사용하여 **Query (?)**, **Key (?)**, **Value (?)**를 다음과 같이 계산합니다:

- ? = ?\_Q × ?
- ? = ?\_K × ?
- ? = ?\_V × ?

여기서 결과는 **(H·W)×C** 차원의 행렬이 되며, **?\_Q, ?\_K, ?\_V**는 각각의 투영(projection) 가중치입니다.

이후 학습 안정성을 위해 **?와 ?에 L2 정규화**를 적용한 뒤 **교차 공분산 기반 Attention**을 계산합니다.  
기존 Self-Attention에서는 **? ⋅ ?ᵀ**을 공간 차원 기준으로 계산하지만 (즉, **(H·W × C) ⋅ (C × H·W)**),  
우리는 이를 채널 차원 기준으로 전치하여 **?ᵀ ⋅ ?** (즉, **(C × H·W) ⋅ (H·W × C)**)로 계산합니다.

그 결과 **C×C** 차원의 softmax로 정규화된 Attention Score 행렬이 생성됩니다.  
최종 Attention 맵은 이 score에 **?를 곱하고 합산(summation)**함으로써 얻어집니다.

Transpose Attention 연산은 다음과 같이 표현할 수 있습니다:

![](/assets/images/posts/550/img_3.png)

여기서 **?**는 입력 특징 맵, **?̂**는 출력 특징 맵입니다.

이후, **두 개의 1×1 Pointwise Convolution 레이어**, **LayerNorm (LN)**, **GELU 활성화 함수**를 통해 **비선형 특징**을 생성합니다.

**표 1**에는 각 계층의 입력 크기와 함께, Extra-extra Small (XXS), Extra Small (XS), Small (S) 모델들에 대한 Conv. 및 STDA 인코더의 순서 및 더 많은 설계 세부 사항이 정리되어 있습니다.

**표 1: EdgeNeXt 아키텍처**  
각 모델의 계층(layer)에 대한 설명을 출력 크기(output size), 커널 크기(kernel size), 출력 채널 수(output channels), 반복 횟수 **n**과 함께 제시하며, 전체 **MAdds**와 **파라미터 수**도 포함합니다.

**Small, Extra-Small, Extra-Extra Small** 모델의 출력 채널 수는 각각 **MobileViT 대응 모델과 파라미터 수가 유사하도록 조정**하였습니다.

또한, Conv. 인코더에는 **적응형 커널 크기(adaptive kernel sizes)**를 사용하여 모델의 복잡도를 줄이면서도 다양한 수준의 특징(feature level)을 포착할 수 있도록 하였습니다.

마지막 단계(stage)의 출력 크기는 **9×9 필터 적용이 가능하도록 패딩 처리(padding)**되었습니다.

![](/assets/images/posts/550/img_4.png)

### **4. 실험(Experiments)**

이 섹션에서는 **ImageNet-1K 분류**, **COCO 객체 탐지**, **Pascal VOC 분할** 벤치마크에서 EdgeNeXt 모델을 평가합니다.

### **4.1 데이터셋(Dataset)**

모든 분류 실험에는 **ImageNet-1K**【35】 데이터셋을 사용합니다.  
이 데이터셋은 1,000개의 클래스에 대해 약 **128만 개의 학습 이미지**와 **5만 개의 검증 이미지**를 제공합니다.  
기존 연구【17, 29】를 따라, 모든 실험에서는 **검증 세트의 Top-1 정확도**를 보고합니다.

객체 탐지에는 **COCO**【22】 데이터셋을 사용하며, 약 **11.8만 개의 학습 이미지**와 **5,000개의 검증 이미지**가 포함되어 있습니다.  
분할(segmentation)에는 **Pascal VOC 2012** 데이터셋【10】을 사용하며, **약 1만 장의 이미지와 시맨틱 분할 마스크**가 포함되어 있습니다.  
기존 연구【29】의 표준 실험 설정을 따라, **추가 데이터와 주석(annotation)**은 【22】 및 【13】에서 가져와 사용합니다.

### **4.2 구현 세부사항(Implementation Details)**

EdgeNeXt 모델은 **입력 해상도 256×256**에서 학습되며, **유효 배치 사이즈는 4096**입니다.  
모든 실험은 **300 에폭(epoch)** 동안 **AdamW 옵티마이저**【27】를 사용해 수행되며, **학습률(learning rate)은 6e-3**, **가중치 감쇠(weight decay)는 0.05**로 설정됩니다.  
**Cosine learning rate schedule**【26】을 사용하고, **20 에폭 동안 선형 워밍업(linear warmup)**을 적용합니다.

훈련 시 사용된 데이터 증강(data augmentation)은 다음과 같습니다:

- **Random Resized Crop (RRC)**
- **수평 뒤집기(Horizontal Flip)**
- **RandAugment**【6】 (단, RandAugment는 EdgeNeXt-S 모델에만 적용됨)

또한 **Multi-scale Sampler**【29】를 훈련에 활용하며, **Stochastic Depth**【18】는 **EdgeNeXt-S 모델에 한해 비율 0.1**로 사용됩니다.  
훈련 중에는 **EMA(Exponential Moving Average)**【32】를 **모멘텀 0.9995**로 적용합니다.

**추론 시**에는 이미지를 **292×292로 리사이즈**한 후, **중앙 크롭(center crop)**으로 **256×256 해상도**를 적용합니다.  
또한, **기존 방법들과 공정한 비교를 위해** EdgeNeXt-S 모델을 **224×224 해상도**에서도 학습하고 정확도를 보고합니다.  
분류 실험은 **A100 GPU 8장**에서 수행되었으며, **EdgeNeXt-S 모델의 평균 학습 시간은 약 30시간**입니다.

**객체 탐지 및 분할 실험**에서는 【29】와 유사한 설정으로 EdgeNeXt를 파인튜닝하고, 각각

- **mAP (mean Average Precision)**: IOU 0.50~0.95 기준
- **mIOU (mean Intersection over Union)**

을 보고합니다.  
이 실험은 **A100 GPU 4장**에서 수행되었으며, 평균 학습 시간은

- 객체 탐지: **약 36시간**
- 분할: **약 7시간**입니다.

또한, **NVIDIA Jetson Nano**<sup>[1](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)</sup>와  
**NVIDIA A100 40GB GPU**에서의 **추론 지연(latency)**도 보고합니다.

- **Jetson Nano**에서는 모든 모델을 **TensorRT 엔진<sup>[2](https://github.com/NVIDIA/TensorRT)</sup>**으로 변환하고, **FP16 모드**, **배치 크기 1**로 추론을 수행합니다.
- **A100**에서는 【25】와 동일하게 **PyTorch v1.8.1**, **배치 크기 256**으로 지연 시간을 측정합니다.

**표 2: ImageNet-1K 분류 작업에서의 성능 비교**  
제안하는 **EdgeNeXt 모델**과 최신 **경량 완전 합성곱(CNN 기반)**, **트랜스포머 기반**, **하이브리드 모델들**을 비교한 결과입니다.  
우리 모델은 **정확도와 연산량(즉, 파라미터 수 및 곱셈-덧셈 연산량, MAdds)** 간의 **더 우수한 균형(trade-off)**을 달성합니다.

![](/assets/images/posts/550/img_5.png)

**표 3: EdgeNeXt 다양한 변형 모델과 MobileViT 대응 모델 간의 비교**  
EdgeNeXt의 여러 변형 모델들과 MobileViT 모델을 비교한 결과입니다.  
마지막 두 열은 **NVIDIA Jetson Nano**와 **A100 디바이스**에서의 **지연 시간(latency)**을 각각 **ms(밀리초)**와 **μs(마이크로초)** 단위로 나타냅니다. 우리의 **EdgeNeXt 모델들은 동일한 모델 크기에서 더 높은 정확도와 더 낮은 지연 시간**을 달성하며, **1.3M 파라미터 수준까지도 유연하게 스케일 다운 가능한 설계 유연성**을 보여줍니다.

![](/assets/images/posts/550/img_6.png)

### **4.3 이미지 분류(Image Classification)**

**표 2**는 제안하는 **EdgeNeXt 모델**을 기존의 최신 **완전 합성곱(ConvNet)**, **트랜스포머 기반(ViT)**, **하이브리드 모델**들과 비교한 결과를 보여줍니다. 전반적으로, EdgeNeXt는 세 가지 범주의 기존 방법들과 비교했을 때, **정확도 대비 연산량(파라미터 수 및 MAdds)의 트레이드오프** 측면에서 더 우수한 성능을 나타냅니다(Fig. 1 참조).

#### **ConvNet과의 비교**

EdgeNeXt는 **유사한 파라미터 수**를 유지하면서도 **Top-1 정확도 측면에서 경량 ConvNet들을 큰 폭으로 능가**합니다(표 2 참조).  
일반적으로 ConvNet은 Attention 연산이 없기 때문에 Transformer 및 하이브리드 모델에 비해 MAdds가 적지만, **전역 수용 영역(global receptive field)**을 갖지 못하는 한계가 있습니다. 예를 들어, **EdgeNeXt-S는 MobileNetV2**【36】보다 더 많은 MAdds를 가지지만, **Top-1 정확도에서 4.1% 더 높은 성능을 더 적은 파라미터 수로 달성**합니다. 또한, EdgeNeXt-S는 **ShuffleNetV2**【28】와 **MobileNetV3**【16】보다 각각 **4.3%**, **3.6% 더 높은 정확도**를 유사한 파라미터 수로 기록했습니다.

#### **ViT와의 비교**

EdgeNeXt는 **더 적은 파라미터 수와 MAdds**로 **최근 ViT 계열 모델들보다 더 나은 성능**을 ImageNet-1K에서 보여줍니다.  
예를 들어, **EdgeNeXt-S는 78.8%의 Top-1 정확도**를 달성하며, 이는 **T2T-ViT**【44】보다 **2.3%**, **DeiT-T**【40】보다 **6.6%** 절대 우위를 보입니다.

#### **하이브리드 모델과의 비교**

제안한 **EdgeNeXt**는 **MobileFormer**【4】, **ViT-C**【42】, **CoaT-Lite-T**【7】와 같은 모델들을 **더 적은 파라미터와 연산량으로 성능을 능가**합니다(표 2 참조). 특히, **MobileViT**【29】와의 공정한 비교를 위해, EdgeNeXt는 **입력 해상도 256×256**에서 학습되었으며, **모델 크기별(S, XS, XXS)**로도 **MAdds는 더 적고, 엣지 디바이스에서 추론 속도는 더 빠르며, 일관된 성능 향상**을 보여줍니다(표 3 참조). 예를 들어, **EdgeNeXt-XXS 모델은 1.3M 파라미터로 71.2% Top-1 정확도**를 달성하며, 대응되는 MobileViT 버전보다 **2.2% 더 높은 성능**을 기록했습니다. 마지막으로, **EdgeNeXt-S 모델은 5.6M 파라미터로 79.4% 정확도**를 달성하였고, 이는 **MobileViT-S보다 1.0% 더 높은 정확도**입니다. 이 결과는 **우리 설계의 효과성과 일반화 능력**을 보여줍니다.

또한, **[34]의 방법을 따라 지식 증류(Knowledge Distillation)**를 적용해 EdgeNeXt-S 모델을 학습한 결과, **Top-1 기준 ImageNet 정확도 81.1%**를 달성하였습니다.

### **4.4 ImageNet-21K 사전학습 (Pretraining)**

EdgeNeXt의 성능 한계를 더욱 탐구하기 위해, 우리는 **18.5M 파라미터와 3.8G MAdds**를 갖는 **EdgeNeXt-B 모델**을 설계하고,  
이를 **ImageNet-21K**【35】의 일부 서브셋에서 사전학습(pretraining)한 후, **표준 ImageNet-1K 데이터셋**에서 파인튜닝(finetuning)했습니다.

**ImageNet-21K (2021년 겨울 버전)**은 약 **1,300만 개의 이미지와 1만 9천 개의 클래스**로 구성되어 있습니다.  
우리는 【33】을 따라, **예제가 적은 클래스들을 제거한 후**, 이를 **약 1,100만 개의 학습 이미지**와 **52만 2천 개의 검증 이미지**로 나누고,  
총 **10,450개 클래스**로 구성된 사전학습용 데이터셋을 구성합니다. 이 데이터셋을 **ImageNet-21K-P**라고 부릅니다.

ImageNet-21K-P의 사전학습은 **ConvNeXt**【25】의 학습 설정을 엄격히 따릅니다.  
또한, 학습 수렴 속도를 높이기 위해 **ImageNet-1K에서 사전학습된 모델을 초기화 가중치로 사용**합니다.

마지막으로, ImageNet-21K-P에서 학습한 모델을 **ImageNet-1K에서 30 에폭 동안 파인튜닝**하며,  
이때 **학습률은 7.5e-5**, **유효 배치 크기는 512**로 설정합니다.

**표 4: EdgeNeXt-B 모델의 대규모 ImageNet-21K-P 사전학습 결과**  
EdgeNeXt-B 모델은 **ConvNeXt**【25】와 **MobileViT-V2**【30】 등 최신 모델들과 비교했을 때,  
**정확도 대비 연산량 측면에서 더 우수한 트레이드오프**를 달성합니다.

![](/assets/images/posts/550/img_7.png)

### **4.5 엣지 디바이스에서의 추론(Inference on Edge Devices)**

우리는 **NVIDIA Jetson Nano** 엣지 디바이스에서 **EdgeNeXt 모델들의 추론 시간**을 측정하고, 이를 **최신 MobileViT**【29】 모델과 비교합니다(표 3 참조). 모든 모델은 **TensorRT 엔진**으로 변환된 후, **FP16 모드**로 추론이 수행됩니다. 우리 모델은 **유사한 파라미터 수**, **더 적은 MAdds**, **더 높은 Top-1 정확도**를 유지하면서도 **엣지 디바이스에서 낮은 지연 시간**을 달성합니다. 표 3에는 **MobileViT와 EdgeNeXt 모델** 모두에 대한 **A100 GPU에서의 추론 시간**도 함께 나와 있습니다. 이를 통해 다음과 같은 사실을 확인할 수 있습니다:

- **EdgeNeXt-XXS 모델은 A100 기준 MobileViT-XXS보다 약 34% 빠르며**,
- Jetson Nano에서는 **약 8% 빠른 성능**을 보입니다.

이 결과는 **EdgeNeXt가 MobileViT보다 고성능 하드웨어를 더 효율적으로 활용함을 시사**합니다.

### **4.6 객체 탐지(Object Detection)**

우리는 **EdgeNeXt를 SSDLite의 백본(backbone)**으로 사용하고, 이를 **COCO 2017 데이터셋**【22】에서 **입력 해상도 320×320**으로 파인튜닝합니다. **SSD**【23】와 **SSDLite**의 차이는, **SSD 헤드의 표준 합성곱(convolution)**이 **분리 합성곱(separable convolution)**으로 대체된 점입니다. 실험 결과는 **표 5**에 보고됩니다. **EdgeNeXt는 일관되게 MobileNet 계열 백본보다 우수한 성능을 보이며**, MobileViT 백본과 비교해도 **경쟁력 있는 성능**을 기록합니다. 특히, **더 적은 MAdds**와 **유사한 파라미터 수**를 유지하면서, **EdgeNeXt는 최고 성능인 27.9 box AP를 달성하며**, 이는 **MobileViT보다 약 38% 더 적은 MAdds로 얻은 결과**입니다.

![](/assets/images/posts/550/img_8.png)

**표 5: COCO 객체 탐지에서의 최신 기법과의 비교**  
EdgeNeXt는 기존 방법들보다 향상된 성능을 보여줍니다.

### **4.7 시맨틱 분할(Semantic Segmentation)**

우리는 **EdgeNeXt를 DeepLabv3**【3】의 백본(backbone)으로 사용하고, **Pascal VOC**【10】 데이터셋에서 **입력 해상도 512×512**로 모델을 파인튜닝합니다.  
**DeepLabv3**는 **계단식 구조(cascade design)**에서 **팽창 합성곱(dilated convolution)**과 **공간 피라미드 풀링(spatial pyramid pooling)**을 활용하여, **다중 스케일 특징(multi-scale features)**을 인코딩하는 데 효과적이며, 다양한 크기의 객체를 인식하는 데 유용합니다. 우리 모델은 **검증 데이터셋에서 80.2 mIOU**를 달성하며, **MobileViT보다 약 1.1 포인트 높은 성능**을 **약 36% 더 적은 MAdds로** 기록합니다.

![](/assets/images/posts/550/img_9.png)

**표 6: VOC 시맨틱 분할 작업에서의 최신 기법들과의 비교**  
우리 모델은 **합리적인 성능 향상**을 제공합니다.

### **5. 설계 요소 분석(Ablations)**

이 섹션에서는 제안한 **EdgeNeXt 모델의 다양한 설계 선택사항들에 대한 ablation 분석**을 수행합니다.

#### **SDTA 인코더와 적응형 커널 크기**

**표 7**은 제안한 아키텍처에서 **SDTA 인코더와 적응형 커널 크기(adaptive kernel sizes)**의 중요성을 보여줍니다.  
SDTA 인코더를 일반 합성곱 인코더로 대체할 경우 **정확도가 1.1% 감소**하며, 이는 SDTA 인코더의 유용성을 나타냅니다.  
또한 네트워크의 모든 단계에서 **커널 크기를 7로 고정**하면, 정확도가 **추가로 0.4% 감소**합니다.

이러한 결과는 제안한 설계가 **정확도와 속도 간의 최적의 트레이드오프**를 제공함을 시사합니다.

또한, **표 7**에서는 **SDTA 구성 요소(예: adaptive branching, positional encoding)**의 기여도도 분석합니다. **적응형 분기(adaptive branching)**와 **위치 인코딩(positional encoding)**을 제거할 경우, **정확도가 소폭 하락**합니다.

**표 7: EdgeNeXt와 SDTA 인코더의 다양한 구성 요소에 대한 Ablation 분석 결과**  
결과는 **SDTA 인코더와 적응형 커널**이 설계에 있어 **실질적인 성능 이점을 제공함**을 보여주며, SDTA 모듈 내에서 **adaptive branching과 위치 인코딩(PE)** 또한 **필수적인 요소**임을 나타냅니다.

![](/assets/images/posts/550/img_10.png)

### **하이브리드 설계(Hybrid Design)**

**표 8**은 **EdgeNeXt 모델의 다양한 하이브리드 설계 선택**에 대한 ablation 실험 결과를 보여줍니다. **MetaFormer**【43】에서 영감을 받아, **마지막 두 단계의 모든 합성곱 모듈을 SDTA 인코더로 대체**해 보았습니다. 그 결과, **마지막 두 단계의 모든 블록을 SDTA로 구성할 경우 정확도가 가장 높아지지만**, **지연 시간(latency)이 증가**하는 것으로 나타났습니다(표의 2행 vs 3행 비교). 우리가 제안한 하이브리드 설계는, **마지막 세 단계 각각의 마지막 블록에만 SDTA 인코더를 사용하는 방식**이며, 이 설정은 **정확도와 속도 간의 최적의 균형**을 제공합니다.

![](/assets/images/posts/550/img_11.png)

**표 8: 하이브리드 아키텍처에 대한 Ablation 분석**  
각 단계의 마지막 블록에 하나씩만 SDTA 인코더를 사용하는 것이 **정확도와 추론 지연 시간 사이에서 가장 좋은 트레이드오프**를 달성합니다.

**표 9**는 네트워크의 **서로 다른 단계에서 SDTA 인코더를 사용할 때의 중요성**을 분석합니다. 마지막 세 단계 각각의 마지막 블록에 **점진적으로 SDTA 인코더를 추가**할수록 **정확도가 향상**되지만, **일부 추론 지연 증가**가 동반됩니다. 그러나 **4행 설정에서는**, **마지막 세 단계의 마지막 블록에 SDTA 인코더를 추가**할 때, **정확도와 속도 사이의 최적 균형**이 달성됨을 확인할 수 있습니다. 또한, 네트워크의 **첫 번째 단계에 전역 SDTA 인코더를 추가**하는 것은, 해당 단계의 **특징이 충분히 성숙하지 않아 성능 향상에 도움이 되지 않음**을 확인했습니다.

![](/assets/images/posts/550/img_12.png)

**표 9: 네트워크 각 단계에서 SDTA 인코더 사용에 대한 Ablation 분석**  
**마지막 세 단계에 SDTA 인코더를 포함**시키는 것이 성능 향상에 도움이 되며, **첫 번째 단계에 전역 SDTA를 추가하는 것은 효과가 없음**을 보여줍니다.

또한, 우리는 각 단계의 **시작 지점**에 SDTA를 넣는 것과 **끝 지점**에 넣는 것의 성능 차이를 비교했습니다. **표 10**은 **각 단계의 끝에 SDTA 인코더를 배치하는 것이 더 유리함**을 보여줍니다. 이러한 결과는 최근 연구【21】와도 일치합니다.

![](/assets/images/posts/550/img_13.png)

**표 10: EdgeNeXt에서 각 단계의 시작/끝에 SDTA를 적용한 Ablation 분석**  
SDTA 인코더는 **각 단계의 끝에 사용하는 것이 일반적으로 더 효과적**임을 보여줍니다.

### **활성화 함수와 정규화(Activation and Normalization)**

**EdgeNeXt**는 네트워크 전체에서 **GELU 활성화 함수**와 **Layer Normalization**을 사용합니다. 하지만, 현재 PyTorch에서 제공되는 **GELU와 LayerNorm의 구현은 고속 추론(high-speed inference)에는 최적화되어 있지 않음**을 발견했습니다. 이에 따라, 우리는 **GELU를 Hard-Swish로**, **LayerNorm을 BatchNorm으로 대체**한 뒤 모델을 다시 학습시켰습니다. **그림 3**은 이러한 변경이 **정확도는 소폭 감소**시키지만, **지연 시간(latency)은 크게 줄여줌**을 보여줍니다.

![](/assets/images/posts/550/img_14.png)

**그림 3: 다양한 활성화 함수 및 정규화 레이어가 네트워크 변형 모델의 정확도와 지연 시간에 미치는 영향에 대한 ablation**  
**GELU와 LayerNorm 대신 Hard-Swish와 BatchNorm을 사용**하면, **정확도는 일부 손해를 보지만**, **지연 시간은 크게 향상**됨을 나타냅니다.

### **6. 정성적 결과(Qualitative Results)**

**그림 4와 그림 5**는 각각 **EdgeNeXt 객체 탐지 모델과 분할 모델의 정성적 결과**를 보여줍니다.우리 모델은 **다양한 시점(view)에서 객체를 효과적으로 탐지하고 분할**할 수 있습니다.

![](/assets/images/posts/550/img.jpg)

**그림 4:**  
COCO 검증 데이터셋에서의 **EdgeNeXt 객체 탐지 모델**의 정성적 결과. 모델은 COCO 데이터셋에서 **80개 탐지 클래스**로 학습되었으며, **다양한 장면에서 객체를 효과적으로 위치 지정하고 분류**할 수 있습니다.

![](/assets/images/posts/550/img_1.jpg)

**그림 5:**  
**COCO 검증 데이터셋(미사용 이미지)**에서의 **EdgeNeXt 분할 모델**의 정성적 결과.  
모델은 **Pascal VOC 데이터셋(20개 클래스)**으로 학습되었습니다.  
(a): 예측된 **시맨틱 분할 마스크**, 검은색은 배경 픽셀을 나타냄  
(b): **원본 이미지 위에 예측 마스크를 중첩**한 결과  
(c): **Pascal VOC의 모든 클래스에 대한 색상 인코딩 정보**  
우리 모델은 **보지 못한 COCO 이미지에 대해서도 고품질의 분할 마스크**를 생성합니다.

### **7. 결론(Conclusion)**

트랜스포머 모델의 성공은 CNN에 비해 **높은 계산 비용(computational overhead)**을 수반합니다. 이 중 **Self-Attention 연산이 가장 큰 원인**이며, 이는 트랜스포머 기반 비전 모델이 **CNN 기반 모바일 아키텍처에 비해 엣지 디바이스에서 느린 주된 이유**입니다.

본 논문에서는,

- **합성곱(convolution)**과
- **효율적인 Self-Attention 기반 인코더**를 결합한 **하이브리드 설계(hybrid design)**를 제안합니다.

이 설계는 **로컬 및 전역 정보를 함께 효과적으로 모델링**하면서도, **파라미터 수와 연산량(MAdds) 모두에서 효율적이며**,  
**최신 방법들보다 우수한 성능**을 비전 과제에서 달성합니다. 다양한 EdgeNeXt 변형 모델에 대한 실험 결과는,**제안한 모델의 효과성과 일반화 능력**을 뚜렷하게 입증합니다.
