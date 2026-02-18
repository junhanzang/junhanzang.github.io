---
title: "Adding Conditional Control to Text-to-Image Diffusion Models"
date: 2024-06-27 21:49:18
categories:
  - 인공지능
---

<https://arxiv.org/abs/2302.05543>

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

초록

우리는 대규모 사전 학습된 텍스트-이미지 확산 모델에 공간 조건 제어를 추가하기 위한 신경망 아키텍처인 ControlNet을 소개합니다. ControlNet은 생산 준비가 된 대규모 확산 모델을 고정하고, 수십억 개의 이미지로 사전 학습된 깊고 강력한 인코딩 레이어를 강력한 백본으로 재사용하여 다양한 조건 제어를 학습합니다. 신경망 아키텍처는 "제로 컨볼루션"(제로로 초기화된 컨볼루션 레이어)과 연결되어 매개변수가 제로에서 점진적으로 성장하도록 하여 미세 조정에 해로운 노이즈가 영향을 미치지 않도록 합니다. 우리는 Stable Diffusion과 함께 엣지, 깊이, 분할, 인간 자세 등의 다양한 조건 제어를 단일 또는 복수의 조건으로, 프롬프트 유무와 상관없이 테스트했습니다. ControlNets의 학습이 작은(<50k) 및 큰(>1m) 데이터셋에서도 견고함을 보여줍니다. 광범위한 결과는 ControlNet이 이미지 확산 모델을 제어하는 더 넓은 응용을 촉진할 수 있음을 보여줍니다.

![](/assets/images/posts/179/img.png)

Figure 1: 학습된 조건으로 안정적인 확산 제어하기. ControlNet은 사용자들이 Canny 에지(상단), 인간 자세(하단) 등과 같은 조건을 추가하여 대규모 사전 학습된 확산 모델의 이미지 생성을 제어할 수 있게 합니다. 기본 결과는 "고품질의 세밀하고 전문적인 이미지"라는 프롬프트를 사용합니다. 사용자는 선택적으로 "주방에서 요리사"와 같은 프롬프트를 제공할 수 있습니다.

### 1. 서론

많은 사람들은 독특한 이미지를 캡처하고 싶어하는 시각적 영감을 경험합니다. 텍스트-이미지 확산 모델[54, 62, 72]의 출현으로 이제 텍스트 프롬프트를 입력하여 시각적으로 놀라운 이미지를 생성할 수 있습니다. 그러나 텍스트-이미지 모델은 이미지의 공간적 구성을 제어하는 데 한계가 있으며, 텍스트 프롬프트만으로 복잡한 레이아웃, 자세, 형태 및 모양을 정확하게 표현하는 것은 어렵습니다. 우리의 정신적 이미지를 정확히 일치시키는 이미지를 생성하려면 프롬프트를 편집하고 결과 이미지를 검사한 후 다시 편집하는 수많은 시행착오가 필요합니다.

사용자가 원하는 이미지 구성을 직접 지정할 수 있도록 추가 이미지를 제공함으로써 더 세밀한 공간 제어를 가능하게 할 수 있을까요? 컴퓨터 비전 및 기계 학습에서는 이러한 추가 이미지(예: 에지 맵, 인간 자세 스켈레톤, 분할 맵, 깊이, 노멀 등)가 이미지 생성 과정에서 조건으로 처리되는 경우가 많습니다. 이미지-이미지 번역 모델[34, 98]은 조건 이미지를 대상 이미지로 매핑하는 방법을 학습합니다. 연구 커뮤니티는 또한 텍스트-이미지 모델을 공간 마스크[6, 20], 이미지 편집 지침[10], 미세 조정을 통한 개인화[21, 75] 등으로 제어하는 방법을 모색해 왔습니다. 이미지 변형 생성, 인페인팅과 같은 몇 가지 문제는 디노이징 확산 프로세스를 제한하거나 주의 레이어 활성화를 편집하는 것과 같은 훈련이 필요 없는 기술로 해결할 수 있지만, 깊이-이미지, 자세-이미지 등과 같은 더 다양한 문제는 엔드-투-엔드 학습과 데이터 기반 솔루션이 필요합니다.

대규모 텍스트-이미지 확산 모델에 대해 엔드-투-엔드 방식으로 조건 제어를 학습하는 것은 도전적입니다. 특정 조건에 대한 학습 데이터의 양은 일반적인 텍스트-이미지 학습에 사용 가능한 데이터보다 훨씬 적을 수 있습니다. 예를 들어, 다양한 특정 문제(예: 객체 형태/노멀, 인간 자세 추출 등)에 대한 가장 큰 데이터셋은 일반적으로 약 100K 정도로, Stable Diffusion[82]을 학습하는 데 사용된 LAION-5B[79] 데이터셋보다 50,000배 작습니다. 대규모 사전 학습된 모델을 제한된 데이터로 직접 미세 조정하거나 계속 학습하면 과적합과 치명적인 망각[31, 75]을 초래할 수 있습니다. 연구자들은 학습 가능한 매개변수의 수 또는 순위를 제한하여 이러한 망각을 완화할 수 있음을 보여주었습니다[14, 25, 31, 92]. 우리의 문제를 해결하기 위해서는 복잡한 형태와 다양한 고급 의미를 가진 야생 조건 이미지를 처리하기 위해 더 깊거나 맞춤형 신경 아키텍처를 설계할 필요가 있을 수 있습니다.

이 논문은 대규모 사전 학습된 텍스트-이미지 확산 모델(우리 구현에서는 Stable Diffusion)을 위해 조건 제어를 학습하는 엔드-투-엔드 신경망 아키텍처인 ControlNet을 제시합니다. ControlNet은 대규모 모델의 품질과 기능을 유지하면서, 해당 모델의 인코딩 레이어의 학습 가능한 복사본을 만들고, 이들을 고정하여 백본으로 활용합니다. 이 아키텍처는 대규모 사전 학습된 모델을 다양한 조건 제어를 학습하는 강력한 백본으로 처리합니다. 학습 가능한 복사본과 원래의 고정된 모델은 제로 컨볼루션 레이어로 연결되며, 이들의 가중치는 제로로 초기화되어 훈련 중에 점진적으로 성장합니다. 이 아키텍처는 훈련 초기 단계에서 대규모 확산 모델의 심층 기능에 해로운 노이즈가 추가되지 않도록 하고, 학습 가능한 복사본의 대규모 사전 학습된 백본이 그러한 노이즈로 인해 손상되지 않도록 보호합니다.

우리의 실험은 ControlNet이 다양한 조건 입력(Canny 에지, Hough 선, 사용자 낙서, 인간 키 포인트, 분할 맵, 형태 노멀, 깊이 등)을 통해 Stable Diffusion을 제어할 수 있음을 보여줍니다(그림 1). 우리는 단일 조건 이미지를 사용하여, 텍스트 프롬프트 유무에 관계없이 접근 방식을 테스트하고, 여러 조건의 구성을 지원하는 방법을 입증합니다. 또한 ControlNet의 학습이 다양한 크기의 데이터셋에서 견고하고 확장 가능하며, 깊이-이미지 조건화와 같은 일부 작업에서는 단일 NVIDIA RTX 3090Ti GPU로 학습한 ControlNets가 대규모 계산 클러스터에서 학습한 산업 모델과 경쟁력 있는 결과를 달성할 수 있음을 보고합니다. 마지막으로, 우리는 모델의 각 구성 요소의 기여도를 조사하는 소거 실험을 수행하고, 사용자 연구를 통해 여러 강력한 조건 이미지 생성 기준과 우리 모델을 비교합니다.

요약하면, (1) 우리는 대규모 사전 학습된 텍스트-이미지 확산 모델에 공간적으로 국한된 입력 조건을 추가할 수 있는 신경망 아키텍처인 ControlNet을 제안하고, (2) Canny 에지, Hough 선, 사용자 낙서, 인간 키 포인트, 분할 맵, 형태 노멀, 깊이, 만화 선화 등을 조건으로 하는 사전 학습된 ControlNets를 제시하며, (3) 여러 대체 아키텍처와 비교한 소거 실험과 다양한 작업에 걸친 이전 기준에 대한 사용자 연구를 통해 방법을 검증합니다.

### 2. 관련 연구

#### 2.1. 신경망 미세 조정

신경망을 미세 조정하는 한 가지 방법은 추가 학습 데이터를 사용하여 직접 계속 학습시키는 것입니다. 그러나 이 접근 방식은 과적합, 모드 붕괴, 치명적인 망각을 초래할 수 있습니다. 많은 연구는 이러한 문제를 피하는 미세 조정 전략을 개발하는 데 초점을 맞추고 있습니다.

**하이퍼네트워크(HyperNetwork)**는 자연어 처리(NLP) 커뮤니티에서 시작된 접근 방식으로, 작은 순환 신경망을 학습시켜 더 큰 신경망의 가중치에 영향을 미치게 합니다[25]. 이 방법은 생성적 적대 신경망(GANs)을 사용한 이미지 생성에도 적용되었습니다[4, 18]. Heathen 등[26]과 Kurumuz[43]은 HyperNetworks를 Stable Diffusion[72]에 적용하여 출력 이미지의 예술적 스타일을 변경했습니다.

**어댑터(Adapters)** 방법은 사전 학습된 트랜스포머 모델에 새로운 모듈 레이어를 삽입하여 다른 작업에 맞게 사용자 정의하는 데 널리 사용됩니다[30, 84]. 컴퓨터 비전에서는 어댑터가 점진적 학습[74]과 도메인 적응[70]에 사용됩니다. 이 기술은 종종 CLIP[66]과 함께 사전 학습된 백본 모델을 다른 작업으로 전환하는 데 사용됩니다[23, 66, 85, 94]. 최근에는 어댑터가 비전 트랜스포머[49, 50]와 ViT-Adapter[14]에서 성공적인 결과를 얻었습니다. 우리의 연구와 동시 진행된 연구에서는 T2I-Adapter[56]가 Stable Diffusion을 외부 조건에 맞게 조정했습니다.

**추가 학습(Additive Learning)**은 원래 모델의 가중치를 고정하고 학습된 가중치 마스크[51, 74], 가지치기[52], 하드 어텐션[80]을 사용하여 소수의 새로운 매개변수를 추가함으로써 망각을 피합니다. **사이드 튜닝(Side-Tuning)**[92]은 고정된 모델과 추가된 네트워크의 출력을 선형적으로 블렌딩하여 추가 기능을 학습하는 사이드 브랜치 모델을 사용합니다. 이는 미리 정의된 블렌딩 가중치 스케줄을 따릅니다.

**저차원 적응(LoRA)**[31]은 과잉 매개변수화된 모델이 저차원 고유 공간에 존재한다는 관찰을 바탕으로, 저차원 행렬로 매개변수의 오프셋을 학습하여 치명적인 망각을 방지합니다[2, 47].

**제로 초기화 레이어**는 네트워크 블록을 연결하기 위해 ControlNet에서 사용됩니다. 신경망에 대한 연구는 네트워크 가중치의 초기화 및 조작에 대해 광범위하게 논의해 왔습니다[36, 37, 44, 45, 46, 76, 83, 95]. 예를 들어, 가중치의 가우시안 초기화는 제로로 초기화하는 것보다 덜 위험할 수 있습니다[1]. 최근 Nichol 등[59]은 확산 모델의 컨볼루션 레이어 초기 가중치를 조정하여 학습을 개선하는 방법에 대해 논의했으며, 그들의 "제로 모듈" 구현은 가중치를 제로로 조정하는 극단적인 사례입니다. Stability의 모델 카드[83]에서도 신경망 레이어에서 제로 가중치의 사용을 언급합니다. 초기 컨볼루션 가중치 조작은 ProGAN[36], StyleGAN[37], Noise2Noise[46]에서도 논의됩니다.

### 2.2 이미지 확산

**이미지 확산 모델**은 Sohl-Dickstein 등[81]에 의해 처음 소개되었으며, 최근 이미지 생성에 적용되었습니다[17, 42]. **잠재 확산 모델(LDM)**[72]은 잠재 이미지 공간에서 확산 단계를 수행하여 계산 비용을 줄입니다[19]. 텍스트-이미지 확산 모델은 CLIP[66]과 같은 사전 학습된 언어 모델을 통해 텍스트 입력을 잠재 벡터로 인코딩하여 최첨단 이미지 생성 결과를 달성합니다. **Glide**[58]는 이미지 생성과 편집을 지원하는 텍스트 유도 확산 모델입니다. **Disco Diffusion**[5]은 CLIP 지침으로 텍스트 프롬프트를 처리합니다. **Stable Diffusion**[82]은 잠재 확산[72]의 대규모 구현입니다. **Imagen**[78]은 잠재 이미지를 사용하지 않고 피라미드 구조를 사용하여 픽셀을 직접 확산시킵니다. 상용 제품으로는 **DALL-E2**[62]와 **Midjourney**[54]가 있습니다.

**이미지 확산 모델 제어**는 개인화, 사용자 정의 또는 작업별 이미지 생성을 용이하게 합니다. 이미지 확산 과정은 색상 변형[53] 및 인페인팅[67, 7]에 대한 일부 제어를 직접 제공합니다. 텍스트 유도 제어 방법은 프롬프트 조정, CLIP 기능 조작 및 교차 주의 수정에 중점을 둡니다[7, 10, 20, 27, 40, 41, 58, 64, 67]. **MakeAScene**[20]은 분할 마스크를 토큰으로 인코딩하여 이미지 생성을 제어합니다. **SpaText**[6]는 분할 마스크를 지역화된 토큰 임베딩으로 매핑합니다. **GLIGEN**[48]은 확산 모델의 주의 레이어에서 새로운 매개변수를 학습하여 기초 생성 기능을 제공합니다. **Textual Inversion**[21]과 **DreamBooth**[75]는 사용자가 제공한 소량의 예제 이미지를 사용하여 이미지 확산 모델을 미세 조정하여 생성된 이미지의 콘텐츠를 개인화할 수 있습니다. 프롬프트 기반 이미지 편집[10, 33, 86]은 프롬프트로 이미지를 조작할 수 있는 실용적인 도구를 제공합니다. Voynov 등[88]은 스케치를 사용하여 확산 과정을 최적화하는 방법을 제안합니다. 동시 진행된 연구[8, 9, 32, 56]는 확산 모델을 제어하는 다양한 방법을 조사합니다.

### 2.3 이미지-이미지 변환

**조건부 GANs**[15, 34, 63, 90, 93, 97, 98, 99]과 트랜스포머[13, 19, 68]는 서로 다른 이미지 도메인 간의 매핑을 학습할 수 있습니다. 예를 들어, **Taming Transformer**[19]는 비전 트랜스포머 접근 방식입니다; **Palette**[77]는 처음부터 훈련된 조건부 확산 모델입니다; **PITI**[89]는 이미지-이미지 변환을 위한 사전 학습 기반 조건부 확산 모델입니다. 사전 학습된 GAN을 조작하여 특정 이미지-이미지 작업을 처리할 수 있습니다. 예를 들어, **StyleGANs**는 추가 인코더[71]로 제어할 수 있으며, 더 많은 응용 프로그램이 다음 연구에서 다뤄졌습니다[3, 22, 38, 39, 55, 60, 65, 71].

![](/assets/images/posts/179/img_1.png)

**그림 2:** 신경 블록은 입력으로 특징 맵 x를 받고 출력으로 또 다른 특징 맵 y를 생성합니다, (a)와 같이. 이러한 블록에 ControlNet을 추가하려면 원래 블록을 고정하고 학습 가능한 복사본을 만들어 제로 컨볼루션 레이어(즉, 가중치와 바이어스가 모두 제로로 초기화된 1 × 1 컨볼루션)로 연결합니다. 여기서 ccc는 네트워크에 추가하려는 조건 벡터입니다, (b)와 같이.

### 3. 방법

ControlNet은 대규모 사전 학습된 텍스트-이미지 확산 모델을 공간적으로 국한되고 작업 특화된 이미지 조건으로 향상시킬 수 있는 신경망 아키텍처입니다. 우리는 먼저 섹션 3.1에서 ControlNet의 기본 구조를 소개하고, 섹션 3.2에서 ControlNet을 이미지 확산 모델인 Stable Diffusion[72]에 적용하는 방법을 설명합니다. 섹션 3.3에서는 우리의 학습 방법을 자세히 설명하고, 섹션 3.4에서는 여러 ControlNet을 조합하는 것과 같은 추론 중의 추가 고려 사항에 대해 설명합니다.

![](/assets/images/posts/179/img_2.png)

![](/assets/images/posts/179/img_3.png)

![](/assets/images/posts/179/img_4.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

고정된 원래 블록은 원래 네트워크의 일부로 그대로 남아 있고, 학습 가능한 블록이 추가되어 원래 블록과 함께 작동하는 형태

```
import torch
import torch.nn as nn

class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # 가중치와 바이어스를 모두 제로로 초기화합니다.
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

# 사용 예시
in_channels = 64
out_channels = 64
zero_conv_layer = ZeroConv2d(in_channels, out_channels)

# 입력 텐서
input_tensor = torch.randn(1, in_channels, 32, 32)  # 예를 들어, 배치 크기 1, 32x32 크기의 특징 맵

# 제로 컨볼루션 레이어 통과
output_tensor = zero_conv_layer(input_tensor)
print(output_tensor.shape)  # (1, out_channels, 32, 32) 형태의 출력 텐서
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

![](/assets/images/posts/179/img_5.png)

**그림 3:** Stable Diffusion의 U-Net 아키텍처는 인코더 블록과 중간 블록에서 ControlNet과 연결됩니다. 고정된 회색 블록은 Stable Diffusion V1.5(또는 동일한 U-Net 아키텍처를 사용하는 V2.1)의 구조를 보여줍니다. 학습 가능한 파란색 블록과 흰색 제로 컨볼루션 레이어가 추가되어 ControlNet을 구축합니다.

### 3.2. 텍스트-이미지 확산을 위한 ControlNet

우리는 Stable Diffusion[72]을 예로 들어 ControlNet이 대규모 사전 학습된 확산 모델에 조건 제어를 추가할 수 있는 방법을 보여줍니다. Stable Diffusion은 기본적으로 인코더, 중간 블록, 그리고 스킵 연결된 디코더가 있는 U-Net[73]입니다. 인코더와 디코더는 각각 12개의 블록을 포함하며, 중간 블록을 포함한 전체 모델은 25개의 블록을 가지고 있습니다. 이 25개의 블록 중 8개는 다운샘플링 또는 업샘플링 컨볼루션 레이어이고, 나머지 17개 블록은 각각 4개의 레즈넷 레이어와 2개의 비전 트랜스포머(ViT)를 포함한 메인 블록입니다. 각 ViT는 여러 교차 주의와 자기 주의 메커니즘을 포함합니다. 예를 들어, 그림 3a에서 "SD 인코더 블록 A"는 4개의 레즈넷 레이어와 2개의 ViT를 포함하며, "×3"은 이 블록이 세 번 반복된다는 것을 나타냅니다. 텍스트 프롬프트는 CLIP 텍스트 인코더[66]를 사용하여 인코딩되며, 확산 시간 단계는 위치 인코딩을 사용하는 시간 인코더로 인코딩됩니다.

ControlNet 구조는 U-Net의 각 인코더 레벨에 적용됩니다 (그림 3b). 특히, 우리는 ControlNet을 사용하여 Stable Diffusion의 12개의 인코딩 블록과 1개의 중간 블록의 학습 가능한 복사본을 만듭니다. 12개의 인코딩 블록은 4가지 해상도(64 × 64, 32 × 32, 16 × 16, 8 × 8)로 구성되며, 각 해상도는 세 번 반복됩니다. 이 출력은 U-Net의 12개의 스킵 연결과 1개의 중간 블록에 추가됩니다. Stable Diffusion은 전형적인 U-Net 구조이므로, 이 ControlNet 아키텍처는 다른 모델에도 적용될 가능성이 있습니다.

ControlNet을 연결하는 방식은 계산 효율이 높습니다. 고정된 복사본 매개변수는 동결되어 있기 때문에, 미세 조정 시 원래 고정된 인코더에서는 그래디언트 계산이 필요하지 않습니다. 이 접근 방식은 학습 속도를 높이고 GPU 메모리를 절약합니다. 단일 NVIDIA A100 PCIE 40GB에서 테스트한 결과, ControlNet을 사용한 Stable Diffusion의 최적화는 ControlNet 없이 최적화한 것에 비해 각 학습 반복에서 약 23% 더 많은 GPU 메모리와 34% 더 많은 시간이 소요됩니다.

이미지 확산 모델은 이미지를 점진적으로 노이즈 제거하고 학습 도메인에서 샘플을 생성하는 방법을 학습합니다. 노이즈 제거 과정은 픽셀 공간 또는 학습 데이터에서 인코딩된 잠재 공간에서 발생할 수 있습니다. Stable Diffusion은 학습 과정의 안정화를 위해 잠재 이미지를 학습 도메인으로 사용합니다[72]. 구체적으로, Stable Diffusion은 VQ-GAN[19]과 유사한 사전 처리 방법을 사용하여 512 × 512 픽셀 공간 이미지를 더 작은 64 × 64 잠재 이미지로 변환합니다. Stable Diffusion에 ControlNet을 추가하려면, 먼저 각 입력 조건 이미지를(예: 에지, 자세, 깊이 등) 512 × 512의 입력 크기에서 Stable Diffusion의 크기에 맞는 64 × 64 특징 공간 벡터로 변환합니다. 특히, 네 가지 컨볼루션 레이어(4 × 4 커널, 2 × 2 스트라이드, 각각 16, 32, 64, 128 채널, ReLU로 활성화되고, 가우시안 가중치로 초기화되어 전체 모델과 함께 공동으로 학습)를 사용하여 이미지 공간 조건 cic\_ici​를 특징 공간 조건 벡터 cfc\_fcf​로 인코딩하는 작은 네트워크 E(⋅)E(\cdot)E(⋅)를 사용합니다.

![](/assets/images/posts/179/img_6.png)

조건 벡터 cfc\_fcf​는 ControlNet으로 전달됩니다.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Stable Diffusion 코드

```
import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder_blocks = nn.ModuleList([ResnetBlock(64) for _ in range(12)])
        self.middle_block = ResnetBlock(64)
        self.decoder_blocks = nn.ModuleList([ResnetBlock(64) for _ in range(12)])
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
        
        # Middle block
        x = self.middle_block(x)
        
        # Decoder
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = block(x + skip)
        
        return x

# Stable Diffusion 예제 실행
stable_diffusion = SimpleUNet()
input_tensor = torch.randn(1, 64, 64, 64)  # 예시 입력
output_tensor = stable_diffusion(input_tensor)
print(output_tensor.shape)
```

ControlNet 코드

```
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x):
        return self.conv(x)

class ControlNet(nn.Module):
    def __init__(self, stable_diffusion):
        super(ControlNet, self).__init__()
        self.stable_diffusion = stable_diffusion
        self.trainable_blocks = nn.ModuleList([ResnetBlock(64) for _ in range(13)])  # 12 encoding blocks + 1 middle block
        self.zero_convs = nn.ModuleList([ZeroConv2d(64, 64) for _ in range(13)])  # zero convs for each block

    def forward(self, x, condition):
        skip_connections = []
        # Encoder with ControlNet
        for i in range(12):
            x = self.stable_diffusion.encoder_blocks[i](x)
            condition_out = self.trainable_blocks[i](condition)
            condition_out = self.zero_convs[i](condition_out)
            x = x + condition_out
            skip_connections.append(x)
        
        # Middle block with ControlNet
        x = self.stable_diffusion.middle_block(x)
        condition_out = self.trainable_blocks[12](condition)
        condition_out = self.zero_convs[12](condition_out)
        x = x + condition_out
        
        # Decoder with skip connections
        for i in range(12):
            x = self.stable_diffusion.decoder_blocks[i](x + skip_connections[-(i+1)])
        
        return x

# ControlNet 예제 실행
control_net = ControlNet(stable_diffusion)
input_tensor = torch.randn(1, 64, 64, 64)  # 예시 입력
condition_tensor = torch.randn(1, 64, 64, 64)  # 예시 조건 입력
output_tensor = control_net(input_tensor, condition_tensor)
print(output_tensor.shape)
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

![](/assets/images/posts/179/img_7.png)

**그림 4:** 갑작스러운 수렴 현상. 제로 컨볼루션 덕분에 ControlNet은 전체 학습 과정 동안 항상 고품질 이미지를 예측합니다. 학습 과정의 특정 단계(예: 굵게 표시된 6133번째 단계)에서 모델이 갑자기 입력 조건을 따르기 시작합니다.

![](/assets/images/posts/179/img_8.png)

### 3.4 추론

우리는 추가 조건이 ControlNet의 확산 제거 과정에 어떻게 영향을 미치는지 여러 가지 방법으로 제어할 수 있습니다.

![](/assets/images/posts/179/img_9.png)

**그림 5:** 분류기 없는 지침(CFG)과 제안된 CFG 해상도 가중치(CFG-RW)의 효과.

![](/assets/images/posts/179/img_10.png)

**그림 6:** 여러 조건의 조합. 깊이와 자세를 동시에 사용하는 응용을 제시합니다.

![](/assets/images/posts/179/img_11.png)

### 여러 ControlNets 구성하기

여러 조건 이미지를(예: Canny 에지와 자세) Stable Diffusion의 단일 인스턴스에 적용하려면, 해당 ControlNets의 출력을 Stable Diffusion 모델에 직접 추가할 수 있습니다(그림 6). 이러한 구성에 추가 가중치나 선형 보간이 필요하지 않습니다.

![](/assets/images/posts/179/img_12.png)

**그림 7:** 프롬프트 없이 다양한 조건으로 Stable Diffusion 제어. 상단 행은 입력 조건이며, 나머지 행은 출력입니다. 입력 프롬프트로 빈 문자열을 사용합니다. 모든 모델은 일반 도메인 데이터로 학습되었습니다. 모델은 입력 조건 이미지에서 의미론적 내용을 인식하여 이미지를 생성해야 합니다.

### 4. 실험

우리는 Stable Diffusion과 함께 ControlNets를 구현하여 Canny Edge[11], Depth Map[69], Normal Map[87], M-LSD 라인[24], HED 소프트 에지[91], ADE20K 분할[96], Openpose[12], 사용자 스케치를 포함한 다양한 조건을 테스트했습니다. 각 조건의 예제와 자세한 학습 및 추론 매개변수는 추가 자료를 참조하십시오.

#### 4.1 정성적 결과

그림 1은 여러 프롬프트 설정에서 생성된 이미지를 보여줍니다. 그림 7은 프롬프트 없이 다양한 조건으로 우리의 결과를 보여줍니다. 여기서 ControlNet은 다양한 입력 조건 이미지에서 콘텐츠 의미론을 강력하게 해석합니다.

![](/assets/images/posts/179/img_13.png)

**표 1:** 결과 품질과 조건 충실도의 평균 사용자 순위(AUR). 다양한 방법에 대한 사용자 선호 순위를 보고합니다. (1에서 5는 최악에서 최고를 나타냅니다.)

#### 4.2 소거 연구

우리는 다음과 같은 대체 구조로 ControlNets를 연구합니다: (1) 제로 컨볼루션을 가우시안 가중치로 초기화된 표준 컨볼루션 레이어로 대체하고, (2) 각 블록의 학습 가능한 복사본을 단일 컨볼루션 레이어로 대체한 ControlNet-lite. 이러한 소거 구조의 전체 세부 사항은 추가 자료를 참조하십시오.

우리는 실제 사용자의 가능한 행동을 테스트하기 위해 4가지 프롬프트 설정을 제시합니다: (1) 프롬프트 없음; (2) 조건 이미지를 완전히 설명하지 않는 불충분한 프롬프트, 예를 들어 이 논문의 기본 프롬프트 "고품질의 세밀하고 전문적인 이미지"; (3) 조건 이미지의 의미를 변경하는 충돌하는 프롬프트; (4) 필요한 콘텐츠 의미를 설명하는 완벽한 프롬프트, 예를 들어 "아름다운 집". 그림 8a는 ControlNet이 모든 4가지 설정에서 성공하는 것을 보여줍니다. 경량 ControlNet-lite(그림 8c)는 조건 이미지를 해석하기에 충분히 강력하지 않으며, 불충분한 프롬프트와 프롬프트가 없는 조건에서 실패합니다. 제로 컨볼루션이 교체되면 ControlNet의 성능이 ControlNet-lite와 거의 동일하게 떨어지며, 이는 학습 가능한 복사본의 사전 학습된 백본이 미세 조정 중에 파괴됨을 나타냅니다(그림 8b).

![](/assets/images/posts/179/img_14.png)

그림 8: 스케치 조건과 다양한 프롬프트 설정에 대한 다양한 아키텍처에 대한 추상적 연구. 각 설정에 대해 체리피킹 없이 6개의 샘플을 무작위로 배치하여 보여줍니다. 이미지는 512 × 512이며 확대했을 때 가장 잘 보입니다. 왼쪽의 녹색 'conv' 블록은 가우시안 가중치로 초기화된 표준 컨볼루션 레이어입니다.

### 4.3 정량적 평가

#### 사용자 연구

우리는 보지 못한 손으로 그린 스케치 20개를 샘플링하고, 각 스케치를 5가지 방법으로 배정했습니다: PITI [89]의 스케치 모델, 기본 엣지 가이드 스케일(β = 1.6)의 Sketch-Guided Diffusion(SGD) [88], 상대적으로 높은 엣지 가이드 스케일(β = 3.2)의 SGD [88], 앞서 언급한 ControlNet-lite, 그리고 ControlNet. 우리는 12명의 사용자에게 이 20개의 그룹에 대한 5개의 결과를 개별적으로 "표시된 이미지의 품질"과 "스케치에 대한 충실도"에 대해 순위를 매기도록 요청했습니다. 이 방식으로 결과 품질에 대한 100개의 순위와 조건 충실도에 대한 100개의 순위를 얻었습니다. 우리는 사용자가 각 결과를 1에서 5까지의 척도로 순위를 매기는 선호 지표로 Average Human Ranking(AHR)을 사용했습니다(낮을수록 나쁨). 평균 순위는 표 1에 나타나 있습니다.

#### 산업 모델과의 비교

Stable Diffusion V2 Depth-to-Image(SDv2-D2I) [83]는 대규모 NVIDIA A100 클러스터, 수천 시간의 GPU 시간, 1200만 개 이상의 학습 이미지를 사용하여 학습되었습니다. 우리는 동일한 깊이 조건을 사용하여 SD V2를 위해 ControlNet을 학습했지만, 단지 200,000개의 학습 샘플, 단일 NVIDIA RTX 3090Ti, 5일간의 학습만 사용했습니다. 우리는 각 SDv2-D2I와 ControlNet으로 생성된 100개의 이미지를 사용하여 12명의 사용자가 두 방법을 구별하도록 가르쳤습니다. 이후 200개의 이미지를 생성하여 사용자가 각 이미지를 어느 모델이 생성했는지 맞추도록 요청했습니다. 사용자의 평균 정확도는 0.52 ± 0.17로, 두 방법의 결과가 거의 구별되지 않음을 나타냅니다.

![](/assets/images/posts/179/img_15.png)

표 2: 의미론적 분할 레이블 재구성 평가(ADE20K)와 교차합집합(IoU ↑).

#### 조건 재구성과 FID 점수

우리는 ADE20K [96]의 테스트 세트를 사용하여 조건 충실도를 평가합니다. 최첨단 분할 방법인 OneFormer [35]는 원본 세트에서 0.58의 교차합집합(IoU)을 달성합니다. 우리는 다양한 방법을 사용하여 ADE20K 분할로 이미지를 생성한 다음 OneFormer를 적용하여 다시 분할을 검출하여 재구성된 IoU를 계산합니다(표 2). 또한, 우리는 Frechet Inception Distance(FID) [28]를 사용하여 다양한 분할 조건 방법을 사용하여 무작위로 생성된 512×512 이미지 세트에 대한 분포 거리를 측정하고, 텍스트-이미지 CLIP 점수 [66] 및 CLIP 미적 점수 [79]를 표 3에 보고합니다. 자세한 설정은 추가 자료를 참조하십시오.

![](/assets/images/posts/179/img_16.png)

표 3: 의미론적 분할로 조건화된 이미지 생성 평가. 우리는 우리의 방법과 다른 기준선에 대한 FID, CLIP 텍스트-이미지 점수 및 CLIP 미적 점수를 보고합니다. 우리는 또한 분할 조건 없이 Stable Diffusion의 성능을 보고합니다. "\*"로 표시된 방법은 처음부터 학습되었습니다.

![](/assets/images/posts/179/img_17.png)

그림 9: 이전 방법과의 비교. 우리는 PITI [89], Sketch-Guided Diffusion [88], 그리고 Taming Transformers [19]와의 정성적 비교를 제시합니다.

### 4.4. 이전 방법들과의 비교

그림 9는 기준선과 우리의 방법(Stable Diffusion + ControlNet)의 시각적 비교를 보여줍니다. 특히, 우리는 PITI [89], Sketch-Guided Diffusion [88], 그리고 Taming Transformers [19]의 결과를 보여줍니다. (참고로 PITI의 백본은 다른 시각적 품질과 성능을 가지는 OpenAI GLIDE [57]입니다.) 우리는 ControlNet이 다양한 조건 이미지를 견고하게 처리하고 선명하고 깨끗한 결과를 달성하는 것을 관찰했습니다.

![](/assets/images/posts/179/img_18.png)

#### 그림 10: 다양한 학습 데이터 세트 크기의 영향. 확장된 예제는 추가 자료를 참조하십시오.

### 4.5. 논의

#### 학습 데이터셋 크기의 영향

그림 10에서 ControlNet 학습의 견고성을 보여줍니다. 학습은 1,000개의 제한된 이미지로도 무너지지 않으며, 모델이 알아볼 수 있는 사자를 생성할 수 있도록 합니다. 더 많은 데이터가 제공될 때 학습은 확장 가능합니다.

![](/assets/images/posts/179/img_19.png)

#### 그림 11: 콘텐츠 해석. 입력이 모호하고 사용자가 프롬프트에서 객체 콘텐츠를 언급하지 않으면, 결과는 모델이 입력 형상을 해석하려고 시도하는 것처럼 보입니다.

#### 콘텐츠 해석 능력

그림 11에서 ControlNet이 입력 조건 이미지에서 의미론을 포착하는 능력을 보여줍니다.

![](/assets/images/posts/179/img_20.png)

#### 그림 12: 사전 학습된 ControlNets를 추가 학습 없이 커뮤니티 모델 [16, 61]로 이전합니다.

#### 커뮤니티 모델로의 전이

ControlNet은 사전 학습된 Stable Diffusion 모델의 네트워크 토폴로지를 변경하지 않기 때문에, Comic Diffusion [61]과 Protogen 3.4 [16]와 같은 다양한 Stable Diffusion 커뮤니티 모델에 직접 적용될 수 있습니다 (그림 12 참조).

### 5. 결론

ControlNet은 대규모 사전 학습된 텍스트-이미지 확산 모델을 위한 조건 제어를 학습하는 신경망 구조입니다. 소스 모델의 대규모 사전 학습된 레이어를 재사용하여 특정 조건을 학습하기 위한 깊고 강력한 인코더를 구축합니다. 원래 모델과 학습 가능한 복사본은 학습 중 유해한 노이즈를 제거하는 "제로 컨볼루션" 레이어를 통해 연결됩니다. 광범위한 실험을 통해 ControlNet이 단일 또는 다중 조건으로, 프롬프트 유무에 관계없이 Stable Diffusion을 효과적으로 제어할 수 있음을 검증했습니다. 다양한 조건 데이터셋에서의 결과는 ControlNet 구조가 더 넓은 범위의 조건에 적용될 가능성이 높고, 관련 응용을 촉진할 수 있음을 보여줍니다.

### 감사의 글

이 연구는 부분적으로 Stanford Institute for Human-Centered AI와 Brown Institute for Media Innovation의 지원을 받았습니다.

[2302.05543v3.pdf

15.59MB](./file/2302.05543v3.pdf)
