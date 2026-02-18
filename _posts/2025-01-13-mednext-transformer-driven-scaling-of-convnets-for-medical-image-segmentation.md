---
title: "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation"
date: 2025-01-13 15:44:46
categories:
  - 인공지능
tags:
  - mednext
---

<https://arxiv.org/abs/2303.09975>

[MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation](https://arxiv.org/abs/2303.09975)

<https://github.com/MIC-DKFZ/MedNeXt/tree/0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba>

[GitHub - MIC-DKFZ/MedNeXt: [MICCAI 2023] MedNeXt is a fully ConvNeXt architecture for 3D medical image segmentation.](https://github.com/MIC-DKFZ/MedNeXt/tree/0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba)

**초록(Abstract)**  
의료 영상 분할을 위해 트랜스포머(Transformer) 기반 아키텍처를 적용하려는 관심이 최근 폭발적으로 늘어나고 있다. 그러나 대규모로 주석된 의료 데이터셋이 부족하여, 자연 이미지 분야와 유사한 수준의 성능을 내기에는 여전히 어려움이 있다. 반면 합성곱 신경망(Convolutional Neural Network)은 더 높은 귀납적 편향(inductive bias)을 지니고 있어, 상대적으로 간단한 학습 과정으로도 높은 성능을 달성하기 쉽다. 최근에 발표된 ConvNeXt 아키텍처는 트랜스포머 블록을 반영함으로써 전통적인 ConvNet을 현대적으로 재설계하려는 시도를 보여주었다.

본 연구에서는 이를 한 단계 발전시켜, 데이터가 부족한 의료 환경의 특수성을 고려한 확장 가능하고 현대화된 합성곱 기반 아키텍처를 설계한다. 구체적으로, 우리는 트랜스포머로부터 영감을 받은 대형 커널(large kernel) 분할 네트워크인 **MedNeXt**를 제안하며, 다음과 같은 주요 기여점을 갖는다.

1. **3D ConvNeXt 기반 인코더-디코더(Encoder-Decoder) 네트워크**: 의료 영상 분할을 위해 완전히 ConvNeXt 구조로 구성된 3D 인코더-디코더를 제안한다.
2. **Residual ConvNeXt 업샘플링·다운샘플링 블록**: 다중 스케일에서의 의미 정보를 풍부하게 유지하기 위해 잔차(residual) 구조를 적용한 ConvNeXt 기반 업·다운샘플링 블록을 도입한다.
3. **점진적 커널 확장 기법**: 한정된 의료 데이터로 인한 성능 포화를 방지하기 위해, 작은 커널 크기를 갖는 네트워크를 업샘플링하여 커널 크기를 단계적으로 키우는 새로운 학습 전략을 제안한다.
4. **복합적 스케일링(Compound Scaling)**: MedNeXt의 깊이(depth), 너비(width), 커널 크기(kernel size) 등 여러 측면에서 동시에 확장하는 복합 스케일링 방식을 채택한다.

제안된 MedNeXt 모델은 CT와 MRI 등 서로 다른 영상 기법, 다양한 데이터셋 크기를 포함한 네 가지 과제에서 최신 수준(state-of-the-art)의 성능을 달성하였다. 이는 의료 영상 분할을 위한 현대화된 딥러닝 아키텍처로서의 가능성을 보여준다. 해당 연구의 코드는 다음 링크에서 공개되어 있다: <https://github.com/MIC-DKFZ/MedNeXt>.

**키워드(Keywords)**: 의료 영상 분할, 트랜스포머, MedNeXt, 대형 커널(large kernels), ConvNeXt

**1 서론(Introduction)**  
트랜스포머(Transformers) [30, 7, 21]는 의료 영상 분할 분야에서, 하이브리드 아키텍처의 일부 [3, 9, 33, 2, 8, 31] 혹은 독립적인 기법 자체 [34, 25, 15]로 활용되어 최신 성능(state-of-the-art)을 달성하는 사례가 급격히 늘어나고 있다. 트랜스포머가 시각적 과업에서 제공하는 주요 장점 중 하나는 장거리(long-range) 공간적 종속성을 학습할 수 있다는 점이다. 그러나 트랜스포머는 본질적으로 귀납적 편향(inductive bias)이 제한적이기 때문에, 대규모 주석(annotated) 데이터셋이 필요하다는 문제가 있다. 자연 이미지 분야에는 대규모 데이터셋(ImageNet-1k [6], ImageNet-21k [26])이 풍부하지만, 의료 영상 분야는 일반적으로 고품질 주석 데이터의 부족으로 어려움을 겪는다 [19].

이를 해결하기 위해, 합성곱(Convolution) 본연의 귀납적 편향을 유지하면서도 트랜스포머 아키텍처의 이점을 동시에 누릴 수 있는 방법이 모색되고 있다. 최근 ConvNeXt [22]가 제안되어, 자연 이미지 분류 분야에서 합성곱 기반 네트워크(ConvNet)가 트랜스포머에 버금가는 경쟁력을 다시 확보할 수 있음을 입증하였다. ConvNeXt는 트랜스포머와 유사하게 깊이별 합성곱(depthwise convolution), 채널 확장 층(expansion layer), 채널 축소 층(contraction layer)으로 구성된 인버티드 보틀넥(inverted bottleneck) 구조를 채택한다(섹션 2.1). 여기에 더해, 큰 커널(large depthwise kernels)을 통해 트랜스포머가 갖는 확장성 및 장거리 표현 학습 능력을 모방한다. ConvNeXt 저자들은 방대한 데이터셋과 결합된 큰 커널 ConvNeXt 네트워크를 활용해, 기존에 제시된 트랜스포머 기반 네트워크보다 뛰어난 성능을 보고하였다.

반면, 의료 영상 분할 분야에서는 VGGNet [28]이 제시한 방식인 작은 커널을 여러 층으로 쌓는 전통적인 합성곱 설계가 여전히 주류를 이루고 있다. 범용적이고 데이터 효율적인 기법인 nnUNet [13] 또한 표준적인 UNet [5]의 변형을 사용하며, 폭넓은 과업에서 여전히 뛰어난 성능을 보이고 있다.

ConvNeXt 아키텍처는 Vision Transformer [7]와 Swin Transformer [21]의 확장 가능성(scalability)과 장거리 공간 표현 학습 능력을, ConvNet의 본질적 귀납적 편향과 결합한 형태다. 특히 인버티드 보틀넥 디자인은 커널 크기에 구애받지 않고 너비(채널 수)를 확장할 수 있게 해준다. 의료 영상 분할에서 이를 적극적으로 활용할 경우, 1) 큰 커널을 통해 장거리 공간 정보를 학습할 수 있고, 2) 여러 네트워크 차원을 동시에 확장(scale)할 수 있다는 장점이 있다. 하지만, 이러한 대형 네트워크는 제한된 의료 데이터로 학습할 때 과적합(overfitting)이 발생하기 쉽기 때문에, 이를 방지할 기법이 필요하다.

---

ConvNeXt가 “Vision Transformer와 Swin Transformer의 확장 가능성(scalability)과 장거리 공간 표현 학습 능력을, 동시에 ConvNet의 귀납적 편향과 결합했다”는 말은 다음과 같은 맥락을 담고 있습니다:

1. **트랜스포머의 특성**
   - **장거리(long-range) 관계 학습**: 트랜스포머는 self-attention 메커니즘을 통해 입력 전체에서 중요한 정보들을 주고받으면서, 매우 먼 거리의 특징 간 상호 작용도 쉽게 학습할 수 있습니다.
   - **우수한 확장성(scalability)**: Vision Transformer(ViT)나 Swin Transformer 등은 모델 규모(레이어 수, 채널 수 등)를 매우 크게 확장해도 비교적 안정적으로 학습이 가능합니다. 즉, 데이터나 모델 크기를 늘릴 때 적절한 하드웨어 환경이 주어지면 성능이 더 좋아지는 경향이 있습니다.
2. **CNN(ConvNet)의 ‘본질적 귀납적 편향(inductive bias)’**
   - CNN은 이미지 같은 2차원 입력에서 ‘지역성(locality)’을 자연스럽게 가정합니다. 즉, 인접 픽셀들끼리는 관련성이 높을 것이라는 가정이 있으며, 필터를 이동시키며 동일한 연산을 반복(파라미터 공유)하기 때문에 **지역적 특징 탐색에 최적화**되어 있습니다.
   - 이러한 귀납적 편향 덕분에, 상대적으로 적은 양의 데이터로도 학습을 잘해낼 수 있고, 시각적으로 의미 있는 에지(edge)나 영역 등의 지역 패턴을 빠르게 잡아낼 수 있습니다.
3. **ConvNeXt에서의 결합 아이디어**
   - ConvNeXt는 CNN을 기반으로 하지만, **인버티드 보틀넥(inverted bottleneck) 구조**, **깊이별 합성곱(depthwise convolution)**, **큰 커널(large kernel) 적용** 등 트랜스포머 블록에서 착안한 설계를 활용합니다.
   - 이를 통해 **장거리 공간 정보를 좀 더 폭넓게 학습**할 수 있도록 수용영역(receptive field)을 늘리고, 네트워크 규모(깊이, 너비 등)도 원활히 확장할 수 있도록 했습니다.
   - 동시에 CNN 특유의 **지역 패턴 학습 능력**, **적은 데이터로도 학습 성능이 안정적**이라는 장점을 최대한 살렸습니다.

정리하자면,

- **트랜스포머의 장점**: 이미지를 전역적(global)으로 바라보면서 네트워크를 크게 키워도 학습이 잘 된다(확장성).
- **CNN의 장점**: 지역적 특성을 빠르게 잡아내는 데 특화되어 있고, 데이터가 풍부하지 않아도 비교적 안정적으로 학습이 가능하다.

이 두 가지가 절묘하게 융합된 것이 ConvNeXt이며, “트랜스포머의 확장성과 장거리 표현 학습” + “CNN의 본질적 귀납적 편향”이 서로 보완적인 강점으로 작용한다는 의미입니다.

---
