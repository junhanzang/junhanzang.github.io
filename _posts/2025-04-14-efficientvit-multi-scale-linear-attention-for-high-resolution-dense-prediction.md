---
title: "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction"
date: 2025-04-14 17:21:17
categories:
  - 인공지능
tags:
  - efficientvit
---

<https://arxiv.org/abs/2205.14756>

[EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](https://arxiv.org/abs/2205.14756)

**초록**  
고해상도 밀집 예측(high-resolution dense prediction)은 계산 사진학(computational photography), 자율 주행(autonomous driving) 등 다양한 실세계 응용에 매력적인 가능성을 제공합니다. 그러나 최신 고해상도 밀집 예측 모델은 막대한 연산 비용으로 인해 실제 하드웨어에 배포하기 어렵습니다. 본 논문에서는 **EfficientViT**라는 새로운 고해상도 비전 모델 계열을 제안하며, **다중 스케일 선형 어텐션(multi-scale linear attention)**이라는 새로운 기법을 도입합니다. 기존 고해상도 밀집 예측 모델들이 높은 성능을 달성하기 위해 무거운 softmax 어텐션, 하드웨어 비효율적인 대형 커널 합성곱, 복잡한 토폴로지 구조에 의존한 반면, **EfficientViT의 다중 스케일 선형 어텐션은 경량이면서도 하드웨어 효율적인 연산만으로 글로벌 수용 영역(global receptive field)과 다중 스케일 학습(multi-scale learning)을 모두 달성**합니다.

이러한 특성 덕분에 EfficientViT는 모바일 CPU, 엣지 GPU, 클라우드 GPU 등 다양한 하드웨어 플랫폼에서 기존 최고 성능 모델 대비 **획기적인 속도 향상과 성능 향상**을 동시에 제공합니다.

- **Cityscapes** 데이터셋에서는 성능 손실 없이 SegFormer 대비 최대 **13.9배**, SegNeXt 대비 **6.2배** GPU 지연 시간(latency)을 줄였습니다.
- **초해상도(super-resolution)** 작업에서는 Restormer 대비 최대 **6.4배** 속도 향상과 함께 **PSNR 0.11dB 증가**를 달성했습니다.
- **Segment Anything** 작업에서는 A100 GPU에서 **48.9배 높은 처리량(throughput)**을 기록하면서도 COCO 데이터셋에서 **제로샷 인스턴스 분할(zero-shot instance segmentation)** 성능을 소폭 상회하였습니다.

### 1. 서론 (Introduction)

고해상도 밀집 예측(high-resolution dense prediction)은 컴퓨터 비전에서 근본적인 과제로, 자율 주행, 의료 영상 처리, 계산 사진학(computational photography) 등 다양한 실제 분야에서 널리 활용됩니다. 따라서, 최신 고해상도 밀집 예측 모델(state-of-the-art, 이하 SOTA 모델)을 하드웨어 장치에 배포할 수 있다면 많은 응용 분야에서 실질적인 이점을 얻을 수 있습니다.

하지만 이러한 SOTA 모델들이 요구하는 막대한 연산량은 실제 하드웨어 장치의 한정된 자원과 큰 격차를 보이며, 이는 현실적인 응용을 어렵게 만듭니다. 특히, 고해상도 밀집 예측 모델은 **고해상도 입력 이미지**와 **강력한 문맥 정보 추출 능력**이 필요합니다 [1~6]. 따라서 기존의 이미지 분류용 효율적인 모델 구조를 단순히 가져다 쓰는 것은 적절하지 않습니다.

![](/assets/images/posts/536/img.png)

**[그림 1: Latency/Throughput vs. Performance]**  
다양한 하드웨어 플랫폼(Jetson AGX Orin, A100 GPU)에서 TensorRT와 FP16으로 추론 시, EfficientViT는 기존 분할/분류 모델 대비 뛰어난 성능을 유지하면서도 지속적으로 향상된 속도를 보여줍니다.

본 연구에서는 고해상도 밀집 예측을 위한 새로운 비전 트랜스포머 계열 모델인 **EfficientViT**를 제안합니다. EfficientViT의 핵심은, 하드웨어 효율적인 연산만으로도 **글로벌 수용 영역(global receptive field)**과 **다중 스케일 학습(multi-scale learning)**을 가능하게 하는 새로운 **다중 스케일 선형 어텐션(multi-scale linear attention)** 모듈입니다.

기존의 SOTA 모델들은 다중 스케일 학습 [3, 4]과 글로벌 수용 영역 [7]이 성능 향상에 핵심이라는 사실을 보여주었지만, 실제 하드웨어 효율성은 고려하지 않았습니다.  
예를 들어, **SegFormer** [7]는 softmax 어텐션 [8]을 백본(backbone)에 적용하여 글로벌 수용 영역을 확보하였지만, 연산 복잡도가 입력 해상도에 대해 **이차적(quadratic)**으로 증가하여 고해상도 이미지를 처리하기 어렵습니다.  
또한 **SegNeXt** [9]는 커널 크기 최대 21의 대형 커널 합성곱을 다중 브랜치 모듈로 도입했지만, 이는 하드웨어에서 최적화되기 어려운 연산이어서 실제 장치에서 비효율적입니다 [10, 11].

따라서, 우리는 이러한 두 가지 핵심 요소(글로벌 수용 영역, 다중 스케일 학습)를 지원하면서도 하드웨어에 비효율적인 연산은 피하는 방향으로 모듈을 설계하였습니다.  
구체적으로, **softmax 어텐션을 경량화된 ReLU 기반 선형 어텐션(ReLU linear attention)**으로 대체하여 글로벌 수용 영역을 확보합니다.  
ReLU 선형 어텐션은 행렬곱의 결합 법칙(associative property)을 활용하여 연산 복잡도를 **이차 → 선형(linear)**으로 줄일 수 있으며, softmax와 같은 하드웨어 비우호적 연산을 제거해 하드웨어 배포에 적합합니다 (**그림 4** 참조).

하지만 ReLU 선형 어텐션만으로는 **국지적(local) 정보 추출**과 **다중 스케일 학습**이 부족해 성능에 한계가 있습니다. 이를 보완하기 위해,

- 우리는 **소형 커널 합성곱**을 통해 인접 토큰들을 집계하여 **다중 스케일 토큰(multi-scale tokens)**을 생성하고,
- 여기에 ReLU 선형 어텐션을 적용해 **글로벌 수용 영역 + 다중 스케일 학습**을 결합합니다 (**그림 2** 참조).
- 또한 FFN(feed-forward network) 레이어에 **depthwise convolution**을 삽입하여 국지적 정보 추출 능력을 강화했습니다.

우리는 EfficientViT를 두 가지 대표적인 고해상도 밀집 예측 과제, 즉 **시맨틱 분할(semantic segmentation)**과 **초해상도(super-resolution)**에 대해 폭넓게 평가하였습니다. 그 결과, 기존 SOTA 모델 대비 **성능 향상**은 물론, **하드웨어 비효율 연산의 배제** 덕분에 연산량(FLOPs) 감소가 실제 하드웨어 지연 시간(latency) 감소로 이어졌습니다 (**그림 1** 참조).

또한, 우리는 EfficientViT를 최근 주목받는 **Segment Anything** [13]에도 적용해 보았습니다. 이 작업은 다양한 비전 과제에 제로샷 전이(zero-shot transfer)를 가능하게 하는 prompt 기반 분할 과제입니다. EfficientViT는 **SAM-ViT-Huge** [13] 대비 **A100 GPU에서 48.9배 빠른 속도**를 보이면서도 성능 저하 없이 작업을 수행했습니다.

### 본 논문의 주요 기여는 다음과 같습니다:

- **다중 스케일 선형 어텐션 모듈**을 새롭게 도입하여, 글로벌 수용 영역과 다중 스케일 학습을 모두 달성하면서도 하드웨어 효율성을 유지했습니다. 본 연구는 **선형 어텐션을 고해상도 밀집 예측에 효과적으로 적용한 최초의 사례**입니다.
- 제안한 모듈을 바탕으로 한 새로운 고해상도 비전 모델 계열인 **EfficientViT**를 설계하였습니다.
- EfficientViT는 시맨틱 분할, 초해상도, Segment Anything, ImageNet 분류 등 다양한 과제에서 **기존 SOTA 모델 대비 획기적인 속도 향상**을 달성했습니다. 또한 **모바일 CPU, 엣지 GPU, 클라우드 GPU 등 다양한 하드웨어 플랫폼**에서 우수한 성능을 보였습니다.

![](/assets/images/posts/536/img_1.png)

**[그림 2: EfficientViT의 구성 블록 및 다중 스케일 선형 어텐션]**  
좌측: EfficientViT는 다중 스케일 선형 어텐션 + depthwise convolution이 결합된 FFN으로 구성됩니다.  
우측: Q/K/V 토큰을 선형 투영 후, 소형 커널 합성곱으로 다중 스케일 토큰을 만들고, 각 스케일에 대해 ReLU 선형 어텐션을 수행한 후 결과를 결합합니다.

![](/assets/images/posts/536/img_2.png)

**[그림 3: Softmax 어텐션 vs. ReLU 선형 어텐션]**  
ReLU 선형 어텐션은 비선형 유사도 함수가 없어 날카로운 주의 분포(sharp attention)를 만들지 못해, softmax 어텐션보다 국지 정보 추출력이 약할 수 있습니다.

![](/assets/images/posts/536/img_3.png)

**[그림 4: Softmax vs. ReLU 어텐션의 지연 시간 비교]**  
ReLU 선형 어텐션은 softmax보다 **3.3~4.5배 빠른 속도**를 기록하였으며, Qualcomm Snapdragon 855 CPU + TensorFlow Lite에서 실측하였습니다.

![](/assets/images/posts/536/img_4.png)

**[그림 5: EfficientViT의 전체 구조]**  
표준 backbone-head (encoder-decoder) 구조를 따르며, EfficientViT 모듈은 stage 3, 4에 삽입됩니다. 마지막 세 단계(P2, P3, P4)의 피처들을 단순히 더해(head) 출력으로 활용합니다.

---

![](/assets/images/posts/536/img_5.png)

![](/assets/images/posts/536/img_6.png)

![](/assets/images/posts/536/img_7.png)

---
