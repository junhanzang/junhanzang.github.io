---
title: "LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
date: 2026-01-23 18:13:40
categories:
  - 인공지능
---

<https://arxiv.org/abs/2406.03459>

[LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection](https://arxiv.org/abs/2406.03459)

## LW-DETR: 실시간 객체 검출을 위한 YOLO 대체 Transformer

**Qiang Chen**  
(동등 기여, † 교신저자)

**Xiangbo Su**, **Xinyu Zhang**, **Jian Wang**, **Jiahui Chen**, **Yunpeng Shen**,  
**Chuchu Han**, **Ziliang Chen**, **Weixiang Xu**, **Fanrong Li**, **Shan Zhang**,  
**Kun Yao**, **Errui Ding**, **Gang Zhang**, **Jingdong Wang**

## 초록 (Abstract)

본 논문에서는 실시간 객체 검출(real-time object detection)을 위해 YOLO 계열을 능가하는 경량 검출 트랜스포머 **LW-DETR**를 제안한다. 제안하는 아키텍처는 **ViT 인코더(ViT encoder)**, **프로젝터(projector)**, 그리고 **얕은(shallow) DETR 디코더**로 구성된 단순한 스택 구조를 가진다.

본 접근법은 최근의 다양한 고급 기법들을 효과적으로 활용한다. 구체적으로는 개선된 손실 함수와 사전학습(pretraining)과 같은 **학습 효율 향상 기법(training-effective techniques)**을 적용하고, ViT 인코더의 연산 복잡도를 줄이기 위해 **윈도우 기반(window) 어텐션과 글로벌 어텐션을 교차(interleaved)로 사용하는 구조**를 도입한다.

또한 ViT 인코더에서 **다중 수준(multi-level)의 feature map**을 집계하고, 인코더 내부의 중간(feature maps at intermediate layers) 및 최종 feature map을 함께 결합함으로써 보다 풍부한 표현력을 갖는 feature map을 형성한다. 더 나아가, **window-major 방식의 feature map 구성**을 제안하여 interleaved attention 연산의 효율을 향상시킨다.

실험 결과, 제안한 방법은 COCO 및 기타 벤치마크 데이터셋에서 YOLO 및 그 변형 모델을 포함한 기존 실시간 객체 검출기들을 전반적으로 능가하는 성능을 보임을 확인하였다. 코드와 사전 학습된 모델은 다음 링크에서 공개되어 있다.  
<https://github.com/Atten4Vis/LW-DETR>

[GitHub - Atten4Vis/LW-DETR: This repository is an official implementation of the paper "LW-DETR: A Transformer Replacement to YO](https://github.com/Atten4Vis/LW-DETR)

## 키워드 (Keywords)

객체 검출(Object Detection), 실시간 검출(Real-Time Detection), 트랜스포머(Transformer)

![](/assets/images/posts/620/img.png)

## 그림 1 설명 (Figure 1)

제안하는 방법은 기존의 최신 실시간 객체 검출기(SoTA)를 능가하는 성능을 보인다. x축은 **추론 시간(inference time)**을, y축은 **COCO val2017 기준 mAP 점수**를 나타낸다. 모든 모델은 Objects365 데이터셋으로 사전학습(pretraining)된 상태에서 학습되었다. YOLO, RTMDet 등의 기존 모델에 대해서는 NMS 후처리 시간이 포함되어 있으며, 이는 공식 구현 설정을 기반으로 COCO val2017에서 측정되었다. 또한, 잘 튜닝된 NMS 후처리 설정을 적용한 경우는 “\*”로 표시하였다.

## 2. 관련 연구 (Related Work)

### 실시간 객체 검출 (Real-time object detection)

실시간 객체 검출은 다양한 실제 응용 분야에서 폭넓게 활용되고 있다 [31, 18, 25, 30, 49].  
YOLO-NAS [1], YOLOv8 [29], RTMDet [46]와 같은 최신(state-of-the-art) 실시간 객체 검출기들은 초기 YOLO [50]에 비해 크게 발전하였으며, 그 배경에는 검출 프레임워크의 개선 [56, 20], 아키텍처 설계 [2, 33, 32, 65, 16, 59], 데이터 증강 기법 [69, 2, 20], 학습 기법 [20, 29, 1], 그리고 손실 함수 설계 [38, 68, 73] 등이 있다.

이러한 기존 실시간 검출기들은 대부분 **합성곱(convolution)** 기반 구조에 의존한다. 반면, 본 논문에서는 아직 충분히 탐구되지 않은 **트랜스포머 기반(transformer-based)** 실시간 객체 검출 해법을 연구 대상으로 삼는다.

### 객체 검출을 위한 ViT (ViT for object detection)

Vision Transformer(ViT) [17, 66]는 이미지 분류 분야에서 뛰어난 성능을 보였다. ViT를 객체 검출에 적용할 경우, 메모리 사용량과 연산 비용을 줄이기 위해 일반적으로 **윈도우 어텐션(window attention)** [17, 43]이나 **계층적(hierarchical) 아키텍처** [62, 43, 70, 21]가 활용된다.

UViT [8]는 점진적 윈도우 어텐션(progressive window attention)을 사용하며, ViTDet [36]는 사전 학습된 plain ViT에 **interleaved window 및 global attention** [37]을 적용한다. 본 연구의 접근법은 ViTDet을 따라 interleaved window 및 global attention을 사용하되, 추가적으로 **window-major 순서의 feature map 구성 방식**을 도입하여 메모리 재배치(permutation)에 따른 비용을 줄인다.

### DETR 및 그 변형들 (DETR and its variants)

Detection Transformer(DETR)는 **anchor 생성** [51]이나 **비최대 억제(non-maximum suppression, NMS)** [24]와 같은 다수의 수작업(hand-crafted) 구성 요소를 제거한 **end-to-end 객체 검출 방법**이다. 이후 DETR을 개선하기 위한 다양한 후속 연구들이 제안되었으며, 여기에는 아키텍처 설계 [74, 47, 19, 67], object query 설계 [63, 41, 11], 학습 기법 [35, 67, 6, 27, 48, 7, 75], 그리고 손실 함수 개선 [42, 3] 등이 포함된다.

또한 연산 복잡도를 줄이기 위해 아키텍처 설계 [40, 34], 연산 최적화 [74], 가지치기(pruning) [53, 72], 지식 증류(distillation) [5, 9, 5, 64] 등 다양한 시도가 이루어졌다. 본 논문의 관심사는 이러한 기존 방법들이 충분히 탐구하지 않은 영역, 즉 **실시간 객체 검출을 위한 단순한(simple) DETR 기반 베이스라인**을 구축하는 데 있다.

### 동시 연구 (Concurrent work)

본 연구와 거의 동시에, RT-DETR [45] 역시 DETR 프레임워크를 실시간 객체 검출에 적용하였으며, 특히 **CNN 백본을 인코더로 사용하는 구조**에 초점을 맞추었다. 다만, 상대적으로 대형 모델 위주의 연구가 이루어졌고, **초소형(tiny) 모델에 대한 연구는 부족한 편**이다. 이에 비해, 본 논문의 **LW-DETR**는 **plain ViT 백본과 DETR 프레임워크가 실시간 객체 검출에 적용 가능한지**를 탐구한다는 점에서 차별성을 가진다.

![](/assets/images/posts/620/img_1.png)

### 그림 2 설명 (Figure 2)

다중 수준 feature map 집계와 interleaved window 및 global attention을 사용하는 트랜스포머 인코더의 예시를 보여준다. 설명의 명확성을 위해 FFN과 LayerNorm 계층은 그림에서 생략되었다.

## 3. LW-DETR

### 3.1 아키텍처 (Architecture)

LW-DETR는 **ViT 인코더(ViT encoder)**, **프로젝터(projector)**, 그리고 **DETR 디코더(DETR decoder)**로 구성된다.

### 인코더 (Encoder)

본 연구에서는 객체 검출용 인코더로 **ViT**를 채택한다. 일반적인 plain ViT [17]는 패치화(patchification) 계층과 다수의 트랜스포머 인코더 계층으로 구성된다. 초기 ViT의 트랜스포머 인코더 계층은 모든 토큰에 대해 **글로벌 자기어텐션(global self-attention)**과 FFN 계층을 포함한다. 그러나 글로벌 자기어텐션은 토큰(패치) 수에 대해 **이차(quadratic)** 시간 복잡도를 가지므로 계산 비용이 매우 크다.

이를 완화하기 위해, 우리는 일부 트랜스포머 인코더 계층에서 **윈도우 자기어텐션(window self-attention)**을 적용하여 계산 복잡도를 줄인다(자세한 내용은 3.4절 참조). 또한 인코더 내부의 **중간 계층(feature maps at intermediate layers)**과 **최종 계층의 feature map**, 그리고 **다중 수준(multi-level) feature map**을 집계(aggregation)하여, 보다 강력한 인코딩 feature map을 형성한다. 이러한 인코더 구조의 예시는 그림 2에 제시되어 있다.

### 디코더 (Decoder)

디코더는 여러 개의 트랜스포머 디코더 계층으로 구성된 스택 구조이다. 각 계층은 **자기어텐션(self-attention)**, **교차어텐션(cross-attention)**, 그리고 FFN으로 이루어진다. 계산 효율성을 위해 **deformable cross-attention** [74]을 채택한다.

기존 DETR 및 그 변형 모델들은 일반적으로 **6개**의 디코더 계층을 사용한다. 반면, 본 구현에서는 **3개의 트랜스포머 디코더 계층**만을 사용한다. 이를 통해 디코더 추론 시간을 **1.4 ms에서 0.7 ms로 감소**시킬 수 있었으며, 이는 tiny 버전 기준으로 나머지 구성 요소가 차지하는 **1.3 ms**의 시간 비용과 비교했을 때 매우 유의미한 절감이다.

객체 쿼리(object query)는 **혼합 쿼리 선택 방식(mixed-query selection scheme)** [67]을 사용하여 **콘텐츠 쿼리(content query)**와 **공간 쿼리(spatial query)**의 합으로 구성한다. 콘텐츠 쿼리는 DETR과 유사하게 학습 가능한 임베딩이다. 공간 쿼리는 2단계(two-stage) 방식에 기반하여 생성되며, 구체적으로는 Projector의 마지막 계층에서 상위 K개의 feature를 선택한 뒤 bounding box를 예측하고, 해당 박스를 임베딩으로 변환하여 공간 쿼리로 사용한다.

### 프로젝터 (Projector)

프로젝터는 인코더와 디코더를 연결하는 역할을 한다. 인코더에서 집계된(encoded) feature map을 입력으로 받아 디코더에 전달한다. 본 논문에서는 YOLOv8 [29]에 구현된 **C2f 블록**(cross-stage partial DenseNet [26, 60]의 확장 형태)을 프로젝터로 사용한다.

LW-DETR의 **large** 및 **xlarge** 버전에서는, 프로젝터가 두 개의 스케일 (1/8, 1/32) feature map을 출력하도록 수정하고, 이에 맞추어 **멀티스케일 디코더(multi-scale decoder)** [74]를 사용한다. 이때 프로젝터는 두 개의 병렬(parallel) C2f 블록으로 구성된다. 하나는 deconvolution을 통해 업샘플링된 1/8 feature map을 처리하고, 다른 하나는 stride convolution을 통해 다운샘플링된 1/32 feature map을 처리한다. 단일 스케일 프로젝터와 멀티스케일 프로젝터의 전체 파이프라인은 그림 3에 제시되어 있다.

### 목적 함수 (Objective function)

분류 손실로는 **IoU-aware classification loss**, 즉 **IA-BCE loss** [3]를 사용한다.

![](/assets/images/posts/620/img_2.png)

![](/assets/images/posts/620/img_3.png)

![](/assets/images/posts/620/img_4.png)

![](/assets/images/posts/620/img_5.png)

### 그림 3 설명 (Figure 3)

(a) tiny, small, medium 모델을 위한 **단일 스케일 프로젝터**,  
(b) large, xlarge 모델을 위한 **멀티스케일 프로젝터** 구조를 보여준다.

## 3.2 구체화 (Instantiation)

우리는 **tiny, small, medium, large, xlarge**의 다섯 가지 실시간 검출기 인스턴스를 구성한다. 각 설정의 세부 사항은 표 1에 정리되어 있다.

**tiny** 검출기는 **6개**의 계층으로 이루어진 트랜스포머 인코더를 사용한다. 각 계층은 다중 헤드 자기어텐션(multi-head self-attention) 모듈과 FFN(feed-forward network)으로 구성된다. 각 이미지 패치는 **192차원**의 표현 벡터로 선형 매핑된다. 프로젝터는 **256 채널**의 단일 스케일 feature map을 출력하며, 디코더에는 **100개의 객체 쿼리(object queries)**가 사용된다.

**small** 검출기는 **10개의 인코더 계층**과 **300개의 객체 쿼리**를 사용한다. 입력 패치 표현 차원과 프로젝터 출력 차원은 tiny와 동일하게 각각 **192**, **256**이다. **medium** 검출기는 small과 유사하지만, 입력 패치 표현 차원이 **384**로 증가하며 이에 따라 인코더의 차원도 **384**로 설정된다.

**large** 검출기는 **10계층 인코더**를 사용하고, **두 개의 스케일 feature map**을 활용한다(3.1절의 Projector 참조). 입력 패치 표현 차원과 프로젝터 출력 차원은 각각 **384**, **384**이다. **xlarge** 검출기는 large와 유사하되, 입력 패치 표현 차원이 **768**이라는 점이 다르다.

![](/assets/images/posts/620/img_6.png)

## 3.3 효과적인 학습 (Effective Training)

### 추가적인 감독 신호 (More supervision)

DETR 학습을 가속화하기 위해 **더 많은 감독 신호(supervision)**를 도입하는 다양한 기법들이 제안되어 왔다 [6, 27, 75]. 본 연구에서는 **추론 과정(inference)을 변경하지 않으면서도 구현이 간단한 Group DETR** [6]를 채택한다. [6]을 따라, 학습 시에는 **가중치를 공유하는 13개의 병렬 디코더**를 사용한다. 각 디코더에 대해, 프로젝터의 출력 feature로부터 각 그룹에 대응하는 객체 쿼리(object query)를 생성한다. 추론 단계에서는 [6]과 동일하게 **기본(primary) 디코더 하나만**을 사용한다.

### Objects365 사전학습 (Pretraining on Objects365)

사전학습 과정은 두 단계로 구성된다.  
첫 번째 단계에서는, 사전학습된 모델을 기반으로 **MIM(Masked Image Modeling)** 기법인 **CAEv2** [71]를 사용하여 **Objects365 데이터셋**에서 ViT를 사전학습한다. 이 과정만으로도 COCO 기준 **0.7 mAP의 성능 향상**을 얻을 수 있다.

두 번째 단계에서는 [67, 7]을 따라, **Objects365 데이터셋에서 감독 학습(supervised learning) 방식으로 인코더를 재학습**하고, 동시에 **프로젝터와 디코더를 학습**한다.

## 3.4 효율적인 추론 (Efficient Inference)

우리는 간단한 수정 [37, 36]을 통해 **interleaved window attention과 global attention**을 적용한다. 즉, 일부 글로벌 자기어텐션(global self-attention) 계층을 **윈도우 자기어텐션(window self-attention)** 계층으로 대체한다. 예를 들어, 6 계층으로 구성된 ViT의 경우, 첫 번째, 세 번째, 다섯 번째 계층을 윈도우 어텐션으로 구현한다. 윈도우 어텐션은 feature map을 **겹치지 않는(non-overlapping) 윈도우**로 분할한 뒤, 각 윈도우 내부에서만 자기어텐션을 수행하는 방식이다.

우리는 interleaved attention을 효율적으로 수행하기 위해 **window-major feature map 구성 방식**을 채택한다. 이 방식은 feature map을 **윈도우 단위로 정렬**한다. 기존 ViTDet 구현 [36]에서는 feature map을 **행 우선(row-major)** 방식으로 정렬하는데, 이 경우 윈도우 어텐션을 적용하기 위해서는 row-major에서 window-major로 변환하는 **비용이 큰 permutation 연산**이 필요하다. 반면, 본 구현에서는 이러한 연산을 제거함으로써 모델의 지연 시간(latency)을 줄인다.

![](/assets/images/posts/620/img_7.png)

## 표 2. 효과적인 학습 및 효율적인 추론 기법의 영향

글로벌 어텐션만을 사용하는 초기 검출기부터 최종 **LW-DETR-small** 모델까지, 단계별로 적용한 기법들의 영향을 실험적으로 분석하였다. ‘†’는 ViTDet 구현을 사용했음을 의미한다. 마지막 행을 제외한 모든 결과는 45K iteration(12epoch에 해당)에서 측정되었으며, 마지막 행은 180K iteration으로 학습한 최종 모델의 결과이다.

![](/assets/images/posts/620/img_8.png)

---

## 1️⃣ 먼저 핵심 요약부터

이 문단의 요지는 딱 하나입니다.

> **“윈도우 어텐션과 글로벌 어텐션을 섞어 쓸 때,  
> feature map을 처음부터 ‘윈도우 기준(window-major)’으로 저장하면  
> 비싼 메모리 재배열(permutation)을 안 해도 된다.”**

성능(mAP) 얘기가 아니라, **순수하게 latency 줄이는 엔지니어링 트릭**입니다.

---
