---
title: "Chapter 10 Various artificial intelligence"
date: 2023-05-20 21:57:26
categories:
  - 인공지능
tags:
  - adversarial attacks
  - Out-of-Distribution Detection
---

인공지능이 활용되고 있는 다양한 영역을 소개하고 기본적인 인공지능 리뷰를 끝내려한다.

먼저 소개할 것은 Adversarial Attacks이다.

![](/assets/images/posts/74/img.png)

위의 이미지가 햇갈리는가? 개와 머핀, 개와 대걸래의 사진이다. 우리는 명확하게 인식하지만 컴퓨터는 명확하게 인식하지 못한다.

Adversarial Attack이란 이미지 분류와 같은 인공지능 모델을 속이기 위해, 입력에 감지할 수 없는 노이즈를 추가하여 모델의 결과를 변경하는 것이다.

![](/assets/images/posts/74/img_1.png)

Adversarial Attack

다음 그림과 같이 많은 형태의 Adversarial Attack이 있다.

![](/assets/images/posts/74/img_2.png)

Adversarial examples는 기계 학습 모델에 대한 보안 우려를 일으킨다.   
- 하나의 네트워크를 속이기 위해 만들어진 공격은 다른 네트워크도 속인다.   
- 공격은 물리적 세계에서도 작동한다.   
- 심층 신경망의 경우 적대적 예제를 생성하는 것이 매우 쉽지만, 이 문제는 다른 기계 학습 분류기에도 영향을 미친다.   
- 다양한 방어 전략이 제안되었지만, 강력한 공격에 대해 모두 실패한다.

특히, 마지막에 관련된 강력한 논문이 존재한다. 모든 분류기에 작동하는 noise에 대한 논문으로 찾아보면 있을 것이다.

샘플은 결정 경계를 횡단한다.  
- 입력 픽셀의 작은 차이가 가중치 및 출력에 대해 극적인 차이를 만든다.   
- 간단한 결정 경계는 데이터 포인트 주변의 논리적 구역을 분리하지 못한다.

이는 그림으로 보는게 더 이해하기 쉽다.

![](/assets/images/posts/74/img_3.png)

수식적으로 보면 다음과 같다.

![](/assets/images/posts/74/img_4.png)

우리 입력이 일정 거리안에 있다면 정확한 분류가 된다는 말이다.

우리가 사용하는 차원에서의 거리는 다음으로 측정한다.

![](/assets/images/posts/74/img_5.png)

차원에 따른 설명은 다음과 같다.

![](/assets/images/posts/74/img_6.png)

0차원은 픽셀 갯수, 1차원은 픽셀 갯수의 변화, 2차원은 유클리디안 거리, 무한대는 해당 coordinate의 변경된 픽셀 value이다.

 따라서 이걸 본다면, Adversarial Example은 특정 아키텍처에 특화되지 않으며, 잘못 분류된 클래스들도 다양한 모델들에서 대부분 동일할 수 있다.  
  
일부 원인들로 생각되는 것은 다음과 같다.  
- 신경망의 비선형성   
- 충분하지 않은 규제화(Regularization)   
- 충분하지 않은 모델 평균화(Model averaging)

Threat Models은 다음과 같이 구성된다.

Black-box 조건

- 도메인 데이터에 대한 접근 권한   
- 모델에 입력을 제공하고 출력을 관찰할 수 있음

White-box 조건

- 데이터, 아키텍처 및 파라미터에 대한 완전한 지식을 가지고 있음

White-box Attacks

- 네트워크의 모든 매개변수를 알고 있음  
- 네트워크를 분석하고 적대적인 예를 찾아 사용

Black-box Attacks

- 네트워크의 매개변수는 알 수 없지만 교육 데이터는 공개되어 있다

- 이를 활용하여, 모델을 만들어서 adversarial examples를 찾으면 공격하고자 하는 모델에서도 적용이된다.

![](/assets/images/posts/74/img_7.png)

Black-box Attacks

공격에도 Targeted와 Untargeted 형태가 있다.

Non-targeted attacks은 공격자는 잘못된 클래스를 얻기 위해 분류자를 속이려고 하는 것이다.

![](/assets/images/posts/74/img_8.png)

Non-targeted attacks

Targeted attacks은 공격자는 특정 클래스를 예측하기 위해 분류자를 속이려고 한다.

![](/assets/images/posts/74/img_9.png)

Targeted attacks

이에 대해서 알아보자.

One Step Gradient Method들에 대해서 알아보자

![](/assets/images/posts/74/img_10.png)

Fast Gradient Sign Method

- Untargeted: Fast Gradient Sign Method (FGSM)

실제 레이블의 역경사 방향을 찾아 노이즈 추가

![](/assets/images/posts/74/img_11.png)

Fast Gradient Sign Method

- Targeted: One step least-likely class method

대상 레이블의 기울기 방향을 찾아 노이즈 추가  
대부분, 가능성이 가장 낮은 클래스가 사용된다.

![](/assets/images/posts/74/img_12.png)

One step least-likely class method

- Randomized Fast Gradient Sign Method (RAND+FGSM)

FGSM을 사용하기 전에 먼저 작은 임의의 섭동을 적용합니다.

![](/assets/images/posts/74/img_13.png)

Randomized Fast Gradient Sign Method

One Step Gradient Method는 Effective and Fast하다. 그리고 일반적으로 기본적으로 사용되는 형태이다

![](/assets/images/posts/74/img_14.png)

Randomized Fast Gradient Sign Method

Iterative Methods들에 대해서 알아보자

- Basic iterative method (iter. basic)

True Label의 역경사 방향을 반복적으로 찾아 노이즈 추가

![](/assets/images/posts/74/img_15.png)

- Iterative least-likely class method (iter. l.l.)

대상 레이블의 기울기 방향을 반복적으로 찾아 노이즈 추가

![](/assets/images/posts/74/img_16.png)

Iterative Methods는 one step methods보다 강력하다. 경계면을 계속 파고들어가기 때문이다.

![](/assets/images/posts/74/img_17.png)

Basic iterative method

![](/assets/images/posts/74/img_18.png)

- Projected Gradient Descent (PGD)

데이터 주변의 임의 시작점을 기준으로 FGSM을 반복적으로 적용하고 실패할 경우 임의의 지점에서 다시 시작한다.

아래의 식을 적용해서 사용한다. 둘중 하나를 선택.

![](/assets/images/posts/74/img_19.png)

![](/assets/images/posts/74/img_20.png)

장점으로는 강력하지만, 단점은 너무 느리다.

![](/assets/images/posts/74/img_21.png)

Projected Gradient Descent

공격 방법에 대해서 알았으니 방어법에 대해서 알아보자.

- Adversarial Training

적대적 예와 natural 데이터를 사용하여 모델 학습

- Filtering/Detecting

적대적인 예 또는 섭동의 패턴을 학습

수 측면 모델을 사용하여 적대적 샘플을 분류하지 않고 거부

- Denoising (Preprocessing)

denoiser를 사용하여 입력의 노이즈를 줄입니다.

Adversarial Training의 Idea는 다음과 같다.

교육 데이터에 올바른 분류가 있는 적대적 샘플 포함  
안장점 문제로 공식화

![](/assets/images/posts/74/img_22.png)

saddle point

Training 방법은 다음과 같다.

– 사전 학습된 모델 또는 처음부터 시작  
– 미니 배치당 적대적 샘플의 비율  
– 원본 샘플을 적대적 샘플로 교체하거나 둘 다 유지  
– 적대적 샘플의 사전 계산 대 훈련된 모델의 현재 버전에서 즉석에서 계산

적용 paper는 [Madry et al., 2017] trains robust models on MNIST and CIFAR10

– 무작위 재시작과 함께 PGD 사용  
– 깨끗한 샘플을 적대적인 샘플로 교체

![](/assets/images/posts/74/img_23.png)

Adversarial Training와 추가적인 방법을 사용한 것들은 다음과 같다.

Adversarial Logit Pairing [Kannan et al., 2018]

– 교육 및 테스트 중에 표적 PGD 공격 사용  
– ALP(Adversarial Logit Pairing) 적용:

![](/assets/images/posts/74/img_24.png)

ALP

Defensive Distillation [Papernot et al. 2015]

– 원본 데이터로 교사 모델을 교육한다  
– 교사의 예측을 기반으로 학생 모델을 교육한다  
– 예측을 위해 학생만 사용한다

Label Smoothing

방어적 학습을 하는 것과 동등한 효과를 보여준다. [Warde-Farley and Goodfellow, 2016]

Adversarial Training + ?에서 일반적으로 사용되는 방법들

![](/assets/images/posts/74/img_25.png)

Detection에서 일반적으로 사용되는 방법들

![](/assets/images/posts/74/img_26.png)

Denoising에서 일반적으로 사용되는 방법들

![](/assets/images/posts/74/img_27.png)

이제 Out-of-Distribution Detection에 대해서 알아보자.

![](/assets/images/posts/74/img_28.png)

위의 그림처럼 개와 고양이만 학습한 NN이 말을 입력받으면  "모르겠어요"라고 말할 수 있을까?

우리는 일반적으로 분류를 위해 softmax를 사용한다. 따라서 도메인에 있는 이미지의 경우 softmax는 선명한 출력을 생성하고 아니면 모호한 출력을 생성한다.

![](/assets/images/posts/74/img_29.png)

이에 대한 답들을 알아보자

1. Over Confidence

NN은 과신 예측을 출력하는 경향이 있다. NN은 노이즈 이미지에 대해 높은 신뢰도로 예측을 반환하는 것이다.

위에 개, 고양이만 학습한 NN이 말 입력을 받고 softmax로 개에 대해서 0.95를 출력하는것과 같다.

![](/assets/images/posts/74/img_30.png)

Over Confidence

![](/assets/images/posts/74/img_31.png)

Over Confidence

Threshold-based Detection을 통해 이를 해결하고자 한다.

임계값을 기반으로 이상 혹은 정상 상태를 감지하는 방법이다. 하지만 제한 사항으로 이전 작업의 성능은 분류기를 훈련하는 방법에 따라 크게 달라진다.

![](/assets/images/posts/74/img_32.png)

Threshold-based Detection

![](/assets/images/posts/74/img_33.png)

Threshold-based Detection 분류기에 따른 차이

Confidence Calibration은 모델의 예측 확신도를 실제 확률에 근접하도록 조정하여 모델의 예측을 보다 신뢰할 수 있도록 만드는 방식이다. 분포 외 샘플에 대한 신뢰도가 낮도록 특별히 신경망을 훈련시켜서 이를 활용하는 것이다.

![](/assets/images/posts/74/img_34.png)

분포 외 데이터의 KL 다이버전스 최소화를 사용해 Confident Loss를 일으켜 이를 해결한다.

![](/assets/images/posts/74/img_35.png)

Confident Loss

이를 위해서는 일반적으로 주어진 분포 외 데이터는 일반적으로 분포 외 샘플을 모델링하기에 충분하지 않다. 즉, 유통되지 않은 샘플이 더 필요하다. 그렇다면 GAN을 사용하면 되지 않을까? 실제 GAN을 사용했더니 다음의 형태를 보였다.

– (a) 및 (b) 분포 외 데이터는 분포 내에서 희박합니다.   
– (c) & (d) out-of-distribution 데이터는 in-dist 주위에 밀집되어 있습니다.

![](/assets/images/posts/74/img_36.png)

따라서 ID를 중심으로 합성 OOD를 조밀하게 생성되야한다. 아래의 그림과 같이 말이다.

![](/assets/images/posts/74/img_37.png)

이를 위해서 다음의 손실 함수를 사용한다.

![](/assets/images/posts/74/img_38.png)

Output of classifier를 추가하는 방식이다.

다른 방법으로 Confidence Calibration을 고치는 법은 Joint Loss: Confidence Loss + GAN Loss를 사용하는 방법이다.

![](/assets/images/posts/74/img_39.png)

Joint Loss: Confidence Loss + GAN Loss

이를 위해 실험한 결과 다음과 같다.

Result with Confident loss (without GAN loss)

![](/assets/images/posts/74/img_40.png)

어쩔때는 좋고, 어쩔때는 나쁘다.

Result with Joint Loss (with GAN Loss)

![](/assets/images/posts/74/img_41.png)

훨씬 일관성 있는 것을 확인할 수 있다.

Distribution-based Detection은 주어진 데이터의 분포를 기반으로 이상 상태를 감지하는 방법이다. 이 방법은 데이터가 정상적인 분포를 따른다고 가정하고, 이를 기준으로 이상 값을 식별한다.

![](/assets/images/posts/74/img_42.png)

출처 : https://www.semanticscholar.org/paper/Energy-based-Out-of-distribution-Detection-Liu-Wang/35b966347dae2f0d496ea713edf03a68211838a5/figure/0

Variance-based Detection은 데이터의 변동성을 기반으로 이상 상태를 감지하는 방법이다. 이 방법은 데이터의 분산이나 변동성을 측정하여 정상적인 변동 범위에서 벗어나는 값을 식별한다.

![](/assets/images/posts/74/img_43.png)

노이즈가 이상분포를 보이기 때문에 잡아낼 수 있다.

**Slimmable Neural Networks**

Slimmable Neural Networks는 주어진 입력에 대해 다양한 너비(즉, 네트워크의 크기나 용량)를 가진 모델을 훈련하고, 추론 단계에서 다양한 연산 요구사항에 맞게 이를 조정할 수 있게 해주는 모델이다. 이 방법은 하나의 모델이 다양한 하드웨어 환경에서 작동하도록 하거나, 더 적은 리소스를 사용하면서 성능을 유지하거나 개선하도록 하는 데 유용하다.  
이를 위해, Slimmable Neural Networks는 모든 레이어에서 네트워크 너비를 동적으로 변경할 수 있는 구조를 가지며, 다양한 너비 설정에서의 모델 성능에 대한 일관성을 보장하기 위한 훈련 전략을 채택한다.  
이러한 방식은 "Universal Slimmable Networks"라는 개념으로 확장되기도 하는데, 이는 네트워크의 레이어별로 독립적인 너비 설정을 허용함으로써 더욱 다양한 모델 크기와 연산 복잡도를 가능하게 한다. 이런 접근법은 복잡한 모델과 효율적인 모델 간의 성능 차이를 줄이는 데 도움이 된다.  
마지막으로, Slimmable Neural Networks와 이와 유사한 방법들은 가중치 공유(weight sharing) 방식을 이용하여 메모리 사용량을 줄이고, 다양한 크기의 네트워크를 효율적으로 훈련하거나 테스트하는 데 사용될 수 있다.

![](/assets/images/posts/74/img_44.png)

핵심은 유연한 모델 크기와 위 그림과 같은 공유 가중치이다.

**ImageNet-Trained CNNs are biased towards texture; Increasing shape bias improves accuracy and robustness**

"ImageNet - Trained CNNs are biased towards texture; Increasing shape bias improves accuracy and robustness"는 ImageNet으로 훈련된 컨볼루션 신경망(CNN)이 텍스처에 편향되어 있으며, 형태에 대한 편향을 높이면 정확도와 견고성이 향상된다는 것을 의미한다.   
ImageNet은 대규모 이미지 데이터셋으로, 다양한 클래스의 이미지를 포함하고 있다. CNN은 이러한 ImageNet 데이터셋으로 훈련될 때, 일반적으로 텍스처와 관련된 특징을 감지하는 경향이 있다. 이는 텍스처 정보가 이미지 분류에 유용하게 작용하기 때문이다.   
그러나 텍스처에만 의존하는 모델은 형태나 구조와 같은 다른 중요한 시각적 특징을 감지하는 데 어려움을 겪을 수 있다. 이는 모델의 일반화 능력과 견고성을 제한할 수 있다. 텍스처에 강한 편향이 있는 모델은 텍스처 정보에 민감하게 반응하여 동일한 클래스의 객체에 대해 다른 텍스처를 가진 이미지에 대한 분류 정확도가 낮아질 수 있다.  
반면, 형태에 대한 편향을 높이면 모델은 객체의 형태, 구조, 경계 등과 같은 시각적 특징에 더 집중하게 된다. 이는 객체 인식과 분류의 정확도를 향상시킬 수 있으며, 모델의 견고성을 강화할 수 있다. 형태에 대한 편향이 높은 모델은 텍스처 변화나 배경 변화에 덜 민감하게 된다.   
따라서, ImageNet-Trained CNNs는 텍스처에 편향되어 있으며, 형태에 대한 편향을 높이면 모델의 정확도와 견고성을 향상시킬 수 있다.
