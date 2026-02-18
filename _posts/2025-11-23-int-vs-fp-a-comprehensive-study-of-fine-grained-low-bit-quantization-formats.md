---
title: "INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats"
date: 2025-11-23 01:52:08
categories:
  - 인공지능
tags:
  - int
  - fp
---

<https://arxiv.org/abs/2510.25602>

[INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](https://arxiv.org/abs/2510.25602)

**초록**  
최신 AI 하드웨어, 특히 Nvidia의 Blackwell 아키텍처는 대규모 언어 모델(LLM)에서 빈번하게 발생하는 activation outlier 문제를 처리하기 위해 저정밀도 부동소수점(FP) 형식을 적극적으로 채택하고 있다. 그러나 이러한 산업적 흐름에도 불구하고, 다양한 정밀도와 양자화 단위(granularity)에 걸쳐 부동소수점(FP)과 정수(INT) quantization을 체계적으로 비교한 연구는 부족했으며, 이로 인해 알고리즘–하드웨어 공동 설계에 명확한 방향성이 부재한 상태였다. 본 논문은 FP와 INT 형식 간의 트레이드오프를 체계적으로 분석함으로써 이러한 공백을 채우고자 한다.

우리는 중요한 성능 교차점을 발견했다. 즉, **거친(granularity가 큰) 양자화에서는 FP가 우수하지만, 미세한 블록 단위(fine-grained, block-wise)에서는 비교가 훨씬 복잡해진다.** 우리의 종합적인 비교에 따르면, 널리 사용되는 8비트 미세 블록 기반 형식(예: 블록 크기 32의 MX)에서는 알고리즘 정확도와 하드웨어 효율성 모두에서 **MXINT8이 해당 FP 형식보다 우수**함을 확인했다.

반면 4비트 형식에서는 FP(MXFP4, NVFP4 등)가 대체로 정확도 측면에서 우위를 점한다. 그러나 outlier 완화를 위한 Hadamard rotation과 같은 기법을 적용할 경우 **NVINT4가 NVFP4를 능가**할 수 있음을 우리는 보여준다.

또한, 본 논문은 미세 블록 단위의 저비트 INT 훈련에서 발생하는 gradient bias를 해결하기 위한 **대칭적 클리핑(symmetric clipping)** 기법을 제안하며, 이를 통해 **MXINT8 훈련에서 거의 손실 없는 성능**을 달성할 수 있음을 보인다.

이러한 결과는 현재의 하드웨어 개발 방향에 도전한다. 즉, **단일 FP 기반 접근은 최적이 아니며**, 특히 **MXINT8과 같은 미세 블록 기반 INT 형식이 정확도, 전력 효율, 하드웨어 효율성 측면에서 더 균형 잡힌 선택지**임을 강조한다.

[Code]<https://github.com/ChenMnZ/INT_vs_FP>

[GitHub - ChenMnZ/INT\_vs\_FP: A framework to compare low-bit integer and float-point formats](https://github.com/ChenMnZ/INT_vs_FP)

**1 서론**  
대규모 언어 모델(LLM)의 확산은 계산량과 메모리 요구량의 급격한 증가와 함께 이루어졌으며 [43], 이로 인해 효율적인 모델 배포를 위해 양자화(quantization)는 필수적인 기술이 되었다. Transformer 아키텍처 기반 LLM을 양자화할 때 가장 큰 난제 중 하나는 활성화(activation) 분포 내에 존재하는 **강한 outlier**들이다 [38, 12]. 이러한 outlier는 **크기는 크지만 빈도는 매우 낮은 값들**로, 저정밀도 표현 방식에서는 큰 문제를 야기한다.

이처럼 넓은 동적 범위를 표현해야 하는 어려움 때문에, AI 하드웨어 업계 [31]는 FP8, FP4와 같은 **저정밀도 부동소수점(FP) 형식**으로 빠르게 이동하고 있다. NVIDIA의 Blackwell 아키텍처 [31]는 이러한 흐름을 잘 보여주는 대표적 사례로, 전통적인 정수(INT) 형식보다 FP 형식이 갖는 더 넓은 동적 범위를 활용해 outlier 문제를 더 안정적으로 처리할 수 있다는 점이 강조되고 있다.

그러나 업계 전반에서 이어지고 있는 이러한 FP 형식 중심의 흐름은 **불완전한 이해**를 기반으로 하고 있다. FP와 INT의 상대적 장단점은 다양한 양자화 단위(granularity)에 대해 **통합된 관점에서 체계적으로 평가된 적이 거의 없다**. 기존 연구들 [41, 6, 22]은 대부분 하나의 형식에만 집중하거나, 비교하더라도 per-channel과 같은 **거친 단위(coarse granularity)**에서만 수행되어, 중요한 질문에 답하지 못하고 있다. 즉, **양자화 단위가 미세해질수록(INT와 FP 사이의 성능·효율 트레이드오프가 어떻게 변화하는가?)**라는 핵심 문제가 여전히 밝혀지지 않았다.

현재 outlier 문제 해결을 위해 **미세 블록 단위(fine-grained, block-wise) 양자화**가 표준 기술로 자리잡고 있는 만큼 [34, 32], 이러한 구조가 **수 표현 형식(number format)**과 어떻게 상호작용하는지 이해하는 것은 효과적인 알고리즘–하드웨어 공동 설계를 위해 필수적이다.

본 논문에서는 **미세 블록 단위(fine-grained) INT 및 FP quantization에 대한 포괄적이고 체계적인 비교**를 수행한다. 우리의 분석 결과, 성능 면에서 중요한 **“교차 지점(crossover point)”**이 존재함을 확인했다. 거친 양자화 단위(coarse-grained)에서는 FP 형식이 확실한 우위를 보이지만, **블록 크기가 작아질수록 INT 형식이 매우 경쟁력 있게 변화**하며, 그 효과는 비트 폭(bit width)에 크게 의존한다.

양자화의 단위가 더 미세해질수록 각 블록 내의 **지역적(dynamic range)이 크게 감소**하며, 이에 따라 INT 형식이 가진 **균일 정밀도(uniform precision)**가 더욱 효과적으로 작용하게 된다. 이러한 현상은 Microscaling(MX) 형식의 32-element 블록이나 NVIDIA(NV) 형식의 16-element 블록과 같은 현대적인 블록 기반 양자화 방식 전반에서 관찰된다.

정확한 비교를 위해, 우리는 기존 FP 형식(MXFP8, MXFP6, MXFP4, NVFP4)에 대응하는 **정수 기반 변형(INT 기반 quantization)**을 정의하고 평가한다. 예를 들어 **MXINT8, MXINT6, MXINT4, NVINT4** 등을 새롭게 도입하여 그 성능을 FP 대응 모델들과 직접 비교한다.

우리의 주요 기여는 다음과 같다:

• **INT와 FP 형식 모두에 대해 quantization SNR(QSNR)을 모델링하는 이론적·통계적 프레임워크를 제안**한다. 이 프레임워크는 두 형식 간의 성능 트레이드오프를 이론적으로 직접 비교할 수 있게 하며, **성능 교차 지점(crossover point)**을 명확히 규명한다.

• **MXINT8이 직접 캐스팅(direct-cast) 기반 추론과 저비트(low-bit) 훈련 모두에서 MXFP8을 일관적으로 능가함**을 보인다. 또한 Hadamard rotation을 적용할 경우 **NVINT4가 NVFP4를 넘어설 수 있음**을 실험적으로 입증한다. 특히, 우리는 **gradient bias 문제를 해결하는 대칭 클리핑(symmetric clipping) 기법**을 새롭게 제안하여 **MXINT8 저비트 훈련에서도 거의 손실 없는 성능**을 달성한다.

• **하드웨어 비용 분석**을 제시하여, 동일한 처리량(throughput) 기준에서 미세 블록 기반 INT 형식이 FP 형식 대비 **칩 면적(area)**과 **에너지 효율(energy efficiency)** 측면에서 훨씬 우수함을 보여준다.

• 이러한 결과들을 종합하면, 현재의 **FP 중심 하드웨어 설계 흐름이 최적이 아님**을 분명히 보여주며, **미세 블록 기반 INT 형식이 정확도와 효율의 균형 측면에서 차세대 AI 가속기에 더 적합한 선택지**임을 강하게 제안한다.

**2 사전 지식(Preliminaries)**  
Quantization은 고정밀(high-precision) 텐서 **?**를 더 낮은 비트 폭(bit-width)으로 변환하는 과정이다. 본 절에서는 저비트 정수(INT) quantization과 부동소수점(FP) quantization을 소개하고, 특히 **미세 블록 단위(fine-grained block-wise)** 방식을 중심으로 quantization granularity를 설명한다. 또한 기존에 사용되는 다양한 저비트 블록 기반 형식들에 대한 개요를 제공한다.

**2.1 저정밀도 정수(Integer) 형식**

b비트 정수 양자화는 다음과 같이 정의된다:

![](/assets/images/posts/601/img.png)

여기서  
• s는 텐서 **X** 를 목표 정수 범위로 정규화하는 **스케일(scale)** 이고,  
• ⌊⋅⌉ 은 **가장 가까운 정수로 반올림(round-to-nearest)** 을 의미하며,  
• X\_q는 **dequantized된 텐서**다.

클리핑(clipping)은 양자화된 값이 정해진 정수 구간 [Q\_min⁡,Q\_max⁡]안에 있도록 보장한다. 예를 들어, signed b비트 정수의 경우:

![](/assets/images/posts/601/img_1.png)

위 구간이 정수 표현 가능한 전체 범위를 정의하게 된다.

**2.2 저정밀도 부동소수점(Floating-Point) 형식**

부동소수점 표현 방식 [24]은 **부호 비트(sign, S)**, **지수(exponent, E)**, **가수(mantissa, M)**의 세 부분으로 구성된다. 우리는 형식을 E\_xM\_y로 표기하며, 여기서  
• x는 지수 비트 수,  
• y는 가수 비트 수를 의미한다.

부호 비트는 값의 부호를 결정하고, 지수는 동적 범위(dynamic range)를, 가수는 정밀도(precision)를 결정한다. 부동소수점 수는 다음과 같이 복원된다:

![](/assets/images/posts/601/img_2.png)

여기서  
• s,e,m은 각각 부동소수점 수의 **부호(sign)**, **지수(exponent)**, **가수(mantissa)** 값을 의미한다.  
따라서 C\_FP​는 해당 저정밀도 부동소수점 형식에서 **표현 가능한 값들의 집합**이다.

부동소수점 quantization은 다음과 같이 정의된다:

![](/assets/images/posts/601/img_3.png)

여기서  
• Nearest(⋅,CFP)는 정규화된 값을 C\_FP​ 집합에 속한 **가장 가까운 부동소수점 값으로 매핑**하는 함수이다.

식 (3)은 일반적인 양자화 표현이며, C\_FP​를 C\_INT​로 대체하면 정수 기반 quantization이 동일한 형태로 복원된다.

**2.3 Quantization Granularity**

Quantization granularity는 스케일(scale) 파라미터가 텐서 내에서 어떻게 적용되는지를 정의한다. 일반적으로 **granularity가 더 미세할수록 정확도는 향상되지만**, 스케일 개수가 증가하기 때문에 **연산 및 메모리 오버헤드**도 커진다. 대표적인 granularity 선택지들은 다음과 같다:

(i) **Per-tensor**: 텐서 전체에 대해 하나의 스케일을 사용.  
(ii) **Per-channel**: 특정 축을 기준으로 각 채널마다 하나의 스케일을 사용.  
(iii) **Block-k**: 텐서를 한 차원에서 1×k 블록으로 나누고, 각 블록마다 고유한 스케일을 사용.

Block quantization은 **저정밀도 환경에서 정확도를 향상시키는 핵심 기법**이며, 본 논문에서는 주로 block quantization을 중심으로 다룬다.

![](/assets/images/posts/601/img_4.png)

**2.4 블록 기반(Block) 양자화 형식**

저비트 환경에서 정확도를 향상시키기 위해 OCP [34]는 **Microscaling(MX) 형식**을 제안했으며, 이는 32개 요소로 구성된 각 블록마다 하나의 공유 **UE8M0**¹ 스케일을 사용하는 방식이다. 이러한 미세 블록 단위 스케일링은 양자화 오차를 크게 줄여 준다. 최근 NVIDIA의 Blackwell GPU 시리즈 [32]는 **MXFP8, MXFP6, MXFP4**에 대한 **하드웨어 수준 지원**을 제공한다.

전통적으로 FP8은 E4M3, E5M2 두 가지 변형을 갖고 있으며, FP6은 E2M3, E3M2 변형을 가진다. 본 논문에서는 선행 연구 [21, 27, 34]와 동일하게, **MXFP8에는 E4M3**, **MXFP6에는 E2M3**를 사용한다. 이는 **가수 비트(mantissa bit)**가 미세 블록 기반 양자화의 성능에 더 중요한 역할을 하기 때문이다.

또한 NVIDIA는 MXFP4를 확장한 **NVFP4**를 제안했는데, 이는  
• 블록 크기를 32에서 16으로 줄이고,  
• 스케일 형식을 UE8M0에서 E4M3로 변경했으며,  
• 1단계 스케일(E4M3)의 overflow를 방지하기 위해 **2단계 per-tensor scale**을 도입한 형식이다.

이러한 흐름으로 인해 **최신 하드웨어는 저비트 미세 블록 기반 FP 형식을 주로 지원하는 방향**으로 발전하고 있다.

저비트 FP 형식과 정수 형식을 공정하게 비교하기 위해, 본 연구에서는 이에 대응하는 정수 기반 변형 또한 정의한다. 즉, **MXINT8, MXINT6, MXINT4, NVINT4** 네 가지를 도입하여 FP 형식과 직접 비교할 수 있도록 한다. 각 형식의 세부 사양은 표 1에 제시되어 있다.

---

¹ **UE8M0**: 지수 8비트, 가수 0비트로 구성된 8비트 unsigned floating-point 형식.

---
