---
title: "MXFP8, MXFP4 및 NVFP4에 대한 자세한 설명"
date: 2025-11-23 21:23:41
categories:
  - Article
---

<https://zhuanlan.zhihu.com/p/1969465397670551963>

### 1. 왜 mxfp8, mxfp4, nvfp4 같은 저정밀도 형식이 필요한가?

#### 1) 대규모 모델의 폭발적 성장으로 인한 계산 및 메모리 병목 심화

LLM의 파라미터 수는 이미 조 단위에 도달했고, 학습 시 필요한 FLOPs는 10²⁵을 넘는다.  
기존 FP32나 BF16 형식은 대역폭과 메모리 사용량이 매우 높아, 처리량과 에너지 효율에 제한을 준다.  
단순히 비트 너비만 줄인 INT8 또는 FP8을 적용하면 동적 범위가 충분하지 않아 학습 발산이나 성능 저하가 발생한다.

#### 2) 기존 저정밀도 형식의 구조적 한계

- **INT8**: 사전에 scale을 설정해야 하며, LLM의 activation·gradient가 따르는 멱함수(power-law) 분포에서 outlier가 많아 clipping이 쉽게 발생한다.
- **표준 FP8(E4M3, E5M2 등)**: 부동소수점 구조이지만 전체 텐서에 단일 scale(per-tensor scaling)을 적용하기 때문에, 동일 블록 안의 큰 값과 작은 값을 동시에 잘 표현하기 어렵다. 이로 인해 양자화 오차가 커진다.

이 문제를 해결하기 위해 **MX(Microscaling) 저정밀 형식**이 등장했다.  
이는 **‘블록 단위 scale 공유 + 저정밀 요소’**를 결합한 하이브리드 표현 방식이며 기본 구조는 다음과 같다.

![](/assets/images/posts/602/img.jpg)

### FP8과 MXFP8의 저장 방식 및 스케일링의 핵심 차이

- **공유 scale 1개**: E8M0 형식(8비트 exponent 전용, 2의 거듭제곱 기반)
- **MXFP 블록**:
  - 32개의 저정밀 요소
    - FP8 / FP6 / FP4 / INT8 중 선택하여 구성

![](/assets/images/posts/602/img_1.jpg)

### NVFP4는 MX 개념을 기반으로 한 NVIDIA의 공학적 확장 버전이다

- MX 형식의 핵심 패러다임인 **‘블록 단위 공유 scale + 저정밀 요소’** 구조를 그대로 계승한다.
- 하지만 **더 작은 블록 크기(블록당 16개 요소)** 및 **더 정밀한 scale(E4M3)** 등을 적용해,  
  MXFP4가 4비트 환경에서 겪던 **수치적 병목**을 해결했다.  
  이 병목은 scale이 2ⁿ 형태로만 가능해(지수 기반 이산적 scale),  
  원래 데이터를 FP4가 표현할 수 있는 전체 구간(full representable range)에 최적으로 압축하지 못해  
  **유효 정밀도 및 동적 범위가 낭비되는 문제**를 야기했었다.

전체적으로 보면 MX/NVFP 계열은  
**‘블록 단위 공유 scale + 저정밀 요소’** 를 결합한 하이브리드 표현 방식으로,  
4bit에서 8bit 구간까지 **넓은 동적 범위, 높은 수치 안정성, 높은 하드웨어 효율**을 동시에 만족하는 통합 솔루션을 제공한다.  
이는 LLM 대규모 학습이 지속적으로 확장되기 위한 핵심 기술적 경로라 할 수 있다.

### 2. mxfp8, mxfp4, nvfp4 소개

mxfp8, mxfp4, nvfp4는 모두 **양자화(quantization) 형식**으로,  
기준이 되는 **BF16 대비 메모리 사용량이 훨씬 적고 계산 속도는 더 빠르다.**  
다만 변환 과정에서 **추가적인 변환 오버헤드**와 **정확도 손실**이 발생할 수 있으며,  
실제 성능은 다음 요소에 의해 좌우된다.

- GEMM 크기(M, K, N 차원)
- 양자화 오버헤드를 커널 내에서 얼마나 효과적으로 **fusion**했는지 여부

참고로, 본문에서는 **mxfp6(E3M2 / E2M3)** 은 다루지 않는다.  
해당 형식은 mxfp4와 mxfp8 사이의 중간 수준 동적 범위 및 정밀도를 가지지만,  
최신 NV Native 코드에서는 별도의 구현이 없는 것으로 확인된다.

![](/assets/images/posts/602/img_2.jpg)

PyTorch 소스 코드

![](/assets/images/posts/602/img_3.jpg)

mxfp8과 mxfp4는 **OCP(Open Compute Project)** 에서 정의한 **표준 저정밀 형식**이며,  
**AMD MI350x**와 **NVIDIA Blackwell** 모두에서 지원된다.  
반면 **nvfp4는 NVIDIA가 자체 개발한 형식**으로,  
Blackwell 아키텍처에서만 **네이티브로 지원**되며  
표준 MX 형식보다 더 우수한 수치적 성능을 제공하는 것을 목표로 한다.

NVIDIA가 이전 연구에서 밝힌 바에 따르면,  
**10조(10 trillion) 토큰 규모의 초장기 학습**에서도 NVFP4를 사용해  
**파라미터 12B의 LLM을 성공적으로 학습**할 수 있었고,  
학습 손실 곡선은 FP8과 거의 완전히 일치했으며,  
다운스트림 작업에서의 성능 저하도 거의 없었다.  
아래 그림이 그 결과를 보여준다.

![](/assets/images/posts/602/img_4.jpg)

### 3. PyTorch에서의 Microscaling 형식: 핵심 파라미터 비교

4096×4096 크기의 BF16 텐서를 기준으로 할 때, mxfp8, mxfp4, nvfp4가 PyTorch에서 사용하는 핵심 기술 파라미터는 다음 표와 같다.

![](/assets/images/posts/602/img_5.jpg)

### 4. Microscaling 형식의 구성 모듈

mxfp8, mxfp4, nvfp4는 모두 유사한 원리로 구성되며, 여기서는 **mxfp8 기반 GEMM(일반 행렬 곱셈) 구성**을 예시로 설명한다.

핵심 구성 요소는 다음 세 가지이다.

1. 새로운 데이터 타입
2. 새로운 스케일링 방식
3. 새로운 GEMM 계산 방식

전체 흐름은 아래와 같다.

![](/assets/images/posts/602/img_6.jpg)

구체적인 흐름은 다음과 같다.

BF16 형식의 입력 데이터(x\_bf16)가  
→ 새로운 스케일링 방식과 데이터 타입 변환을 거쳐  
→ MXFP8 형식 데이터(x\_mxfp8)로 변환되고  
→ 새롭게 정의된 GEMM 연산에 참여한 뒤  
→ BF16 형식의 결과(y\_bf16)를 출력한다.

이 과정에서 가중치(w) 역시  
MXFP8 형식(w\_mxfp8)으로 변환되어 연산에 사용된다.

### (1) 데이터 타입

#### 1. torch.float8\_e8m0fnu

**용도:**  
mxfp8 및 mxfp4 형식에서 사용되는 **스케일링 데이터 타입**으로,  
torch.float32(단정밀도 부동소수점)의 **부호 없는 지수 부분**을 저장하는 데 활용된다.

![](/assets/images/posts/602/img_7.jpg)

### 형식 구조

총 8비트로 구성되며, **지수 비트(0–7비트)** 만 포함하고 **가수 비트는 없다**.  
구체적인 비트 의미는 float32의 구조를 참고할 수 있다.  
(float32는 부호 비트, 지수 비트, 가수 비트를 모두 포함하지만,  
이 데이터 타입은 **부호 없는 지수(exponent)만 추출해 저장**한다.)

### 접미어 의미

- **f**: finite(유한값)
- **n**: nonstandard NaN(비표준 NaN)
- **u**: unsigned(부호 없음)

### 지원 버전

PyTorch **2.7.0 이상**에서 사용 가능하다.

### 지원되는 연산

**지원되는 기능**

- 텐서 생성: empty, fill, zeros
- 바이트 단위 데이터 이동: cat, torch.view, torch.reshape
- 타입 변환(casting)
- scaled\_mm(스케일링 기반 행렬곱)에서 mxfp8·mxfp4의 **스케일 데이터 타입**으로 사용 가능

**지원되지 않는 기능**

- 대부분의 기타 수학 연산은 미지원

![](/assets/images/posts/602/img.png)

### 2. torch.float8\_e4m3fn

**용도:**  
mxfp8 형식에서 **요소(element) 데이터 타입**으로 사용되는 **8비트 부동소수점 형식**이다.

**형식 구조:**  
총 8비트로 구성되며

- 1비트 부호 비트(S)
- 4비트 지수 비트(e)
- 3비트 가수 비트(m)  
  로 이루어져 실제 요소 값을 저장한다.

**접미어 의미:**

- **f**: finite(유한값)
- **n**: nonstandard NaN(비표준 NaN)

**반올림 방식:**  
기본적으로 **RTNE(Round to Nearest, Ties to Even)**  
즉, 가장 가까운 값으로 반올림하며, 정확히 중간값일 경우 **짝수 쪽으로(round-to-even)** 반올림한다.  
이는 PyTorch의 기본 반올림 규칙과 동일하다.

### 3. torch.float4\_e2m1fn\_x2

**용도:**  
mxfp4 및 nvfp4 형식에서 사용하는 **요소(element) 데이터 타입**으로,  
두 개의 float4(4비트 부동소수점 수)를 **1바이트(8비트)** 안에 패킹하여 저장한다.

![](/assets/images/posts/602/img_8.jpg)

### 형식 구조

1바이트(8비트)를 두 부분으로 나누어, 각각 하나의 float4 데이터를 저장한다.

- **상위 4비트(7–4비트):**
  - 1비트 부호(S, 7비트)
  - 2비트 지수(e, 6–5비트)
  - 1비트 가수(m, 4비트)
- **하위 4비트(3–0비트):**
  - 1비트 부호(S, 3비트)
  - 2비트 지수(e, 2–1비트)
  - 1비트 가수(m, 0비트)

### 접미어 의미

- **f**: finite(유한값)
- **n**: nonstandard NaN(비표준 NaN)
- **x2**: 1바이트에 2개의 float4를 패킹(pack)한다는 의미

### 지원 버전

PyTorch **2.8.0 이상**에서 사용 가능

### 지원되는 연산

**지원 기능:**

- 텐서 생성: empty, fill, zeros
- 바이트 단위 데이터 이동: cat, torch.view, torch.reshape
- scaled\_mm 연산에서 mxfp4 및 nvfp4의 **요소 타입(element data type)** 로 사용 가능

**지원되지 않는 기능:**

- 대부분의 기타 수학 연산은 미지원

### 값의 표현 범위

각 float4는 **총 16개의 값만 표현 가능**하며, 값은 다음과 같다.

![](/assets/images/posts/602/img_1.png)

### (2) 스케일링 방식

#### 1. 스케일링의 핵심 원리

부동소수점 양자화는 고정밀 텐서(FP64, FP32, FP16, BF16 등)를  
저정밀 텐서(FP8, FP4 등, 즉 본문에서 다루는 mxfp8, mxfp4, nvfp4)로 저장하고,  
여기에 하나 또는 여러 개의 **스케일링 인자(scale)** 를 함께 사용한다.

스케일링 인자를 사용하는 목적은  
고정밀 텐서의 실제 값 범위를  
저정밀 텐서가 표현할 수 있는 값 범위에 맞게 **정렬(alignment)** 하는 것이다.

구체적인 과정은 다음과 같다.

![](/assets/images/posts/602/img_9.jpg)

- **스케일링 인자를 계산하여**, 고정밀 데이터를 저정밀 형식의 **유효 표현 범위** 안으로 매핑한다.
- 고정밀 데이터를 스케일링 인자와 곱한 뒤, **직접 저정밀 데이터로 변환**한다(필요 시 절단·반올림 포함).
- 원래 데이터가 필요하면, 저정밀 데이터를 다시 고정밀 형식으로 변환한 후 **스케일의 역수(1/scale)** 를 곱해 복원한다.  
  다만 이 복원 과정에서는 **일정 수준의 정밀도 손실**이 발생한다.

### 2. 왜 스케일링이 필요한가(왜 직접 변환하면 안 되는가)

![](/assets/images/posts/602/img_10.jpg)

위 그림에서 볼 수 있듯이,  
FP32 값을 FP8(예: mxfp8)로 **직접 변환**하면 값이 잘려(truncation) **큰 오차가 발생**한다  
(최대 오차가 352.00까지 발생할 수 있음).

반면, 먼저 데이터에 **스케일링(예: FP32 값을 0.56 배)** 을 적용한 뒤 FP8로 변환하면  
FP8이 표현할 수 있는 **유효 범위 안에 값이 완전히 들어오게 되며**,  
그 결과 변환 오차를 크게 줄일 수 있다.

### 3. 주요 스케일링 방식

하드웨어 특성이나 모델 구조에 따라  
스케일링의 방식(스케일링 규칙, 적용 단위 등이 달라지며),  
주요 방식은 다음과 같다.

![](/assets/images/posts/602/img_11.jpg)

Hopper, MI300 Scaling

![](/assets/images/posts/602/img_12.jpg)

DeepSeekV3

![](/assets/images/posts/602/img_13.jpg)

MX Scaling

![](/assets/images/posts/602/img_14.jpg)

NVFP4

![](/assets/images/posts/602/img_2.png)

### 4. torch.float8\_e8m0fnu의 스케일(scale) 계산 방식

데이터의 최대 절대값 **max(abs(x))** 로부터  
E8M0(torch.float8\_e8m0fnu) 스케일 인자를 계산하는 방식은 크게 두 가지가 있다.

#### **① OCP MX 규격(floor 방식)**

- max(abs(x))의 **지수 비트(exponent)** 를 추출한다.
- 그 지수에서 **요소 데이터 타입(elem\_dtype)의 최댓값 2의 거듭제곱(elem\_dtype\_maxpow2)** 을 빼서 스케일을 결정한다.
- 하지만 NVIDIA는 기존 연구에서 이 방식이 일부 값에 대해 **오버플로우(overflow)가 발생할 수 있다**고 지적했다.  
  → 이를 해결하기 위해 **올림(round-up)** 방식이 필요하다고 제안했다.

#### **② NVIDIA 방식(rceil, round-up-ceil)**

- 먼저 max(abs(x))을 해당 데이터 타입이 표현 가능한 **최대 절대값(max\_abs\_dtype)** 으로 나눈다.
- 그 결과를 **올림(ceil)** 처리한 뒤,
- 그 값의 지수 비트를 추출하여 스케일을 만든다.
- floor 방식보다 오버플로우 위험이 낮아 **더 안정적인 스케일링**을 제공한다.

### (3) GEMM 계산

mxfp8 GEMM, nvfp4 GEMM 등 Microscaling 기반 GEMM의 핵심은,  
**저정밀 텐서(mxfp8, nvfp4 등) + 해당 스케일 인자**를 조합해  
전용 커널을 통해 고효율 행렬 곱셈을 수행하는 것이다.

구체적인 예시는 다음과 같다.

1. 블록 스케일링 mxfp8 GEMM

```
# 1. 입력 텐서 A, B를 mxfp8 형식으로 변환하고,
#    각 텐서의 스케일 인자 A_scale, B_scale을 함께 얻는다.
A_scale, A_fp8 = to_mxfp8(A)
B_scale, B_fp8 = to_mxfp8(B)

# 2. scaled_mm을 호출하여 mxfp8 형식으로 행렬 곱을 수행한다.
#    scale_recipe_b는 B의 스케일링 방식을 1×32 블록 단위(Blockwise1x32)로 지정한다.
#    output_dtype은 출력 결과를 bf16 형식으로 지정한다.
result = scaled_mm(
    A_fp8, B_fp8,
    scale_recipe_a=ScalingType.Blockwise1x32,
    scale_recipe_b=ScalingType.Blockwise1x32,
    output_dtype=torch.bfloat16
)
```

여기서 to\_mxfp8(형식 변환), scaled\_mm(스케일링 행렬 곱) 등의 함수와 커널은  
PyTorch의 torch/ao 모듈에서 제공된다.

2. NVFP4 블록 스케일링 GEMM

```
# scaled_mm을 호출해 nvfp4 형식으로 행렬 곱을 실행한다.
# scale_a는 A의 스케일 인자를 지정:
#  - 1×16 블록 단위 스케일 인자(to_blocked(A.scales))
#  - + 글로벌 텐서 스케일 인자(A_global)
#
# scale_recipe_a는 A의 스케일링 방식을 지정:
#  - 1×16 블록 단위(Blockwise1x16)
#  - + 텐서 전체 단위(TensorWise)
result = scaled_mm(
    A.fp.t(), B.fp,
    scale_a=[to_blocked(A.scales), A_global],
    scale_recipe_a=[ScalingType.Blockwise1x16, ScalingType.TensorWise],
    # 기타 파라미터는 실제 사용 환경에 맞게 설정
)
```

이 계산 방식은

- **여러 개의 스케일 인자**,
- **독립적인 메모리 재배치 패턴**,
- 그리고 **전용 커널(cuBLAS, Cutlass, rocBLAS, Composable Kernel 등)**

에 대한 스케줄링을 모두 지원한다.

### 5. 성능 평가

NVIDIA B200 GPU에서의 테스트 결과에 따르면,  
**mxfp8**과 **nvfp4**는 **BF16 대비 GEMM 계산 성능에서 매우 큰 이점을 제공**하며,  
구체적인 결과는 다음과 같다.

![](/assets/images/posts/602/img_15.jpg)

### (1) 절대 성능 비교

행렬 크기(M×K×N)가 256×256×256에서 16384×16384×16384로 커질수록,  
**mxfp8과 nvfp4의 커널 성능은 BF16을 지속적으로 초월하며 그 격차도 점점 커진다.**

예를 들어,

- 행렬 크기가 **16384×16384×16384**일 때
  - **nvfp4 성능은 약 6000 TFLOPS에 근접**,
  - **mxfp8은 약 4000 TFLOPS**,
  - **bf16은 약 2000 TFLOPS** 수준에 그친다.

### (2) 상대 성능 비교(BF16 대비 속도 향상 비율)

- 작은 행렬 크기(예: 256×256×256)에서는  
  **mxfp8과 nvfp4의 가속 비율이 거의 1에 가까워** 눈에 띄는 이점이 없는데,  
  이는 이 구간에서는 스케일링 및 형 변환 오버헤드의 비중이 높기 때문이다.
- 행렬 크기가 커질수록(예: 2048×2048×2048 이상)  
  **가속 비율이 점차 증가**한다.
  - mxfp8: 최대 **약 2배**
  - nvfp4: 최대 **약 3.5배**  
    이는 이론적 최대 가속 효과(mxfp8은 최대 2배, nvfp4는 최대 4배)에 부합한다.

### (3) 스케일링 오버헤드 비교

스케일링 방식에 따라 오버헤드가 달라지며, 같은 행렬 크기에서도 차이가 나타난다.

- **Per-row 스케일링**과 **MX 스케일링**은 단일 커널에서 수행할 수 있어  
  → 오버헤드가 낮다.
- **Per-tensor 스케일링**은 여러 커널이 협력해야 하므로  
  → 오버헤드가 높다.  
  예: 16384×16384×16384 행렬에서
  - per-tensor 스케일링 오버헤드: 약 **0.15**
  - per-row 스케일링 오버헤드: 약 **0.1**

### 6. 학습 및 추론에서의 성능 고려사항

#### (1) 학습(Training) 시나리오

1. **스케일링(fusion) 커널의 통합이 핵심**

![](/assets/images/posts/602/img_16.jpg)

NVIDIA B200 GPU에서 **16384×16384 크기(M=K=16384)** 의 행렬을 대상으로 테스트한 결과,  
스케일링과 형 변환을 결합한 **fused to\_mxfp8 / to\_mxfp4 / to\_nvfp4 커널**은  
비(非)결합 커널 대비 **10배 이상 빠른 성능**을 보였다.

### 2. GEMM 행렬 크기는 충분히 커야 한다

GEMM의 행렬 크기(M×K×N)가 충분히 커야  
스케일링 및 변환 오버헤드를 상쇄하고 실제 성능 향상을 얻을 수 있다.

**성능 조건:**

저정밀 형식의 성능이 BF16보다 우위가 되려면,

> **BF16 GEMM 시간 > 저정밀 GEMM 시간 + 저정밀 스케일링 오버헤드**

여야 한다.

여기서

- **GEMM 계산 시간**은 행렬 크기의 **세제곱(O(M×K×N))** 에 비례하고
- **스케일링 오버헤드**는 **이차항(O(M×K + M×N + K×N))** 에 비례한다.

따라서 **행렬 크기가 커질수록**,  
GEMM의 삼차 복잡도가 스케일 오버헤드의 이차 복잡도를 압도하게 되어  
저정밀 형식(mxfp8, nvfp4 등)의 **성능 이점이 더 뚜렷하게 나타난다**.

![](/assets/images/posts/602/img_17.jpg)

### 임계치(Threshold) 범위

- **mxfp8:**  
  행렬 크기가 **약 2048×2048×2048 이상**이어야 성능 이점이 나타난다  
  (roofline 모델 기준 상한).
- **mxfp4 / nvfp4:**  
  행렬 크기가 **약 1800×1800×1800 이상**이어야 성능 이점이 확보된다  
  (roofline 모델 기준 상한).

### 3. 학습 과정에서는 텐서를 여러 번 양자화해야 한다

예를 들어,  
mxfp8 기반의 행렬곱(mm) 연산에서 **순전파(forward)** 와 **역전파(backward)** 를 수행하는 경우를 보면 다음과 같다.

![](/assets/images/posts/602/img_3.png)

저정밀 GEMM 커널은

- **첫 번째 입력은 row-major(행 우선)**,
- **두 번째 입력은 col-major(열 우선)**

형식을 요구한다.

따라서  
입력(input), 가중치(weight), 출력 기울기(grad\_output)은  
각각 **row-major 방식과 col-major 방식**으로 모두 양자화해야 한다.

그 결과,  
순전파·역전파 과정에서 **총 6개의 저정밀 텐서(mxfp8 등)** 가 생성되며,  
이는 데이터 처리의 복잡도를 증가시키는 요인이 된다.

![](/assets/images/posts/602/img_18.jpg)

### 결합(fusion) 필요성

스케일링, 형 변환, GEMM과 같은 여러 연산을  
**컴파일러(torch.compile 등)** 의 도움을 받거나  
**직접 작성한 fused 커널**을 통해 하나로 결합해야 한다.  
이렇게 해야 오버헤드를 줄이고 성능을 더욱 향상시킬 수 있다.

### 4. 최적화 방안: 2D 블록 포맷으로 가중치 양자화 횟수 감소

![](/assets/images/posts/602/img_19.jpg)

가중치 행렬 **W를 32×32(또는 16×16)** 크기의 2차원 블록으로 분할한다.  
각 블록은 **하나의 scale을 공유**하며  
(또는 각 행/열 단위로 scale을 공유하도록 설정할 수도 있으나, 레이아웃은 대칭적이다.)

이 방식은,  
블록을 전치(transpose)하더라도 메모리 상에서의 **scale 해석이 동일하게 유지되기 때문에**,  
가중치를 **한 번만 양자화하면** 충분하다.

## (2) 추론(Inference) 시나리오

### 1. 가중치는 한 번만 양자화하면 된다

추론에서는 가중치가 변하지 않기 때문에  
추론 전 **한 번 양자화해 저장해두면**,  
추론 시에는 해당 저정밀 가중치를 그대로 사용하면 된다.

실시간으로 양자화해야 하는 것은 **활성값(activation)** 뿐이므로  
전체 양자화 오버헤드가 크게 줄어든다.

### 2. 스케일링 결합(fusion) 커널은 여전히 핵심

학습과 동일하게,  
**fused to\_mxfp8 / to\_nvfp4 커널은 비결합 버전 대비 10배 이상 빠르다.**

torch.compile을 활용하거나  
직접 fused 커널을 작성하여 스케일링과 변환 과정을 통합해야 한다.

### 3. nvfp4는 글로벌 스케일 인자를 오프라인에서 보정해야 한다

nvfp4 형식은 **전역 FP32 스케일 인자 1개**가 필요하다.  
이 값은 추론 전에 **오프라인(calibration)** 으로 계산할 수 있으며  
(예: 추론용 검증 데이터셋으로 보정)  
실시간 계산이 필요 없으므로 추론 단계의 오버헤드를 감소시킨다.

### 4. GEMM 행렬 크기가 충분히 커야 한다

행렬 크기가 작으면 스케일 및 변환 비용이 이득을 상쇄하므로  
저정밀 형식의 성능 향상을 얻기 위해서는  
**충분히 큰 행렬 크기**가 필요하다.

![](/assets/images/posts/602/img_20.jpg)

동적인 활성값(activation) 양자화가 필요한 추론 환경에서는  
행렬 크기가 **1024×1024×1024 또는 2048×2048×2048 이상**이어야  
양자화 오버헤드를 상쇄하고 성능 가속을 얻을 수 있다.

Roofline 모델을 보면,  
행렬 크기가 커질수록 **mxfp8, mxfp4, nvfp4의 가속 비율은 더욱 높아지며**,

- **nvfp4는 최대 약 3.5배**,
- **mxfp8은 최대 약 2배**  
  가속 효과를 보인다.

![](/assets/images/posts/602/img_4.png)

### 참고 문헌

- mxfp8, mxfp4, nvfp4 formats and applications in PyTorch, PyTorch Conference 2025
- Recipes for Pre-training LLMs with MXFP8
- Pretraining Large Language Models with NVFP4
- Quartet: Native FP4 Training Can Be Optimal for Large Language Models
