---
title: "Gaussian Mixture Flow Matching Models"
date: 2025-04-22 17:52:28
categories:
  - 인공지능
tags:
  - gaussian mixture flow matching
  - gmflow
---

<https://github.com/Lakonik/GMFlow?_bhlid=81f89db39632970b22a6d8f7641c5518cad90ef0>

[GitHub - Lakonik/GMFlow: Gaussian Mixture Flow Matching Models (GMFlow)](https://github.com/Lakonik/GMFlow?_bhlid=81f89db39632970b22a6d8f7641c5518cad90ef0)

<https://www.arxiv.org/abs/2504.05304?_bhlid=4fcaeeb727d7a8544f84a8ad986fa9ea922fee09>

[Gaussian Mixture Flow Matching Models](https://www.arxiv.org/abs/2504.05304?_bhlid=4fcaeeb727d7a8544f84a8ad986fa9ea922fee09)

**초록 (Abstract)**  
확산 모델(diffusion models)은 잡음 제거(denoising) 분포를 가우시안(정규분포)으로 근사하고 그 평균(mean)을 예측합니다. 반면, 플로 매칭(flow matching) 모델은 이 가우시안 평균을 흐름 속도(flow velocity)로 재매개변수화합니다. 그러나 이러한 모델들은 **샘플링 횟수가 적을 때(few‑step sampling)** 이산화(discretization) 오차로 인해 성능이 떨어지며, **Classifier‑Free Guidance(CFG)** 하에서는 과도하게 채도가 높은(oversaturated) 색상을 생성하는 경향이 있습니다.

이를 해결하기 위해 우리는 **Gaussian Mixture Flow Matching (GMFlow)** 이라는 새로운 모델을 제안합니다. GMFlow는 평균을 직접 예측하는 대신, 다중 모드 흐름 속도 분포를 포착하기 위해 **동적 가우시안 혼합(dynamic Gaussian mixture, GM)** 의 매개변수를 예측하며, 이는 **KL 발산 손실(KL divergence loss)** 로 학습됩니다. 우리는 GMFlow가 단일 가우시안을 **L2 잡음 제거 손실(L2 denoising loss)** 로 학습하던 기존 확산 및 플로 매칭 모델을 일반화함을 보였습니다.

추론 단계에서 우리는 정확한 소수 단계 샘플링을 위해 분석적 잡음 제거 분포와 속도장을 활용하는 **GM‑SDE/ODE 해법(solvers)** 을 도출했습니다. 또한 CFG의 과채도 문제를 완화하고 이미지 생성 품질을 향상시키는 새로운 **확률적 가이던스 스킴(probabilistic guidance scheme)** 을 도입했습니다.

광범위한 실험 결과, GMFlow는 플로 매칭 기반 모델 대비 일관되게 우수한 생성 품질을 달성했으며, **ImageNet 256×256** 데이터셋에서 단 **6번의 샘플링 단계** 만으로 **Precision 0.942**를 기록했습니다.

![](/assets/images/posts/544/img.png)

![](/assets/images/posts/544/img_1.png)

### 1. 서론

**확산 확률 모델**(diffusion probabilistic models, Sohl‑Dickstein et al., 2015; Ho et al., 2020), **스코어 기반 모델**(score‑based models, Song & Ermon, 2019; Song et al., 2021b), **흐름 매칭 모델**(flow matching models, Lipman et al., 2023; Liu et al., 2022)은 공통된 이론적 틀을 공유하는 생성 모델 가계(家系)를 이룬다. 이들은 이미지·비디오 생성 분야에서 괄목할 만한 발전을 이루었다(Yang et al., 2023; Po et al., 2024; Rombach et al., 2022; Saharia et al., 2022b; Podell et al., 2024; Chen et al., 2024; Esser et al., 2024; Blattmann et al., 2023; Hong et al., 2023; HaCohen et al., 2024; Kong et al., 2025).

표준 **확산 모델(diffusion models)** 은 잡음 제거 분포를 가우시안으로 근사하고, 신경망을 학습시켜 그 **평균(mean)**(선택적으로 분산까지; Nichol & Dhariwal, 2021; Bao et al., 2022a,b)을 예측한다. **흐름 매칭 모델(flow matching models)** 은 이 가우시안 평균을 **흐름 속도(flow velocity)** 로 재매개변수화하여, 잡음에서 데이터로 매핑되는 **상미분 방정식(ODE)** 을 구성한다. 사용자 선호도 조사(Artificial Analysis, 2025; Jiang et al., 2024)에 따르면 이러한 정식화는 여전히 이미지 생성의 주류를 이룬다.

그러나 일반 확산·흐름 모델은 높은 품질의 생성을 위해 수십 단계의 샘플링이 필요하다. 이는 가우시안 근사가 작은 스텝 크기에서만 성립하고, 수치적 ODE 적분 과정에서 **이산화(discretization) 오차**가 발생하기 때문이다. 또한 고품질 생성을 위해서는 **분류기 프리 가이던스(Classifier‑Free Guidance, CFG)** 스케일을 크게 하는 것이 일반적이지만(Ho & Salimans, 2021), 스케일을 과도하게 키우면 **과채도(oversaturated) 색상**이 나타난다(Saharia et al., 2022a; Kynkäännienmi et al., 2024). 이는 **분포 밖 외삽(out‑of‑distribution extrapolation)** 현상(Bradley & Nakkiran, 2024) 때문이며, 수백 스텝을 사용해도 전체 이미지 품질을 제한한다.

### GMFlow의 제안

이 한계를 극복하기 위해 우리는 기존 **단일 가우시안 가정(single‑Gaussian assumption)** 에서 벗어나 **Gaussian Mixture Flow Matching (GMFlow)** 을 제안한다. 일반 흐름 모델이 흐름 속도의 **평균 u** 만을 예측하는 것과 달리, **GMFlow** 는 **가우시안 혼합 분포의 파라미터**를 예측하여 u의 **확률 밀도 함수(PDF)** 를 직접 모델링한다. 이는 두 가지 핵심 이점을 제공한다.

1. **정밀한 몇‑스텝 샘플링**  
   GM 정식화는 보다 복잡한 잡음 제거 분포를 포착하므로 큰 스텝 크기에서도 정확한 전이 추정을 가능케 하여, **적은 스텝으로도 고품질 생성**(그림 1)을 달성한다.
2. **안정된 조건부 가이던스**  
   CFG를 **외삽** 대신 **GM 확률 재가중(reweighting)** 으로 재정식화할 수 있어, 조건부 분포 내에서 샘플이 제한되고 과채도를 방지하여 전반적 이미지 품질을 개선한다.

### 학습·추론 방법

- **학습**: GMFlow는 **KL 발산 손실(KL divergence loss)** 로 예측한 속도 분포와 실제 분포를 정합시킨다. 이는 기존 확산·흐름 매칭 모델(L2 잡음 제거 손실)보다 일반화된 형태임을 § 3.1에서 보인다.
- **추론**: 예측된 GM으로부터 **역방향 전이 분포**와 **흐름 속도장**을 해석적으로 도출하는 **GM‑SDE/ODE 해법**을 새롭게 제안하여, 빠르고 정확한 소수 단계 샘플링을 가능케 한다(§ 3.3).  
  동시에 **확률적 가이던스(probabilistic guidance)** 기법을 고안해, 조건 정합성을 높이도록 GM PDF를 가우시안 마스크로 재가중한다(§ 3.2).

### 실험 결과

2차원 장난감 데이터셋과 **ImageNet**(Deng et al., 2009)에서 GMFlow를 일반 흐름 매칭 기법과 비교하였다. 폭넓은 실험 결과, GMFlow는 고급 솔버(Lu et al., 2022; 2023; Zhao et al., 2023; Karras et al., 2022)를 탑재한 베이스라인보다 항상 더 나은 성능을 보였다. 특히 **ImageNet 256×256** 에서 **8 스텝 미만**으로도 Precision과 FID 모두 우수했으며, **32 스텝**에서는 Precision 0.950의 최신(state‑of‑the‑art) 성능을 달성했다(그림 6 참조).

### 핵심 기여

- **GMFlow 제안**: 가우시안 혼합 기반 잡음 제거 분포로 확산 모델을 일반화한 새로운 흐름 매칭 프레임워크
- **GM 기반 샘플링 체계**: 새로운 GM‑SDE/ODE 솔버 및 확률적 가이던스로 구성된 효율적 샘플링 방법
- **경험적 검증**: 적은 스텝(few‑step)과 많은 스텝(many‑step) 모두에서 GMFlow가 기존 흐름 매칭 베이스라인을 능가함을 실험으로 입증

### 2. 확산(Diffusion) 및 플로 매칭(Flow Matching) 모델

이번 절에서는 **GMFlow**의 기반이 되는 확산 모델과 플로 매칭 모델의 배경을 설명한다. 실제로 두 방법은 상당 부분이 겹치기 때문에, 플로 매칭을 **특정 파라미터화(parameterization)를 취한 확산 모델**로 소개한다(Albergo & Vanden‑Eijnden, 2023; Gao et al., 2024).

![](/assets/images/posts/544/img_2.png)

![](/assets/images/posts/544/img_3.png)

![](/assets/images/posts/544/img_4.png)

![](/assets/images/posts/544/tfile.svg)

![](/assets/images/posts/544/img_5.png)

![](/assets/images/posts/544/img_6.png)

---

‘제한 사항’을 한눈에 이해하기

1. **두 가지 오류 원인**
   1. **숫자 계산 오차**
      - 이미지를 되살리는 과정을 여러 ‘스텝’으로 나누어 계산합니다.
      - 스텝을 **굵게** 잡으면 계산은 빠르지만, 매번 조금씩 틀려서 최종 이미지 품질이 떨어질 수 있습니다.
      - 스텝을 **촘촘히** 잡으면 정확해지지만, 계산 횟수(NFE)가 크게 늘어 시간이 오래 걸립니다.
   2. **모델 예측 오차**
      - 네트워크가 ‘얼마나 노이즈를 빼야 할지’를 완벽히 배우지 못하면 결과가 흐릿하거나 이상해집니다.
2. **Classifier‑Free Guidance(CFG)의 역할과 부작용**
   - **CFG** 는 “조건(예: 텍스트, 클래스 라벨)을 더 강하게 반영해!” 하고 모델 출력을 **섞어서**(외삽) 이미지를 개선하는 방법입니다.
   - **가이던스 스케일 w** 를 키우면(1 → 2 → 3 …)
     - 원하는 조건엔 더 잘 맞지만
     - 색이 과하게 진해지거나(Oversaturation) 다양성이 줄어들 수 있습니다.
   - 이런 과채도 문제는 보통 ‘값을 클리핑(thresholding)’ 하는 식의 임시방편으로 완화합니다.
3. **결국**
   - **스텝 수**를 줄이면 빨라지지만 계산 오차↑
   - **가이던스 w** 를 키우면 조건 정합성↑ / 과채도·다양성↓
   - 적절한 **타협**과 **휴리스틱(경험적 조정)** 이 필요합니다.

---

![](/assets/images/posts/544/img_7.png)

### 3.1 파라미터화와 손실 함수

![](/assets/images/posts/544/img_8.png)

**왜 Gaussian Mixture(가우시안 혼합)을 선택했는가?**  
모수화된 확률분포는 매우 다양하지만, 우리는 다음과 같은 장점 때문에 **Gaussian Mixture(GM)** 를채용하였다.

![](/assets/images/posts/544/img_9.png)

![](/assets/images/posts/544/img_10.png)

---

![](/assets/images/posts/544/img_11.png)

---

![](/assets/images/posts/544/img_12.png)

![](/assets/images/posts/544/img_13.png)

![](/assets/images/posts/544/img_14.png)

### 3.2 가우시안 혼합 재가중을 통한 **확률적 가이던스(Probabilistic Guidance)**

기존 **CFG(Classifier‑Free Guidance)** 는 **무한 외삽(unbounded extrapolation)** 으로 인해 색이 지나치게 진해지는(oversaturation) 문제가 있다. 이는 샘플이 실제 데이터 분포 범위를 벗어나면서 발생한다. 반면, **GMFlow** 는 잘 정의된 조건부 분포 qθ(u∣xt,c)를 제공하므로, **분포의 본래 경계와 구조를 그대로 유지하면서 가중치만 조정**하는 **확률적 가이던스**를 설계할 수 있다.

![](/assets/images/posts/544/img_15.png)

![](/assets/images/posts/544/img_16.png)

![](/assets/images/posts/544/img_17.png)

### 3.3 GM‑SDE 및 GM‑ODE 솔버

이번 절에서는 **GMFlow** 가 **역전이 분포(reverse transition distribution)** 와 **흐름 속도장(flow velocity field)** 를 해석적으로 도출하여, **이산화 오차(discretization errors)** 를 크게 줄여 주는 고유한 **SDE·ODE 솔버**를 가능하게 함을 설명한다.

![](/assets/images/posts/544/img_18.png)

![](/assets/images/posts/544/img_19.png)

![](/assets/images/posts/544/img_20.png)

![](/assets/images/posts/544/img_21.png)

![](/assets/images/posts/544/img_22.png)

**4. 실험(Experiments)**  
GMFlow의 성능을 검증하기 위해, 우리는 기존 바닐라 플로 매칭(vanilla flow matching) 베이스라인과 다음 두 가지 데이터셋에서 비교 실험을 수행하였다.  
(a) **2차원 체커보드 분포**(simple 2D checkerboard distribution) – 샘플 히스토그램을 시각화하고 내부 동작 원리를 분석하기에 적합한 간단한 예제.  
(b) **클래스 조건부 ImageNet**(class‑conditioned ImageNet, Deng et al., 2009) – 대규모 이미지 생성에서 GMFlow의 실제 장점을 확인할 수 있는 고난도 벤치마크.

#### 4.1 2차원 체커보드 분포 샘플링

이 절에서는 Lipman 등(2023)의 실험 설정을 따르며, **2차원 체커보드 분포**에서 GMFlow와 바닐라 플로 모델(vanilla flow model)을 비교한다.  
모든 설정은 **5‑레이어 MLP** 아키텍처를 사용해 2D 좌표를 디노이즈하며, 단지 **출력 채널 수**만 다르다.

- **GMFlow**의 경우, 서로 다른 혼합 성분 수 K 를 갖는 여러 모델을 학습하였고(전이 비율 λ=0.9 사용)
- **GM‑ODE** 샘플링에서는 n=⌈128/NFE⌉개의 서브‑스텝(sub‑steps)을 사용하였다.

![](/assets/images/posts/544/img_23.png)

**그림 2.** 다양한 솔버를 적용한 바닐라 플로 모델과 GMFlow의 비교.  
SDE와 ODE 두 경우 모두, GMFlow는 **적은 스텝(few‑step) 샘플링**에서도 더 높은 품질을 보여준다.

**플로 모델 베이스라인과의 비교**  
그림 2에서는 **2차 GM 솔버(GM‑SDE/ODE 2)** 를 사용한 **GMFlow**의 2D 샘플 히스토그램을, 기존 **SDE·ODE 솔버**(Lu et al., 2023; Zhao et al., 2023; Ho et al., 2020; Song et al., 2021a)를 적용한 **바닐라 플로 매칭(vanilla flow matching) 베이스라인**과 비교하였다.

- 바닐라 플로 모델은 **합리적인(histogram이 크게 깨지지 않는) 품질**을 얻는 데 약 **8 스텝**, **고품질**을 얻는 데는 **16–32 스텝**이 필요했다.
- 반면 **GMFlow(K = 64)** 는 **단 1 스텝**만으로도 체커보드 분포를 근사할 수 있었고, **4 스텝**에서 고품질 샘플을 생성했다.
- 또한 바닐라 플로 모델의 경우, **1차 솔버(first‑order solvers, DDPM·Euler)** 로 생성한 샘플은 중앙으로 몰리고, **2차 솔버(second‑order solvers, DPM++·UniPC)** 는 외곽 가장자리에 집중되는 경향이 있었다.
- 이에 비해 GMFlow의 샘플은 **전 영역에 고르게 분포**하였다.

이 결과는 **GMFlow가 소수 스텝(few‑step) 샘플링과 다중 모드 분포(multi‑modal distribution) 모델링에서 우수함**을 입증한다.

![](/assets/images/posts/544/img_24.png)

![](/assets/images/posts/544/img_25.png)

**GM 구성 요소 수 K의 영향**  
그림 3 (a)는 K 값을 늘리면 **소수 단계 샘플링(few‑step sampling)** 품질이 크게 향상되어 히스토그램이 훨씬 날카로워짐을 보여 준다. 특히 K=1인 **GM‑SDE 샘플링**은 이론적으로 **학습된 분산(learned variance)** 을 사용하는 **DDPM**(Nichol & Dhariwal, 2021)과 동등하므로, 출력 히스토그램이 다소 흐릿하게 나타난다. 이는 **GMFlow의 높은 표현력(expressiveness)** 을 다시 한 번 강조한다.

![](/assets/images/posts/544/img_26.png)

**그림 4.** **GMFlow‑DiT**의 아키텍처. 원본 **DiT**(Peebles & Xie, 2023)는 파란색으로, 수정된 출력 레이어는 보라색으로 표시되어 있다.

**두 번째 차수(Second‑order) GM 솔버**  
그림 3 (a)에서는 서로 다른 **NFE**(Number of Function Evaluations)와 **GM 성분 개수 K** 에 대해 1차(왼쪽 열)와 2차(오른쪽 열) GM 솔버를 비교한다. 2차 솔버는 1차 솔버보다 히스토그램이 더욱 뚜렷하고 이상치(outlier)를 효과적으로 억제하며, 특히 **NFE와 K가 작을 때** 그 차이가 두드러진다. 반면 K=8 이상이면 GM 확률이 이미 충분히 정확해 **2차 솔버가 추가적인 이점을 보이지 않으며**, 이는 우리의 이론적 분석과 일치한다.

**절삭 실험(Ablation studies)**  
2차 GM 솔버에서 **q^(x0∣xt) 변환**의 중요성을 검증하기 위해, 이 변환을 제거하고 직전 두 스텝의 x\_0​ 분포만을 직접 외삽(extrapolation)하는 실험을 수행했다(DPM++ 방식과 유사, Lu et al., 2023). 그림 3 (b)에 나타나듯, 변환을 제거하면 샘플이 분포 경계를 넘어서 과도하게 치우치는 **오버슈팅(overshooting) 편향**이 발생하며, 이는 DPM++ 2M SDE(NFE = 8)와 유사한 현상이다. 결과적으로, 해당 변환이 **샘플 균일성 유지에 필수적**임을 확인할 수 있다.

![](/assets/images/posts/544/img_27.png)

**그림 5.** **GMFlow(GM‑ODE 2)** 와 바닐라 플로 모델 베이스라인(UniPC, Euler)을 최적 Precision(정밀도) 설정에서 정성적으로 비교. GMFlow는 다양한 NFE에서도 **일관된** 결과를 내지만, 베이스라인 모델들은 적은 스텝일 때 구조가 왜곡되는 등 품질 저하를 겪는다.

![](/assets/images/posts/544/img_28.png)

![](/assets/images/posts/544/img_29.png)

![](/assets/images/posts/544/img_30.png)

![](/assets/images/posts/544/img_31.png)

**플로 모델 베이스라인과의 비교**  
베이스라인으로는 바닐라 플로 모델에 다음과 같은 솔버를 적용해 시험했다.

- **1차(1‑st order) 솔버**: DDPM(Ho et al., 2020), Euler(= DDIM, Song et al., 2021a)
- **고급 2차(2‑nd order) 솔버**: DPM++(Lu et al., 2023), DEIS(Zhang & Chen, 2023), UniPC(Zhao et al., 2023), SA‑Solver(Xue et al., 2023)

플로 매칭에 이들 솔버를 적용한 방식은 § A.5에 자세히 설명돼 있다.

그림 6 (a)는 **2차 GM 솔버(GM‑SDE/ODE 2)** 를 장착한 **GMFlow(K = 8)** 와 위 베이스라인들을 비교한 결과이다.

- **Precision**: GMFlow는 SDE/ODE 양쪽 모두에서, 다양한 NFE(함수 평가 횟수) 구간에 걸쳐 일관되게 가장 높은 Precision을 기록했다. 특히 GM‑ODE 2는 **단 6 스텝** 만에 Precision 0.942에 도달했고, GM‑SDE 2는 **32 스텝**에서 **최신 최고 수준(State‑of‑the‑Art) Precision 0.950**을 달성했다.
- **FID**: 소수 스텝(NFE ≤ 8) 구간에서는 GMFlow가 베이스라인보다 확연히 우수하며, 다수 스텝(NFE ≥ 32) 구간에서도 경쟁력을 유지했다.

그림 6 (b)의 Precision–Recall 곡선은 1차·2차 GM 솔버가 각각의 베이스라인 솔버보다 항상 우수한 품질–다양성(trade‑off)을 보여 줌을 시각적으로 확인시켜 준다. 정성적 비교는 그림 5에 제시돼 있다.

**채도(Saturation) 평가**  
표 2는 각 방법이 **최고 Precision 설정**에서 기록한 **Saturation** 지표를 비교한다. **GMFlow(K = 8)** 는 과채도(oversaturation)를 효과적으로 줄여 **실제 데이터에 가장 근접한 Saturation 수준**을 보였다. 그림 5의 시각적 비교 역시 이를 뒷받침한다.

또한 **정직교 투영(orthogonal projection)** 기법을 제거한 절삭 실험에서도 동일한 경향이 나타났다. 즉, 투영을 사용하지 않아도 GMFlow가 가장 낮은(가장 좋은) Saturation 값을 유지했으며, 베이스라인들은 투영이 없을 때 오히려 더 나쁜 결과를 보였다.

**표 2.** ImageNet 평가 결과 (최고 Precision, NFE = 32). 보고된 Saturation 값(Sadat et al., 2024)은 실제 데이터의 통계량(기준 Saturation = 0.318)과 비교한 상대 지표이다.

![](/assets/images/posts/544/img_32.png)

### GM 구성요소 수 KKK의 영향

그림 7은 GM 구성요소의 개수가 각 평가지표에 미치는 영향을 보여준다.

- **Few‑step 샘플링**(NFE = 8)에서는, 특히 **GM‑SDE 2** 솔버 사용 시 K를 늘릴수록 성능 향상이 가장 뚜렷하다.
- 대부분의 지표는 **K=8** 부근에서 **포화(saturation)** 상태에 도달한다.
- 그 이상으로 K를 늘리면, 가우시안 성분 수가 많아질수록 **스펙트럴 샘플링** 과정에서 수치 오차가 커져 **GM‑SDE 2**의 Precision이 오히려 감소한다.

![](/assets/images/posts/544/img_33.png)

![](/assets/images/posts/544/img_34.png)

**그림 7.** GM 구성요소 수 K에 따라 변화하는 GMFlow 모델의 최고 FID와 최고 Precision.

**표 3.** 서로 다른 GMFlow 설정에서 ImageNet 학습 이미지(잠복 표현, latents)의 검증용 **음의 로그 우도**(negative log‑likelihood, **NLL**).

![](/assets/images/posts/544/img_35.png)

**절삭(실험) 결과(Ablation studies)**  
표 4는 우리 방법의 다양한 설계 요소가 성능에 미치는 영향을 보여준다.

- **GM‑SDE**: 스펙트럼 샘플링을 제거(A2)하면 FID가 눈에 띄게 악화된다.
- **확률적 가이던스 → 일반 CFG로 교체(A3)**: 예측된 GM의 평균만 단순히 이동시키는 방식으로 대체하면 Precision이 크게 떨어지고 과채도(oversaturation)가 심해진다(그림 8에서도 확인 가능).
- **GM‑SDE 2 → 2차 DPM++ SDE 솔버로 교체(A4)**: FID와 Precision이 모두 악화된다.
- **전이 손실 → 원래 KL 손실로 축소(A5)**: FID가 저하된다.
- **GM‑ODE 2**: 서브‑스텝 없이 ODE 적분을 바로 적용(B1)하면 FID가 크게 나빠진다.

**추론 시간(Inference time)**  
GMFlow는 출력 레이어만 변경하고, 솔버 역시 단순한 산술 연산으로 구성되어 있다. 그 결과, 플로 매칭 모델 대비 스텝당 **약 0.005 초**(배치 125, A100 GPU)의 오버헤드만 추가된다. 이는 스텝당 총 추론 시간 **0.39 초**(대부분 DiT 연산)에 비해 극히 미미한 수준이다.

### 5. 관련 연구 (Related Work)

기존 연구들 (Nichol & Dhariwal, 2021; Bao et al., 2022a,b)​은 **표준 확산 모델**(standard diffusion models)의 잡음 제거 분포 **분산(variance)** 을 학습하도록 확장했다. 이는 사실상 **GMFlow** 에서 K=1, s만 학습 가능한 특수 사례에 해당한다. **GMS**(Guo et al., 2023)는 이 아이디어를 3차 모멘트(third‑order moments)까지 확장해, 추론 시 양봉(bimodal) GM을 맞춘다. § B.3에서는 **GMFlow (K=2)** 와 **GMS** 를 비교하였다. 반면 Xiao et al.(2022)는 **GAN**(Generative Adversarial Network, Goodfellow et al., 2014)을 사용해 더 표현력 높은 분포를 학습하지만, **adversarial diffusion**은 일반적으로 품질·안정성을 위해 추가 목적함수를 필요로 한다(Jolicoeur‑Martineau et al., 2021; Kim et al., 2024). 이들 방법은 **결정론적 ODE 샘플링** 문제는 다루지 않는다.

최근에는 네트워크가 ODE 적분 결과나 **디노이즈된 샘플(denoised samples)** 자체를 직접 예측하도록 학습해 **매우 적은 스텝(few‑step)** 으로도 샘플링이 가능하도록 하는 연구가 활발하다(Salimans & Ho, 2022; Song et al., 2023; Yin et al., 2024b; Sauer et al., 2024). 이러한 방법은 일종의 **증류(distillation) 기법**으로, 추가적인 적대적 학습(adversarial training) 없이는 품질이 원래 다수 스텝 모델의 성능을 넘기 어렵다(Kim et al., 2024; Yin et al., 2024a).

CFG를 단순 임계값(thresholding, Saharia et al., 2022a)이나 **정직교 투영(orthogonal projection)**(Sadat et al., 2024) 이상으로 개선하려는 시도도 있다. 예를 들어 **초기 스텝에서 CFG를 끄기**(Kynkäännienmi et al., 2024)는 Precision을 희생시키고, **랑주뱅(Langevin) 보정**을 추가하는 방법(Bradley & Nakkiran, 2024)은 효율을 낮춘다.

확산 모델을 넘어, **가우시안 혼합(Gaussian mixture)** 은 다른 생성 모델에도 활용된다. **DeLiGAN**(Gurumurthy et al., 2017)과 **GM‑GAN**(Ben‑Yosef & Weinshall, 2018)은 GAN의 잠재 공간(latent space)에 혼합 사전(mixture prior)을 도입해 표현력을 높인다. **GIVT**(Tschannen et al., 2024)는 **오토리그레시브 트랜스포머(autoregressive Transformer)** 의 출력을 가우시안 혼합으로 모델링해, 양자화(quantization) 기반 방식보다 뛰어난 성능으로 연속 데이터 샘플링을 가능하게 한다.

**표 4.** ImageNet 평가 기준으로 수행한 GMFlow 절삭 실험 결과(설정: K=8, NFE = 8)

![](/assets/images/posts/544/img_36.png)

![](/assets/images/posts/544/img_37.png)

**그림 8.** 확률적 가이던스 절삭 실험.표 4의 A2(스펙트럴 샘플링 포함)과 A3(바닐라 CFG 대체) 샘플 비교.

### 6. 결론(Conclusion)

본 논문에서는 **GMFlow**를 제안하였다. GMFlow는 확산·플로 모델을 일반화하여 **흐름 속도(flow velocity)** 를 **가우시안 혼합 분포(Gaussian mixture)** 로 표현함으로써, 복잡한 다중 모드 구조를 더욱 풍부하게 포착할 수 있다. 우리는 이 모델에 특화된 SDE/ODE 솔버를 이론적으로 도출하고, 과채도를 제거하는 **확률적 가이던스(probabilistic guidance)** 기법을 제안했다.  
광범위한 실험을 통해 GMFlow가 **소수 스텝(few‑step) 생성 성능**을 크게 향상시키고, 전반적인 샘플 품질도 높임을 입증했다. 이러한 프레임워크는 **GMFlow prior를 이용한 사후 샘플링(posterior sampling)** 등, 이론·실무 양 측면에서 후속 연구의 기반을 마련한다.

### 한계(Limitations)

고차원 이미지 데이터에 가우시안 혼합을 적용하기 위해 **픽셀 단위 분해(pixel‑wise factorization)** 를 사용했지만, 이는 GMFlow의 잠재력을 완전히 활용하지 못할 수 있다. 향후 개선의 여지가 있다.

### 감사(Acknowledgments)

Liwen Wu, Lvmin Zhang, Guandao Yang께 유익한 토론과 피드백을 감사드립니다. Hansheng은 본 연구의 일부를 Adobe Research 인턴 기간 중 수행했습니다. 본 프로젝트는 **Qualcomm Innovation Fellowship**, **ARL grant W911NF‑21‑2‑0104**, **Vannevar Bush Faculty Fellowship**의 지원을 일부 받았습니다.

### 영향 진술(Impact Statement)

본 연구는 생성 모델 분야의 발전을 목표로 한다. 사회적 파급 효과가 다양할 수 있으나, 특별히 강조해야 할 부정적 결과는 현재로서는 파악되지 않았다.
