---
title: "NeRFactor: neural factorization of shape and reflectance under an unknown illumination"
date: 2025-01-17 23:23:54
categories:
  - 인공지능
tags:
  - Computer Vision
  - nerfactor
---

<https://dl.acm.org/doi/10.1145/3478513.3480496>

[NeRFactor: neural factorization of shape and reflectance under an unknown illumination: ACM Transactions on Graphics: Vol 40, No

We address the problem of recovering the shape and spatially-varying reflectance of an object from multi-view images (and their camera poses) of an object illuminated by one unknown lighting condition. This enables the rendering of novel views of the ...

dl.acm.org](https://dl.acm.org/doi/10.1145/3478513.3480496)

![](/assets/images/posts/494/img.png)

**그림 1. Neural Radiance Factorization (NeRFactor).** 하나의 미지 조명 조건만 주어진 물체의 다중 뷰 이미지(및 해당 카메라 자세)가 (왼쪽) 주어졌을 때, NeRFactor는 장면의 외관을 표면 노멀, 조명 가시성, 알베도, 반사 특성과 같은 3D 신경 필드로 분해(가운데)할 수 있다. 이를 통해 자유 시점 재조명(그림자 처리 및 소재 편집 포함) 같은 다양한 응용(오른쪽)이 가능해진다.

우리는 하나의 알려지지 않은 조명 조건에서 촬영된 물체의 다중 뷰 이미지(및 해당 카메라 자세)를 바탕으로, 물체의 형태와 공간적으로 변화하는 반사 특성을 복원하는 문제를 다룬다. 이를 통해 임의의 환경 조명 아래에서 물체의 새로운 시점을 렌더링하고, 물체의 재질 특성을 편집할 수 있게 된다. 우리의 접근 방식인 Neural Radiance Factorization(NeRFactor)의 핵심은, [Mildenhall et al. 2020]에서 제안된 Neural Radiance Field(NeRF)의 부피(Volumetric) 형상 표현을 표면 표현으로 추출한 뒤, 형상을 정밀하게 개선하는 동시에 공간적으로 변화하는 반사 특성과 환경 조명을 함께 추정하는 것이다.

구체적으로 NeRFactor는, 재렌더링 손실과 간단한 평활화 사전(prior), 그리고 실제 BRDF 측정에서 학습된 데이터 기반 BRDF 사전만을 활용하여, 어떠한 별도 지도 없이도 표면 노멀, 조명 가시성, 알베도, 양방향 반사 분포 함수(BRDF)의 3D 신경 필드를 복원한다. 특히 조명 가시성을 명시적으로 모델링함으로써, 알베도에서 그림자를 분리하고, 임의의 조명 조건에서 사실적인 부드러운 혹은 강한 그림자를 합성해 낼 수 있다. 이러한 복잡하고 제약이 많은 캡처 환경에서도, NeRFactor는 합성 및 실제 장면 모두에 대해 자유 시점 재조명이 가능한 설득력 있는 3D 모델을 복원하는 데 성공한다. 정성적·정량적 실험 결과에 따르면, NeRFactor는 여러 과업에서 기존의 고전적 접근 방식 및 딥러닝 기반 최신 기법보다 우수한 성능을 보인다. 관련 비디오, 코드, 데이터는 아래 웹사이트에서 확인할 수 있다.  
[**people.csail.mit.edu/xiuming/projects/nerfactor/**](http://people.csail.mit.edu/xiuming/projects/nerfactor/)

[https://people.csail.mit.edu/xiuming/projects/nerfactor/

people.csail.mit.edu](http://people.csail.mit.edu/xiuming/projects/nerfactor/)

**CCS Concepts:**  
• Computing methodologies → Rendering; Computer vision;

**추가 주요 단어 및 문구:**  
inverse rendering, appearance factorization, shape estimation, reflectance estimation, lighting estimation, view synthesis, relighting, material editing

**ACM Reference Format:**  
Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul Debevec, William T. Freeman, and Jonathan T. Barron. 2021. NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination. ACM Trans. Graph. 40, 6, Article 237 (December 2021), 18 pages. https://doi.org/10.1145/3478513.3480496

**1 서론(Introduction)**  
카메라로 촬영한 이미지에서 물체의 기하(geometry)와 재질(material) 특성을 복원하고, 임의의 시점과 새로운 조명 조건에서 렌더링 가능하게 만드는 문제는 컴퓨터 비전 및 그래픽스 분야에서 오랜 숙제다. 이 문제의 어려움은 근본적으로 정보가 부족한(underconstrained) 성격에 기인하며, 기존 연구들은 일반적으로 스캔된 기하 정보나 알려진 조명 조건, 혹은 서로 다른 여러 조명 환경에서 촬영된 이미지 같은 추가 정보를 이용하거나, 물체 전체를 하나의 단일 재질로 가정하거나 자체 그림자(self-shadowing)를 무시하는 등의 제한적인 가정을 적용해 왔다.

본 연구에서는 그림 1과 같이, **하나의 미지 자연 조명 조건**에서 촬영된 물체 이미지만으로도 신뢰할 만한 재조명(relightable) 표현을 복원할 수 있음을 보인다. 우리의 핵심 아이디어는 먼저 입력 이미지에서 Neural Radiance Field(NeRF) Mildenhalletal.2020를 최적화하여 모델의 표면 노멀(surface normals)과 조명 가시성(light visibility)을 초기화(여기서 Multi-View Stereo(MVS) 기하 정보를 이용해도 동작함을 보임)하고, 그런 다음 관측된 이미지를 가장 잘 설명하도록 이 초기 추정값과 공간적으로 변화하는 반사 특성, 그리고 조명 조건을 공동으로 최적화하는 것이다. 초기화에 NeRF가 제공하는 고품질 기하 추정 결과를 활용함으로써, 형상·반사 특성·조명 간의 모호성을 크게 줄이고, 간단한 재렌더링 손실, 각 구성 요소에 대한 평활성(smoothness) 사전(prior), 그리고 새로운 데이터 기반 양방향 반사 분포 함수(BRDF) 사전만을 사용하여 설득력 있는 3D 모델을 복원해 자유 시점(view synthesis) 렌더링과 재조명을 가능하게 만든다. 또한 NeRFactor는 조명 가시성을 명시적이고 효율적으로 모델링하기 때문에, 알베도 추정에서 그림자를 제거하고, 임의의 새로운 조명 조건 하에서 사실적인 부드럽거나 날카로운 그림자를 합성할 수 있다.

NeRF가 추정한 기하는 시점 합성(view synthesis)에 유효하지만, 재조명에 바로 활용하기에는 두 가지 제한이 있다.  
첫째, NeRF는 물체 형태를 볼륨 필드(volumetric field)로 모델링하기 때문에, 카메라 광선(레이)을 따라 모든 점에 대해 반구(hemisphere) 전체의 조명에 대한 음영(shading)과 가시성을 계산하려면 매우 많은 연산이 필요하다.  
둘째, NeRF가 추정한 기하에는 고주파(high-frequency) 성분이 다소 포함되어 있어, 시점 합성 결과에서는 잘 드러나지 않더라도, 해당 기하로부터 계산한 표면 노멀과 조명 가시성에는 고주파 잡음이 생길 수 있다.

---

NeRF 같은 볼륨 기반 기법으로 학습된 기하 정보에는, 원본 이미지가 제한된 해상도 혹은 시점으로 촬영되었을 때 발생하는 ‘과적합(overfitting)’ 혹은 ‘불필요한 세부 표현’이 반영될 수 있습니다. 이는 실제로는 매끄럽게 이어져야 할 표면이 아주 미세하게 요철(凸凹) 형태로 흔들리는 현상(=고주파 성분)을 만들어냅니다. 시점 합성(view synthesis)에서 이러한 미세 굴곡은 화면에 크게 두드러지지 않을 수 있지만, 조명 시뮬레이션에서는 표면 노멀이나 조명 가시성 계산에 직접 영향을 주어 **작은 기하 왜곡**도 **고주파 잡음**으로 크게 확대됩니다. 예컨대 표면이 실제보다 더 울퉁불퉁하게 인식되거나, 광선이 표면과 만나는 위치가 부분적으로 잘못 계산되면, 재조명(relighting) 시 그림자나 하이라이트가 어색하게 나타날 수 있습니다. 따라서 재조명을 위한 실제감 있는 표면 노멀과 그림자 표현을 얻으려면, NeRF가 추정한 기하에서 이런 고주파 잡음을 제거하거나 적절히 보정해주는 과정이 필요합니다.
---

우리는 첫 번째 문제를 해결하기 위해 NeRF 기하에 대해 “경질(hard surface)” 근사 기법을 사용한다. 이는 각 레이에서 볼륨이 종료되는 깊이(termination depth) 지점만을 대표로 삼아 음영 계산을 단 한 번만 수행하는 방식이다. 두 번째 문제에 대해서는, 표면의 임의의 3D 위치에서 표면 노멀과 조명 가시성을 다층 퍼셉트론(MLP)으로 연속적인 함수로 나타내고, 이 함수가 학습된 NeRF에서 얻은 값과 크게 어긋나지 않으면서도 공간적으로 매끄럽게 유지되도록 유도한다.  
이로써 우리가 “Neural Radiance Factorization(NeRFactor)”라고 부르는 모델은 관측 이미지를 추정된 환경 조명(environment lighting)과 물체의 3D 표면 표현(표면 노멀, 조명 가시성, 알베도, 공간적으로 변화하는 BRDF)으로 분해한다. 이를 통해 **임의의 새로운 환경 조명**에서 물체의 새로운 시점을 렌더링할 수 있게 된다.

### 요약하자면, 본 논문의 주요 기술적 기여는 다음과 같다:

- **알려지지 않은 조명 조건**에서 촬영된 물체 이미지를 형상, 반사 특성, 조명으로 분해하여 **자유 시점 재조명(그림자 포함)과 재질 편집**을 지원하는 방법을 제안한다.
- **NeRF로부터 추정한 볼륨 밀도**를 표면 기하(노멀, 조명 가시성)로 변환하여, 형상을 개선하고 반사 특성을 복원할 때 초기값으로 사용하는 전략을 제시한다.
- **실제 측정된 BRDF 데이터**를 이용해 잠재 표현(latent code) 모델을 학습한, **새로운 데이터 기반 BRDF 사전**을 개발한다.

### 입력과 출력

NeRFactor의 입력은 **하나의 미지 환경 조명** 아래에서 촬영된 물체의 다중 뷰 이미지 세트와 각 이미지의 카메라 자세다. NeRFactor는 관측된 영상을 설명할 수 있는 표면 노멀, 조명 가시성, 알베도, 공간적으로 변화하는 BRDF, 그리고 환경 조명(condition)을 동시에 추정한다. 이후 복원된 기하와 반사 특성을 기반으로, 물체를 **임의의 시점과 다양한 조명 조건**에서 새롭게 합성한다. 가시성(visibility)을 명시적으로 모델링하기 때문에, NeRFactor는 알베도에서 그림자를 제거하고, 원하는 조명 아래에서 부드러운 그림자부터 날카로운 그림자까지 사실적으로 합성할 수 있다.

### 가정(Assumptions)

NeRFactor는 물체가 **단일 표면(hard surface)**으로 구성되어 있고 각 광선이 표면과 한 번만 교차한다고 가정하기 때문에, 산란(scattering), 투명(transparency), 반투명(translucency) 같은 볼륨 광 전송(volumetric light transport) 효과는 모델링하지 않는다. 또한 계산 간소화를 위해 **직접 광원(direct illumination)**만을 고려한다. 마지막으로, 반사 모델은 **무채색 스펙큘러(reflectance)**(dielectric 재질)만 다루므로, 금속성(material) 재질은 명시적으로 모델링하지 않는다(단, 표면마다 스펙큘러 색상을 추가로 예측하도록 확장하면 금속 재질도 처리 가능하다).

**2 관련 연구(Related Work)**  
Inverse rendering [Satoetal.1997;Marschner1998;Yuetal.1999;RamamoorthiandHanrahan2001]은 관측된 이미지에서 물체의 외관을 기하(geometry), 재질(material), 조명 조건으로 분해하는(appearance factorization) 오랜 연구 과제다. 문제 설정이 본질적으로 매우 정보가 부족한(underconstrained) 성격이므로, 기존 접근법 대부분은 그림자가 없다고 가정하거나, 형상·조명·반사 특성에 대한 사전(prior)을 학습하거나, 스캔된 기하 정보나 측정된 조명 조건, 혹은 여러 (알려진) 조명 환경에서 물체를 촬영한 추가 이미지를 요구하는 방식을 취해왔다.

싱글 이미지 기반 inverse rendering [BarronandMalik2014;Lietal.2018;Senguptaetal.2019;YuandSmith2019;SangandChandraker2020;Weietal.2020;Lietal.2020;Lichyetal.2021] 기법 다수는, 대규모 데이터셋으로부터 학습된 기하·반사·조명에 대한 강력한 사전 지식(prior)에 크게 의존한다. 최근 방법들은 단일 이미지에서도 꽤 그럴듯한 기하·재질·조명 값을 추론할 수 있으나, 이들은 임의 시점에서 볼 수 있는 완전한 3D 표현을 복원하지는 못한다.

반면, 재조명(relighting)과 시점 합성(view synthesis)이 가능한 3D 모델을 회복하려는 접근법 대다수는, 강력한 사전 지식 대신 추가적인 관측 정보를 이용해왔다. 예를 들어, 액티브 스캐닝(active scanning)으로 얻은 3D 기하 [Lenschetal.2003;Guoetal.2019;Parketal.2020;Schmittetal.2020;Zhangetal.2021a], 프록시(proxy) 모델 [Satoetal.2003;Dongetal.2014;Georgoulisetal.2015;Gaoetal.2020;Chenetal.2020], 실루엣(silhouette) 마스크 [OxholmandNishino2014;Godardetal.2015;Xiaetal.2016], 혹은 다중 뷰 스테레오(Multi-View Stereo, MVS)로 획득한 기하 정보를(이후 표면 재구성과 메싱을 통해) [Laffontetal.2012;Nametal.2018;Philipetal.2019;Goeletal.2020] 활용해 반사 특성과 정교한 기하를 추정하는 식이다. 본 연구에서는 최신 신경 볼륨(neural volumetric) 표현을 통해 추정한 기하를 출발점으로 삼으면, 단 하나의 조명 상태에서 촬영한 이미지들만으로도 충분히 분해된(factorized) 3D 모델을 복원할 수 있음을 보인다. 특히 이러한 기하 초기화 방식을 통해, 매우 반사율이 높은 표면이나 세부 기하가 복잡하여 기존 기하 추정 방법으로는 어려웠던 물체들에 대해서도 효과적인 분해 결과를 얻을 수 있음을 확인했다.

컴퓨터 그래픽스 분야에서는 오랜 기간 재질 취득(material acquisition) 문제, 즉 (주로 평면 형태의) 기하가 이미 알려진 상태에서 양방향 반사 분포 함수(BRDF)를 추정하는 특정 하위 과제에 많은 관심이 있었다. 이 문제에서는 보통 신호 처리(signal processing) 기반의 재구성 기법과 복잡한 카메라·조명 세팅을 사용해 BRDF를 충분히 샘플링해왔다 [Foo2015;Matusiketal.2003;Nielsenetal.2015]. 최근에는 스마트폰 정도의 간단한 장비로도 재질 취득을 가능하게 하려는 연구도 진행되고 있다 [Aittalaetal.2015;Huietal.2017]. 그러나 이들 연구는 기하가 단순하며 완전히 알려져 있다고 가정하는 경우가 많으며, 본 논문에서 다루는 문제는 복잡한 형상과 공간적으로 변화하는 반사 특성을 갖는 물체의 이미지만 주어진 상황을 다룬다.

본 연구는 컴퓨터 비전 및 그래픽스 분야에서 최근 주목받는 추세인 **다층 퍼셉트론(MLP)을 이용해 기하를 연속 함수로 표현**하는 접근법을 토대로 한다. 이 접근법에서는 다층 퍼셉트론(MLP)이 3D 좌표를 입력받아 그 위치에 대한 물체나 장면의 특성(볼륨 밀도, 점유도, 서명 거리 함수(signed distance) 등)을 반환하도록 최적화되어, 기존 폴리곤 메시(mesh)나 3D 격자(voxel grid)를 대체한다. 이러한 방법은 3D 관측으로부터 연속적인 3D 형상 표현을 복원하는 데 [Meschederetal.2019;Parketal.2019;Tanciketal.2020], 그리고 고정된 조명 아래 촬영된 이미지를 이용해 형상을 복원하는 데 [Mildenhalletal.2020;Yarivetal.2020] 성공적으로 활용되어 왔다. 특히 Neural Radiance Fields(NeRF) [Mildenhalletal.2020]는 볼륨 기하와 외관을 최적화하여, 관측 이미지를 기반으로 사실적인 새로운 시점 이미지를 렌더링하는 데 매우 뛰어난 성능을 보인다.

NeRF를 확장해 재조명을 지원하려는 여러 시도가 최근 제안되었다 [Bietal.2020;Bossetal.2021;Srinivasanetal.2021;Zhangetal.2021b]. 이와 동시대(concurrent) 연구들과 본 논문에서 제안하는 NeRFactor의 차이점은 다음과 같다.

- Bi et al. 2020와 NeRV Srinivasanetal.2021는 **여러 알려진 조명 조건**에서 촬영한 이미지가 필요하지만, NeRFactor는 **단 하나의 미지 조명**만 주어진 상황을 다룬다.
- NeRD Bossetal.2021는 가시성과 그림자를 모델링하지 않으나, NeRFactor는 이를 명시적으로 모델링해 그림자와 알베도를 분리한다(뒤에서 보이듯이). 또한 NeRD는 분석적(analytic) BRDF를 사용하고, NeRFactor는 사전 지식을 내포한 학습 기반 BRDF를 사용한다.
- PhySG Zhangetal.2021b 역시 가시성과 그림자를 모델링하지 않고, 분석적 BRDF를 쓴다는 점에서 NeRD와 유사하다. 게다가 PhySG는 **공간적으로 균일한(non-spatially-varying) 반사**만 가정하지만, NeRFactor는 **공간적으로 변화하는(spatially-varying) BRDF**를 모델링한다.

**3 방법(Method)**

![](/assets/images/posts/494/img_1.png)

∗ 옮긴이 주: 관측된 외관(appearance)을 설명한다는 것은, 물체 표면의 광학적 속성(반사, 그림자, 조명 등)을 고려하여 실제 이미지와 일치하도록 모델링한다는 의미.

그림 2에는 NeRFactor 모델 구조와, 이 모델이 수행하는 외관 분해의 예시를 시각적으로 제시했다. 네트워크 아키텍처, 학습 방식, 실행 시간 등의 구현 세부 사항은 부록(Section A)과 우리의 GitHub 저장소에서 확인할 수 있다.

![](/assets/images/posts/494/img_2.png)

![](/assets/images/posts/494/img_3.png)

![](/assets/images/posts/494/img_4.png)

![](/assets/images/posts/494/img_5.png)

**그림 2**. NeRFactor는 한 가지 미지 조명 조건에서만 관측된 장면 외관을 비지도(unsupervised)로 분해하는 좌표 기반(coordinate-based) 모델이다. 이와 같이 정보가 매우 부족한 문제를, 재구성 손실(reconstruction loss), 단순한 평활화 정규화, 데이터 기반 BRDF 사전을 활용해 해결한다. 특히 가시성을 명시적으로 모델링하여, NeRFactor는 물리적으로 타당한 방식의 임의 조명에서의 그림자 표현을 지원한다.

**3.1 기하(Shape)**  
우리 모델의 입력은 NeRF [Mildenhalletal.2020]에서 사용하는 것과 동일하므로, 모델 초기 기하 초기화를 위해 NeRF를 동일하게 적용할 수 있다(또는 4.4절에서 보이듯이 Multi-View Stereo[MVS] 결과를 초기값으로 써도 동작함). NeRF는 임의의 3D 공간 좌표와 2D 시점(viewing direction)을 입력받아, 해당 위치의 볼륨 밀도(volume density)와 그 위치에서 해당 시점 방향으로 방출되는 색상 정보를 출력하는 신경 복사장(neural radiance field) MLP를 최적화한다.

NeRFactor는 학습된 NeRF의 기하 정보를 **“추출(distilling)”**하여 연속적인 표면(solid surface) 표현을 만들고, 이를 NeRFactor 고유의 기하 초기화로 활용한다. 구체적으로, 최적화된 NeRF를 이용해

1. 각 카메라 광선(ray)을 따라 예상되는 표면 위치,
2. 물체 표면상의 각 지점에서의 표면 노멀,
3. 임의의 방향에서 그 지점에 도달하는 조명에 대한 가시성을 구한다. 이 절에서는 NeRF로부터 이런 정보를 어떻게 도출하고(그림 3), 이를 다시 MLP 형태로 재파라미터화(re-parameterize)하여, 전체 재렌더링 오차(재구성 손실)를 줄이기 위해 이후 미세 조정(fine-tuning)하는 과정을 설명한다.

### 표면점(Surface points)

![](/assets/images/posts/494/img_6.png)

우리는 최적화된 NeRF로부터 추출한 이 **표면**을 최종 기하로 고정하고, 기존 NeRF처럼 볼륨 전체를 유지하지 않는다. 이로써 학습 및 추론 단계에서의 재조명(relighting) 연산 효율이 크게 향상된다. 즉, 카메라 광선마다 볼륨 전체가 아닌, “종료 지점” 하나만 고려해 광선이 방출하는 복사(radiance)를 계산하면 되기 때문이다.

### 표면 노멀(Surface normals)

![](/assets/images/posts/494/img_7.png)

![](/assets/images/posts/494/img_8.png)

**조명 가시성(Light visibility)**

![](/assets/images/posts/494/img_9.png)

![](/assets/images/posts/494/img_10.png)

실제로 본 모델을 풀 최적화하기 전에, 가시성 MLP와 노멀 MLP를 각각 독립적으로 미리 학습(pretrain)하여 NeRF의 σ-볼륨에서 계산된 가시성·노멀 값을 먼저 “그대로” 재현하도록 한다(이때는 평활화 정규화나 재렌더링 손실을 걸지 않음). 이를 통해 가시성 맵의 초깃값이 적절히 설정돼, 알베도나 양방향 반사 분포 함수(BRDF) MLP가 그림자를 “반사 특성의 일부”로 오판하여 (일종의 페인팅된 것처럼) 그림자를 설명해버리는 문제를 방지한다(“w/o geom. pretrain.”을 참조하라, 표 1 및 그림 S2).

**3.2 반사 특성(Reflectance)**

![](/assets/images/posts/494/img_11.png)

기존 신경 렌더링 연구에서는 NeRF 유사 환경에서 마이크로패싯(microfacet) 같은 **분석적(analytic) BRDF**를 활용하는 방법 [Bietal.2020;Srinivasanetal.2021]이 탐색되어 왔다. 본 논문에서도 5.1절에서 이 “분석적 BRDF” 버전의 NeRFactor를 실험한다. 분석적 모델은 최적화에 유효한 BRDF 파라미터화를 제공하지만, **파라미터 자체**에 대한 사전 지식(prior)을 부여하지 않는다. 즉, 마이크로패싯으로 표현 가능한 모든 재질이 **동등한 사전 확률**을 가진다. 또한 명시적(analytic) 모델을 쓰면 표현할 수 있는 재질의 폭이 제한되므로, 실세계의 모든 BRDF를 포괄하기에는 부족할 수 있다.

우리는 분석적 BRDF를 가정하는 대신, **실제 측정된 다양한 BRDF를 복원**하도록 **사전학습된 학습형(learned) 반사 함수**를 이용한다. 이렇게 함으로써 실세계 BRDF를 반영하는 **데이터 기반 사전(prior)**을 학습하여, 최적화가 **그럴듯한 반사 함수를** 찾도록 유도한다. 이는 본 문제에서 매우 중요한데, 왜냐하면 우리가 이용할 관측 데이터는 하나의 (미지) 조명 환경에서만 촬영된 이미지들이므로, 문제 설정이 극도로 정보가 부족(ill-posed)하기 때문이다.

### 알베도(Albedo)

![](/assets/images/posts/494/img_12.png)

### 실세계 BRDF로부터 사전(priors) 학습하기

스펙큘러 BRDF의 경우, 우리는 **실세계 BRDF**의 잠재 공간(latent space)과, 해당 잠재 벡터를 완전한 4차원 BRDF로 변환해주는 디코더(decoder)를 공동으로 학습하고자 한다. 이를 위해 Generative Latent Optimization(GLO) [Bojanowskietal.2018] 방식을 채택했는데, 이는 이미 Park 등 2019, Martin-Brualla 등 2021 등 다른 좌표 기반 모델에서도 사용된 기법이다.

![](/assets/images/posts/494/img_13.png)

![](/assets/images/posts/494/img_14.png)

![](/assets/images/posts/494/img_15.png)

† 옮긴이 주: MERL BRDF가 원래 RGB 형태로 되어 있으나, 여기서는 색 정보를 제거(무채색화)하여 스펙큘러 성분을 모델링한다는 의미.
---

![](/assets/images/posts/494/img_16.png)
---

**3.3 조명(Lighting)**  
우리는 조명을 위도-경도(latitude-longitude) 형식의 HDR 라이트 프로브(light probe) 이미지 Debevec1998로 **간단하고 직접적으로** 표현한다. 구면 조화(spherical harmonics)나 구면 가우시안(spherical Gaussians) 혼합 방식을 사용하는 것과 달리, 이 표현은 **고주파(high-frequency) 조명을 세밀히 나타낼 수 있어**, 명확한 경계의 그림자(hard cast shadows)를 지원한다. 하지만 이 방식을 쓰면, 수많은 파라미터가 생기며, 각 픽셀(파라미터)이 서로 독립적으로 값을 가질 수 있다는 문제가 있다. 우리는 이를 **조명 가시성(visibility) MLP**로 어느 정도 보완한다. 이 MLP를 통해 **표면상의 한 지점이 라이트 프로브의 모든 픽셀에 대해 어떤 가시성을 갖는지**를 빠르게 평가할 수 있기 때문이다.

실험적으로는 16×32 해상도의 조명 환경을 사용한다. 이보다 높은 해상도에서는 우리가 복원해야 할 추가 고주파 성분이 없을 것으로 예상하기 때문이다(조명은 물체의 BRDF 특성에 의해 사실상 저역 통과(low-pass) 필터링되고 Ramamoorthi and Hanrahan 2001, 우리가 다루는 물체는 극도로 반사율이 높거나 거울처럼 반사하는 물체가 아니다).

![](/assets/images/posts/494/img_17.png)

**3.4 렌더링(Rendering)**

![](/assets/images/posts/494/img_18.png)

![](/assets/images/posts/494/img_19.png)

![](/assets/images/posts/494/img_20.png)

## 4 결과 & 응용(Results & Applications)

이 절에서는 다음과 같은 내용을 보인다.

1. **NeRFactor로 얻은 고품질 기하** (섹션 4.1)
2. **형상·반사 특성·조명**을 공동 추정하는 NeRFactor의 능력 (섹션 4.2)
3. 이를 활용한 **자유 시점 재조명** 예시 (하나의 점광원 또는 임의 라이트 프로브 활용, 그림 5 및 그림 6) (섹션 4.3)
4. 초기 기하로 **NeRF 대신 Multi-View Stereo(MVS)를 사용**했을 때의 NeRFactor 성능 (섹션 4.4)
5. **재질 편집(material editing)** 사례 (그림 8) (섹션 4.5)

본 논문에서 사용된 다양한 유형의 데이터(합성·실측 등)의 생성, 촬영, 수집 방식에 대해서는 부록(B절)을 참고하라.

**4.1 형상 최적화(Shape Optimization)**  
NeRFactor는 물체의 표면점(surface points)과 그에 대응하는 표면 노멀(surface normals), 그리고 각 광원 위치에 대한 가시성을 동시에 추정한다. 그림 3은 이러한 기하 정보를 시각화한 것이다.  
가시성(visibility)을 시각화하기 위해, 16×32 해상도의 라이트 프로브(light probe) 각 픽셀(총 512개)에 대응하는 가시성 맵들을 픽셀 단위로 평균 낸 뒤(즉, 모든 광원 픽셀에 대한 가시성의 평균), 그 평균 맵(앰비언트 오클루전 형태)을 회색조(grayscale) 이미지로 표현했다. 광원별 가시성 맵(그림자 맵) 영상은 보충 자료(부록 영상)에서 확인할 수 있다. 그림 3을 보면, 재렌더링 오차를 최소화하면서 공간적 평활성(spatial smoothness)을 함께 유도하는 공동 추정 덕분에, 우리의 표면 노멀과 가시성은 부드럽고 실제 장면과 유사하게 복원됨을 알 수 있다.

만약 공간적 평활성 제약을 제거하고(즉, 재렌더링 손실만으로) 학습한다면, 잡음이 심한 형상이 얻어져 렌더링에 부적합하다. 이런 기하 기반 아티팩트(artifacts)는 저주파(low-frequency) 조명 아래에서는 눈에 띄지 않을 수 있지만, 단일 점광원처럼 조명 대역이 좁은(One-Light-at-A-Time [OLAT]) 환경에서는 명확하게 드러난다(부록 영상 참조).

흥미롭게도, 평활성 제약을 꺼도(re-rendering loss만으로 최적화해도), NeRFactor가 추정한 기하는 여전히 원래 NeRF 기하보다 훨씬 매끄럽다(그림 3의 [A]와 [B] 비교, 표 1의 [I] 참조). 이것은 재렌더링 손실만으로도 어느 정도 기하를 부드럽게 만드는 효과가 있기 때문이다. 자세한 내용은 5.1절을 참고하라.

![](/assets/images/posts/494/img_21.png)

**그림 3. NeRFactor가 복원한 고품질 기하**  
**(A)** 학습된 NeRF로부터 바로 표면 노멀과 조명 가시성을 유도할 수 있지만, 이렇게 얻은 기하는 재조명(relighting)에 사용하기엔 잡음이 너무 심하다(부록 영상 참조).  
**(B)** 형상과 반사 특성을 공동 최적화하면 NeRF 기하가 개선되긴 하나, 여전히 잡음이 상당히 남아 있다(예: II 부분의 줄무늬 스트라이프 아티팩트).  
**(C)** 평활성(smoothness) 제약을 추가한 공동 최적화를 통해, 실제 장면과 유사하고 매끄러운 표면 노멀과 조명 가시성을 얻을 수 있다. 모든 입사광 방향에 대해 가시성을 평균 낸 결과가 앰비언트 오클루전이다.

**4.2 형상·반사 특성·조명의 공동 추정**  
본 절에서는 **복잡한 기하나 반사 특성**을 지닌 장면에 대해, NeRFactor가 물체의 외관을 형상, 반사 특성, 조명으로 어떻게 분해(factorize)하는지 살펴본다.

**알베도 시각화**  
알베도를 시각화할 때는 내부 이미지(intrinsic image) 분야에서 통용되는 관례를 따른다. 즉, **알베도와 음영(shading)의 절대적 밝기는 복원 불가능**하다고 가정 [LandandMcCann1971]. 또한 **색상 항등(color constancy) 문제**, 즉 조명 평균 색상과 물체 알베도 평균 색상을 어떻게 구분할 것인지 [Buchsbaum1980]는 범위 밖으로 두었다.

이 두 가정을 바탕으로, 우리는 예측된 알베도를 시각화하거나 정확도를 측정할 때, 각 RGB 채널을 동일한 스칼라(전역 스케일)로 조정(스케일링)한다. 이 스칼라는 **최대한으로 참 알베도와의 평균제곱오차(MSE)를 줄이도록** 정한다 [BarronandMalik2014]. 별도의 언급이 없다면, 합성 장면에서의 모든 알베도 예측에 이 방식의 보정을 적용하고, 그림에서 제대로 표시하기 위해 감마 보정(γ=2.2)도 진행한다.

반면, **추정된 라이트 프로브**(조명)는 이와 같은 스케일링을 적용하지 않는다(조명 추정 자체는 본 연구의 주목적이 아니기 때문). 단지, 전체 RGB 채널에서 최댓값을 1로 정규화한 후 감마 보정(γ=2.2)만 적용해 시각화한다.

**결과 분석**  
그림 4 (B)에 따르면, NeRFactor가 예측한 표면 노멀은 대체로 **매우 고품질**이며 매끄럽고, 실제 노멀과 유사하다. 단, 핫도그 번(빵)처럼 초고주파(high-frequency) 세부가 있는 영역에서는 약간의 오차가 생긴다. **drums(드럼)** 장면에서는, 심벌(cymbal) 중앙의 나사(screw)나 드럼 측면의 금속 테두리 같은 미세한 디테일까지 잘 복원한 모습을 볼 수 있다. **ficus(화분)** 장면에서도 복잡한 잎사귀의 기하가 잘 회복된다. 앰비언트 오클루전 맵 역시, 각 지점이 광원에 노출되는 평균 정도를 제대로 표현한다.

알베도는 그림자나 음영 정보가 잘못 들어가는 일 없이 **깨끗하게** 복원된다. 예를 들어, 드럼 표면의 음영이 알베도 예측 결과에는 거의 나타나지 않는다. 게다가 예측된 라이트 프로브는 메인 광원 위치와 푸른 하늘(그림 4 (I)에서 파란색 픽셀)을 올바르게 반영한다. 세 장면 모두에서, 예측된 BRDF는 공간적으로 변화(spatially-varying)하며, (E)에서 보이듯 지점마다 서로 다른 BRDF 잠재 벡터를 할당받아, 실제로 각기 다른 소재가 존재함을 잘 포착한다.

우리는 구면 조화(spherical harmonics)처럼 더 정교한 표현 대신, **위도-경도(latitude-longitude) HDR 맵**을 사용해 조명을 나타낸다. 이는 광원을 간단명료하게 표현하며, 중등도의 확산성(diffuse)이 있는 BRDF를 거칠 때 조명이 실제로 저역 통과(low-pass) 필터링되기 때문에, 16×321보다 높은 해상도를 복원할 필요가 없을 것으로 예상했기 때문이다 [RamamoorthiandHanrahan2001]. 그림 4 (I)를 보면, NeRFactor가 먼 왼쪽에 위치한 밝은 광원과 푸른 하늘을 잘 포착한 라이트 프로브를 추정했음을 알 수 있다. 비슷하게 그림 4 (II)에서도, 가장 강한 광원 위치(왼쪽의 밝은 영역)가 정확히 복원되었다.

† 주: 실제 장면에서는 참 알베도를 알 수 없으므로 이러한 보정은 불가능하다.

![](/assets/images/posts/494/img_22.png)

**그림 4. 형상, 반사 특성, 조명의 공동 최적화**  
우리의 방법으로 복원된 표면 노멀, 가시성, 알베도는 때때로 세밀한 디테일이 일부 누락되긴 하지만, 전반적으로 실제 장면과 상당히 유사하다. NeRFactor가 복원한 조명은 (우리 장면 물체들이 거울처럼 반짝이지 않기 때문에) 과도하게 부드럽게 표현되고, 실제로 관측 가능한 상반구(上半球) 외 영역에서는 부정확할 수 있다. 그럼에도, 주요 광원과 물체 주변의 가리는 요소(occluders)는 라이트 프로브에서 실제 위치와 유사한 곳에 잘 포착된다. 참고로 실제 장면의 BRDF는 Blender 셰이더 노드 트리로 정의되어 있어, 우리 모델이 학습 기반으로 표현하는 BRDF와 직접 비교하기는 어렵다.

![](/assets/images/posts/494/img_23.png)

**그림 5. 자유 시점 재조명(Free-viewpoint relighting)**  
NeRFactor가 만들어 낸 분해 결과를 이용해, 물체를 임의의 조명 환경—including OLAT(One-Light-at-a-Time) 같은 까다로운 조건—에서 새로운 시점으로 렌더링할 수 있다. NeRFactor의 렌더링은 정성적으로 실제 장면과 유사하며, 스펙큘러 반사나 그림자(강한 그림자·부드러운 그림자 모두) 같은 까다로운 효과도 정확하게 나타낸다.

![](/assets/images/posts/494/img_24.png)

**그림 6. 실제 물체 촬영 결과**  
**(I)** 미지 조명 아래에서 촬영한 실물 객체 이미지(A)가 주어졌을 때, NeRFactor는 이를 알베도(C), 공간적으로 변화하는 BRDF 잠재 벡터(D), 표면 노멀(E), 그리고 모든 입사광 방향에 대한 조명 가시성(그림에서는 앰비언트 오클루전으로 시각화; F)으로 분해한다. 추정된 꽃(Flower) 알베도는 음영(shading)이 제거된 형태임에 주목하자.  
**(II)** 이 분해 결과를 바탕으로, 어떤 조명 환경이든 원하는 대로 장면을 재조명하여 새로운 시점을 합성할 수 있다. 까다로운 실제 장면에서도, NeRFactor는 다양한 조명 조건에서 사실적인 스펙큘러와 그림자를 합성해 낸다.

**4.3 자유 시점 재조명(Free-Viewpoint Relighting)**  
NeRFactor는 형상과 반사 특성을 각각 3D 필드로 추정하므로, 시점 합성(view synthesis)과 재조명을 동시에 수행할 수 있다. 따라서 이 논문과 보충 영상에 제시된 모든 재조명 결과는 새로운 시점에서 렌더링한 것이다. NeRFactor의 한계를 확인하기 위해, 우리는 **OLAT(One-Light-at-a-Time)** 조명 상태—즉, 주변광(ambient light) 없이 한 번에 하나의 점광원만 켠 상태—처럼 강한(혹독한) 테스트 조명 조건을 사용한다. 이때 단단한 경계의 그림자(hard cast shadows)가 생기는데, 이는 부정확한 기하나 재질에 의한 렌더링 아티팩트를 뚜렷이 드러낸다.

시각화를 위해, 우리는 각 시점에서 NeRF가 예측한 외부 파라미터(extrinsics)를 사용하여 재조명된 결과물을 합성한다. 그런 뒤, 기본 vanilla NeRF를 학습해 초기 형상(shape)을 얻고, 이를 NeRFactor로 옮겨와(refine) 반사 특성·조명과 함께 공동 최적화한다. 그림 6 (I)에 나타나듯이, 물체의 외관은 조명과 표면 노멀, 조명 가시성, 알베도, 공간적으로 변화하는 BRDF 잠재 벡터 같은 3D 필드로 분해되어 관측된 이미지를 설명한다. 이렇게 분해된 정보를 이용해, 우리는 추정된 조명을 새로운 라이트 프로브로 교체하여 장면을 재조명할 수 있다(그림 6 [II]). 우리 분해는 완전한 3D 형태로 이루어지므로, 중간에 생성되는 모든 버퍼(geometry, normals, visibility 등)를 어떤 시점에서든 렌더링할 수 있으며, 본문에 제시된 재조명 결과 역시 새로운 시점에서의 것이다. 참고로, 실제 물체 장면에서는 먼 거리의 기하가 특정 방향에서 들어오는 빛을 막아 그림자를 드리우는 일이 없도록, 3D 박스 안에 물체를 가두어(bound) 놓았다.

**4.4 Multi-View Stereo를 활용한 형상 초기화**  
앞서 우리는 NeRFactor가 **NeRF로부터 추출한 기하**를 초기값으로 사용하고, 이를 반사 특성·조명과 함께 공동 최적화해 나가는 과정을 보였다. 이번에는 NeRF 대신 **MVS(Multi-View Stereo)**로 얻은 형상을 초기값으로 써도 NeRFactor가 동작하는지 확인한다. 구체적으로, 우리는 장면마다 약 50장의 다중 뷰 이미지(및 대응하는 카메라 자세)를 제공하는 **DTU-MVS** 데이터셋 [Jensenetal.2014;Aanæsetal.2016]을 사용한다. 여기서 우리는 Furukawa & Ponce 2009가 구현한 MVS 결과를 **Poisson 재구성** [Kazhdanetal.2006]으로 메시(mesh)화한 뒤, 이를 NeRFactor의 초기 형상으로 삼는다. 이 데이터에 대한 자세한 내용은 부록 B절을 참조하라. 이 실험은 다른 초기 형상으로도 NeRFactor가 작동 가능한지를 살피는 동시에, **또 다른 실제 이미지 소스**를 대상으로 NeRFactor를 평가해보려는 목적도 있다.

NeRFactor는 NeRF가 아닌 MVS 기하를 초기값으로 쓸 때에도 **높은 품질의 형상 추정**을 달성한다. 그림 7 (A, B)에서 볼 수 있듯이, NeRFactor가 추정한 표면 노멀과 조명 가시성은 기존 MVS가 갖고 있던 잡음을 제거했을 뿐 아니라, 충분한 기하 디테일도 유지한다. 이러한 고품질 형상을 발판으로, NeRFactor는 입력 이미지와 매우 흡사한 사실적인 시점 합성(view synthesis) 결과를 얻는다(그림 7 [C]). 특히 **scan110** 장면은 재질이 번쩍이는(반사성이 있는) 물체라서, 더 고주파(high-frequency) 조명 조건을 복원할 수 있었음을 [C]의 두 조명 상태 비교를 통해 알 수 있다. 이후 우리는 이 장면들을 새로운 시점에서, 또 다른 두 개의 라이트 프로브로 재조명했으며(그림 7 [D, E]), 이때 눈에 띄는 스펙큘러 하이라이트와 그림자를 NeRFactor가 사실적으로 합성함을 볼 수 있다(이는 가시성 모델링 덕분이다). 한편, NeRFactor는 **scan110**에 대해 “하얀색 알베도 + 금색 조명”으로 설명해버리지만(4.2절에서 다룬 근본적 모호성 탓), 그럼에도 불구하고 이 **그럴듯한 분해**를 통해 장면을 현실감 있게 재조명할 수 있음을 확인했다.

**4.5 재질 편집(Material Editing)**

![](/assets/images/posts/494/img_25.png)

그림 8 (왼쪽)을 보면, NeRFactor가 분해한 원본 재질을, 까다로운 두 가지 OLAT(One-Light-at-a-Time) 조명 조건에서 사실적으로 재조명한 결과를 볼 수 있다. 나아가, 동일한 OLAT 테스트 조명 조건에서, 이렇게 편집된 재질 역시 **사실적인 스펙큘러 하이라이트와 단단한 그림자**를 동반하여 재조명되었다(그림 8 오른쪽).

![](/assets/images/posts/494/img_26.png)

**그림 8. 재질 편집과 재조명**  
NeRFactor로 분해된 결과물 가운데, (왼쪽) 원본 재질을 두 가지 OLAT 조명 조건에서 재조명한 결과와, (오른쪽) **같은 OLAT** 조건에서 편집된 재질을 재조명한 결과를 비교해 놓았다. 알베도와 반사 특성을 어떻게 수정했는지는 본문에서 설명.

## 5 평가 연구(Evaluation Studies)

본 절에서는 각 모델 구성 요소가 갖는 중요성을 밝히기 위한 **어블레이션(ablation) 실험**과, 외관 분해 및 재조명 과업에서 NeRFactor를 기존 고전 방식 및 딥러닝 기반 최신 기법과 비교한 결과를 제시한다. 정량적 평가 지표로는 PSNR(Peak Signal-to-Noise Ratio), SSIM(Structural Similarity Index Measure) Wangetal.2004, LPIPS(Learned Perceptual Image Patch Similarity) Zhangetal.2018를 사용한다.  
또한, 동일 물체를 여러 입력 조명 상태에서 각각 학습했을 때, 알베도 추정 결과가 일관성을 유지하는지에 대해서는 부록 C.1절에서 논의한다.

**5.1 어블레이션 연구(Ablation Studies)**  
이 절에서는 NeRFactor를 여러 합리적인 대안적 설계와 비교하여, 주요 구성 요소를 하나씩 제거(어블레이션)했을 때 성능이 정량적으로 얼마나 떨어지는지 살펴본다. 정량적 어블레이션 실험 결과는 표 1에 제시했으며, 정성적 실험 결과는 부록 C.2절과 보충 영상을 참고하라.

### 학습형(learned) BRDF vs. 분석적(analytic) BRDF

![](/assets/images/posts/494/img_27.png)

### 기하(geometry) 사전학습 유무

![](/assets/images/posts/494/img_28.png)

### 평활성 제약 유무

3장에서 언급했듯이, 우리는 MLP 기반 모델이 정보가 부족한(underconstrained) 상황에서 원활히 동작하도록 **간단하지만 효과적인** 공간적 평활성 제약을 도입했다. 평활성 제약을 제거해도, 표 1에서 보이듯이 뷰 합성(view synthesis) 측면에서는 여전히 괜찮은 성능을 낸다(NeRF 역시 평활성 없이도 고품질 뷰 합성이 가능하기 때문). 그러나 알베도 추정이나 재조명 같은 다른 태스크에서는 이 변형 모델의 성능이 크게 떨어진다. 정성적으로도, 이 변형은 재조명에 쓰기엔 노이즈가 심한 추정을 내놓는다(Figure S2 [B], 보충 영상 참조).

### 형상을 최적화 vs. NeRF 형상만 사용

노멀과 가시성 MLP를 **아예 제거(ablate)**하면, 이 변형 모델은 결국 NeRF가 추정한 노멀과 가시성을 그대로 사용하게 된다(“NeRF의 형상만 사용”). 표 1과 보충 영상을 보면, 이 경우 추정된 반사 특성 자체는 전체 모델에서 유도되는 평활성 사전에 의해 부드럽게 유지되지만, **NeRF 기하의 잡음**이 최종 렌더링에서 아티팩트로 드러나는 것을 확인할 수 있다.

![](/assets/images/posts/494/img_29.png)

다음은 표 1에 관한 해설입니다:

- **평가 방식**: 총 4개의 합성 장면(핫도그, 화분(ficus), 레고, 드럼)에 대해, 8개 시점을 균등하게 샘플링하여 얻은 결과 지표의 **산술평균**을 표에 제시하고 있습니다.
- **성능 순위 시각화**: 각 지표별로 성능이 상위 3위 안에 드는 기법을 빨간색, 주황색, 노란색으로 강조해 보여줍니다.
- **태스크 IV와 V**: 재조명(relighting)을 평가할 때, 새롭게 16가지 조명 조건을 사용했습니다. 이는 8개의 OLAT(One-Light-at-a-Time) 조건과 블렌더(Blender)에 기본 포함된 8개의 라이트 프로브를 합한 것입니다.
- **SIRFS 지표 제한**:
  - SIRFS 기법은 비직교(non-orthographic) 카메라나 ‘월드 스페이스’ 기하(geometry)를 지원하지 않기 때문에, **노멀(normal), 뷰 합성(view synthesis), 재조명(relighting)** 관련 지표를 표에 제시할 수 없었습니다.
  - 부록 그림 S3에 따르면, SIRFS가 복원한 기하가 부정확함을 확인할 수 있습니다.
- **추가 참고**: 본문 5.1절에서 각 기법의 구성 요소와 결과에 대한 더 자세한 논의를 볼 수 있고, 정성적(qualitative) 어블레이션 연구는 부록 C.2절을 참고하라고 안내합니다.

즉, 표 1은 여러 기법들을 다양한 과업(외관 분해, 재조명 등)과 여러 장면·조명·시점 조건에서 종합적으로 비교·평가한 결과를 정리한 것이며, SIRFS는 카메라·기하 제약으로 일부 항목이 측정 불가능했다는 점을 보여줍니다.

![](/assets/images/posts/494/img_30.png)

![](/assets/images/posts/494/img_31.png)

**그림 9. Oxholm and Nishino 201420142014와의 비교**  
섹션 5.2를 참조하라. †이 방법은 섹션 5.2에서 설명한 대로 우리 프레임워크 안에서 크게 개선되었으며, 조명을 직접 추정하지 않기 때문에 **정답 조명(ground-truth illumination)을 입력**으로 받는다.

**5.2 기준 모델(Baseline)과의 비교**  
이 절에서는 NeRFactor를 **고전적(classic) 접근**과 **딥러닝 기반 최신 기법**(Oxholm and Nishino 2014, Philip et al. 2019)과 비교하며, 외관 분해와 자유 시점 재조명 과업에서의 성능을 평가한다.  
(단일 뷰 기반 고전적 접근인 **SIRFS** BarronandMalik2014와의 비교는 부록 C.3절을 참고.)

### Oxholm and Nishino 2014

우리는 Oxholm과 Nishino가 제안한, **알려진 조명 하**에서 물체의 형상과 공간적으로 균일한(non-spatially-varying) BRDF를 추정하는 다중 뷰 기법 OxholmandNishino2014의 **개선판**과 NeRFactor를 비교한다. 원본 코드가 공개되어 있지 않아, 우리 프레임워크 안에서 저자들의 핵심 아이디어(형상 평활화 정규화와 데이터 기반 BRDF 사전)를 재현한 뒤, 다음과 같은 개선을 적용했다.

- 기하 초기화를 실루엣(silhouette) 기반의 볼륨 헐(hull) 대신 **NeRF 형상**으로 변경
- **공간적으로 변화하는 알베도**를 모델링(원논문은 공간적으로 균일한 BRDF만 고려)
- 형상은 메시(mesh)가 아니라 표면 노멀 MLP로 표현
- BRDF 예측은 MERL 기반 Nishino2009;NishinoandLombardi2011;LombardiandNishino2012이 아닌 **사전학습된 BRDF MLP**로 구현

또한 이 기준 모델은 **정답 조명(ground-truth lighting)을 입력**으로 받는다. 반면 NeRFactor는 형상·반사 특성과 함께 조명도 추정해야 하므로, Oxholm & Nishino에 비해 불리한 조건이다.

그림 9 (I)를 보면, 이 개선판조차도 가시성(visibility)을 모델링하지 못해(예: hotdog, lego) 그림자를 제대로 제거하지 못한다. 이는 알베도 추정에서 그림자 잔재가 남는 원인이 된다. 그 결과, 그림 9 (II)에 제시된 재조명 결과에서도, hotdog 접시가 붉게 물드는 등 잔상(artifact)이 발생한다. 게다가 이 기준 모델은 **공간적으로 변화하는 BRDF**를 추정하지 못하므로, hot dog 빵이나 ficus 잎사귀 같은 부위조차도 접시·화병처럼 스펙큘러가 강한 재질로 잘못 추정된다. 마지막으로, 그림자 같은 **비국소(non-local) 빛 전송 효과**를 합성하지 못해, OLAT 환경에서 하드 섀도(hard cast shadow)를 사실적으로 나타내는 데 실패한다. 반면 NeRFactor는 실제감 있는 그림자를 올바르게 표현한다.

### Philip et al. 2019

Philip 등 2019은 **대규모 장면**을 재조명하면서 사실적인 그림자를 합성하는 기법을 제안했다. 입력 조건(미지 조명 아래 다중 뷰)은 우리와 유사하지만, 오직 태양광 같은 **단일 주요 광원**만을 재현하는 데 초점을 맞추므로, NeRFactor처럼 **임의의 광원**(또는 다른 라이트 프로브)으로 재조명하는 것은 지원하지 않는다. 따라서 두 기법을 **점광원 재조명**(OLAT)이라는 특정 과업에서만 비교한다.

이들의 결과(그림 10 [A])에 있는 “노란 안개(yellow fog)”는 기하 재구성이 부정확한 탓으로 보인다. 또한 저자들이 **옥외 야외 장면**(outdoor scene) 위주로 학습했기에, 배경이 있는 이미지를 다루는 데 한계가 있는 듯하다. 우리는 보다 관대한 비교를 위해, 참 물체 마스크(ground-truth object masks)를 사용해 노란 안개 부분을 제외한 영역만으로 지표를 계산한 “Philip et al. 2019 + Masks”도 함께 평가했다. 하지만 그림 10 표를 보면, PSNR과 SSIM에서 NeRFactor가 “Philip et al. 2019 + Masks”를 모두 능가한다.

LPIPS의 경우, Philip et al. 기법이 **입력 이미지를 재투영(reproject)**하여 IBR(Image-Based Rendering) 방식으로 새 시점 영상을 만들기 때문에, **고주파(high-frequency) 디테일**이 그대로 살아 있어 LPIPS 점수가 더 낮다(더 좋은 점수). 반면 NeRFactor는 물리 기반(physically-based) 3D 재렌더링 접근법이라, 그림자를 더욱 사실적으로 표현한다. 예컨대 OLAT 1 시나리오에서 기준 모델의 그림자는 너무 부드럽고(흐릿하게 표현), OLAT 2에서는 그림자가 실제보다 좁은 영역에만 생긴다. 또한 NeRFactor는 태양광이 아닌 **“Studio”**처럼 광원이 네 개나 되는 라이트 프로브도 다룰 수 있다(그림 6 [D]).

![](/assets/images/posts/494/img_32.png)

**그림 10. Philip et al. 201920192019와의 점광원(OLAT) 재조명 비교**  
여기 수치는 8가지 OLAT 조건에 대한 평균이다. 섹션 5.2를 참조하라.

**6 한계(Limitations)**  
본 논문에서 제안한 NeRFactor가 여러 변형(modified variants) 및 기존 기법들과의 비교에서 우수한 성능을 보이긴 하지만, 여전히 해결해야 할 중요한 제약 사항들이 있다.

1. **조명 해상도의 제한**  
   조명 가시성 계산을 tractable(계산 가능한 범위)하게 만들기 위해, 라이트 프로브 이미지를 16×32 해상도로 제한했다. 이는 매우 강한(단단한) 그림자나 극도로 고주파의 BRDF를 복원하기에는 부족할 수 있다. 예컨대 부록 그림 S1 (Case D)처럼 태양 픽셀이 완전 HDR인 초고주파 조명 아래에서, 알베도 추정에 스펙큘러나 그림자 잔재가 남을 가능성이 있다(화병에 나타나는 잔영 등이 예시).
2. **단일 반사 직접 조명(single-bounce direct illumination) 가정**  
   빠른 렌더링을 위해 단일 반사 직접 조명만 고려하므로, 간접 조명(indirect illumination) 효과는 제대로 모델링하지 못한다.
3. **NeRF 혹은 MVS를 이용한 초기 기하(geometry) 의존**  
   NeRFactor는 기하 추정을 NeRF나 MVS 결과로 초기화한다. NeRF가 어느 정도 품질 이상을 보장하면, 본 모델이 해당 기하를 일정 수준까지는 교정할 수 있으나, 뷰 합성에 큰 영향을 주지 않는 방식으로 형상이 잘못 추정된 경우엔(즉, 노이즈가 심한 경우), NeRFactor도 실패할 수 있다. 실제 장면을 대상으로 했을 때, 입력 카메라로는 보이지 않는 먼 곳에 ‘떠 있는(floating)’ 잘못된 기하가 생기는 현상이 관측되었고, 이로 인해 물체 위에 부정확한 그림자를 드리우는 문제가 발생하기도 했다.

**7 결론(Conclusion)**  
본 논문에서는 **Neural Radiance Factorization (NeRFactor)**를 제안하였다. 이는 다중 뷰 이미지와 카메라 자세만으로 물체의 형상과 반사 특성을 복원하는 기법이다. 특히 대부분의 기존 연구들이 **다양한(또는 알려진) 조명 조건**에서 관측된 데이터를 요구하는 것과 달리, NeRFactor는 **하나의 미지 조명**으로만 촬영된 이미지에서도 형상, 반사 특성, 조명을 함께 추정할 수 있다. 이처럼 정보가 부족한 문제(ill-posed)를 해결하기 위해, NeRFactor는 간단하지만 효과적인 공간 평활화(smoothness) 제약(MLP 기반으로 구현)과 실측 데이터를 활용한 **데이터 기반 BRDF 사전(prior)**을 사용한다.

실험을 통해, NeRFactor가

- 재조명(relighting)과 시점 합성(view synthesis)에 충분한 고품질 형상을 복원하고,
- 신뢰도 높은 알베도와 공간적으로 변화하는 BRDF를 제공하며,
- 주요 광원의 존재를 제대로 반영하는 조명 추정 을 수행함을 확인했다. 이로써 NeRFactor가 분해한 정보(형상·조명·반사 특성)를 활용하면, 물체를 점광원(OLAT)이나 라이트 프로브 이미지를 포함한 임의의 광원 아래에서 렌더링하고, 임의의 시점으로 관찰하며, 심지어 물체의 알베도나 BRDF까지 편집할 수 있게 된다. 이는 **간단히 촬영한 사진으로부터 고차원적인 3D 그래픽 자산을 추출**하는 연구 흐름에서 의미 있는 진전을 이룬 것이다.

**감사의 말(Acknowledgments)**  
익명의 리뷰어들의 유익한 피드백에 감사드린다. 또한 데이터나 결과물 제공에 협조해 준 Julien Philip, Tiancheng Sun, Zhang Chen, Jonathan Dupuy, Wenzel Jakob, 그리고 의미 있는 토론에 참여해 준 Zhoutong Zhang, Xuaner(Cecilia) Zhang, Yun-Tai Tsai, Jiawen Chen, Tzu-Mao Li, Yonglong Tian, Noah Snavely에게 감사한다. 논문 개선에 큰 도움을 주신 Noa Glaser와 David Salesin에도 감사를 표한다. 본 연구는 MIT-공군 AI Accelerator의 지원을 일부 받았다. 합성 장면으로 사용된 모델 제공자 blendswap.com의 bryanajones(드럼), Herberhold(화분), erickfree(핫도그), Heinzelnisse(레고)에게도 감사드린다.
