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

[NeRFactor: neural factorization of shape and reflectance under an unknown illumination: ACM Transactions on Graphics: Vol 40, No](https://dl.acm.org/doi/10.1145/3478513.3480496)

![](/assets/images/posts/494/img.png)

**그림 1. Neural Radiance Factorization (NeRFactor).** 하나의 미지 조명 조건만 주어진 물체의 다중 뷰 이미지(및 해당 카메라 자세)가 (왼쪽) 주어졌을 때, NeRFactor는 장면의 외관을 표면 노멀, 조명 가시성, 알베도, 반사 특성과 같은 3D 신경 필드로 분해(가운데)할 수 있다. 이를 통해 자유 시점 재조명(그림자 처리 및 소재 편집 포함) 같은 다양한 응용(오른쪽)이 가능해진다.

우리는 하나의 알려지지 않은 조명 조건에서 촬영된 물체의 다중 뷰 이미지(및 해당 카메라 자세)를 바탕으로, 물체의 형태와 공간적으로 변화하는 반사 특성을 복원하는 문제를 다룬다. 이를 통해 임의의 환경 조명 아래에서 물체의 새로운 시점을 렌더링하고, 물체의 재질 특성을 편집할 수 있게 된다. 우리의 접근 방식인 Neural Radiance Factorization(NeRFactor)의 핵심은, [Mildenhall et al. 2020]에서 제안된 Neural Radiance Field(NeRF)의 부피(Volumetric) 형상 표현을 표면 표현으로 추출한 뒤, 형상을 정밀하게 개선하는 동시에 공간적으로 변화하는 반사 특성과 환경 조명을 함께 추정하는 것이다.

구체적으로 NeRFactor는, 재렌더링 손실과 간단한 평활화 사전(prior), 그리고 실제 BRDF 측정에서 학습된 데이터 기반 BRDF 사전만을 활용하여, 어떠한 별도 지도 없이도 표면 노멀, 조명 가시성, 알베도, 양방향 반사 분포 함수(BRDF)의 3D 신경 필드를 복원한다. 특히 조명 가시성을 명시적으로 모델링함으로써, 알베도에서 그림자를 분리하고, 임의의 조명 조건에서 사실적인 부드러운 혹은 강한 그림자를 합성해 낼 수 있다. 이러한 복잡하고 제약이 많은 캡처 환경에서도, NeRFactor는 합성 및 실제 장면 모두에 대해 자유 시점 재조명이 가능한 설득력 있는 3D 모델을 복원하는 데 성공한다. 정성적·정량적 실험 결과에 따르면, NeRFactor는 여러 과업에서 기존의 고전적 접근 방식 및 딥러닝 기반 최신 기법보다 우수한 성능을 보인다. 관련 비디오, 코드, 데이터는 아래 웹사이트에서 확인할 수 있다.  
[**people.csail.mit.edu/xiuming/projects/nerfactor/**](http://people.csail.mit.edu/xiuming/projects/nerfactor/)

[https://people.csail.mit.edu/xiuming/projects/nerfactor/](http://people.csail.mit.edu/xiuming/projects/nerfactor/)

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
