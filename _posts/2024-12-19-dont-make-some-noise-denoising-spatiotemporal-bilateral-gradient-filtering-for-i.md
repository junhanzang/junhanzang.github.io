---
title: "[(Don't) Make Some Noise: Denoising] Spatiotemporal Bilateral Gradient Filtering for Inverse Rendering"
date: 2024-12-19 22:38:20
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://weschang.com/publications/stadam/>

[Spatiotemporal Bilateral Gradient Filtering for Inverse Rendering](https://weschang.com/publications/stadam/)

![](/assets/images/posts/442/img.jpg)

### **Spatiotemporal Bilateral Gradient Filtering for Inverse Rendering**

#### 1. 연구 배경

역 렌더링은 텍스처, 볼륨, 기하학적 데이터를 복구하는 데 사용되며, 이를 위해 많은 경우 Adam 옵티마이저가 활용됩니다. 하지만, **Adam**은 템포럴 필터링만 지원하며, 스페이셜 코히어런스(spatial coherence)를 적절히 활용하지 못합니다. 이로 인해 높은 주파수 노이즈 또는 엣지 손실이 발생할 수 있습니다.

#### 2. 핵심 기여

- **스페이셜 및 템포럴 필터링 결합**: Adam을 확장하여 스페이셜 필터링과 템포럴 필터링을 결합, 더 나은 수렴을 보장.
- **교차 양방향 필터**: 기존의 라플라시안 스무딩이 가진 과도한 스무딩 문제를 해결, 엣지 주변에서도 높은 품질을 유지.
- **빠른 수렴**: 적은 샘플로도 고품질 복구 가능, 특히 텍스처와 볼륨 복구에서 효과적.

#### 3. 연구의 주요 결과

1. **텍스처 복구**:
   - Adam과 비교해 엣지 디테일을 잘 유지하며, 노이즈를 효과적으로 제거.
2. **볼륨 복구**:
   - 높은 샘플 카운트에서도 기존 방법보다 빠르고 안정적인 수렴.

#### 4. 개인적인 인사이트

이 논문은 기존의 최적화 방식의 한계를 명확히 짚어내고, 이를 보완하는 효과적인 방법을 제안했다는 점에서 의미가 큽니다. 특히, **엣지 주변에서의 디테일 유지**와 같은 특징은 실질적인 응용 가능성을 크게 확장할 수 있습니다. 제 분야에서도 이 기법을 일부 테스트해볼 가치가 있다고 생각합니다.

#### 5. 시사점

이 논문은 최적화, 특히 역 렌더링 최적화에서 엣지 유지와 빠른 수렴을 중시하는 연구자들에게 큰 영감을 줄 것입니다.
