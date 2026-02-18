---
title: "[Look at it Differently: Novel View Synthesis] Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos"
date: 2024-12-19 01:16:28
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

### **<https://arxiv.org/abs/2406.18717>**

[Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos](https://arxiv.org/abs/2406.18717)

### **Dynamic Gaussian Marbles: 모노큘러 비디오를 통한 새로운 시점 합성**

\*\*"Dynamic Gaussian Marbles"\*\*는 단일 카메라로 촬영한 **모노큘러 동영상**을 활용하여 고품질의 3D 장면을 재구성하고, 새로운 시점을 생성하는 혁신적인 접근 방식을 제안합니다. 특히, **4D Gaussian Splatting** 기법을 기반으로 하며, 기존 방법보다 효율적이고 정밀한 결과를 제공합니다.

![](/assets/images/posts/423/img.jpg)

---

### **핵심 내용 요약**

1. **4D Gaussian 기반 장면 표현**
   - **Gaussian Marble**: 각 Gaussian을 구형(isotropic)으로 제한하여 과적합(overfitting)을 방지하고 학습 안정성을 높였습니다.
   - Gaussian은 이미지, 깊이(depth), 세그멘테이션(segmentation)을 결합해 표현되며, 이를 바탕으로 로스를 계산합니다.
2. **Divide-and-Conquer 학습 전략**
   - 긴 동영상을 **짧은 구간으로 나누어** 독립적으로 최적화한 뒤, 이를 다시 병합하여 전체 시퀀스를 학습합니다.
   - 각 구간은 **이동 추정(Motion Estimation)** → **병합(Merging)** → **글로벌 조정(Global Adjustment)** 단계를 거치며, 점진적으로 전체 시퀀스를 최적화합니다.
3. **Isometric Loss와 Neighbor Loss**
   - **Isometric Loss**: Gaussian들이 지역적으로 강체 움직임(rigid motion)을 따르도록 제약을 부여합니다.
   - **Neighbor Loss**: 근처 Gaussian 간의 거리와 일관성을 유지하여 장면 구조를 보존합니다.
4. **Zoom-In, Zoom-Out과 같은 동작 가능**
   - 생성된 3D 장면은 줌 인/줌 아웃 및 좌우 이동과 같은 시점을 변경할 수 있어 다양한 응용 가능성을 제공합니다.

---
