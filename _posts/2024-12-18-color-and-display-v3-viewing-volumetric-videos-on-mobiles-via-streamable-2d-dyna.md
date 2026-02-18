---
title: "[Color and Display] V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians"
date: 2024-12-18 02:01:00
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3687935>

[V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687935)

### **V³: 모바일에서 볼류메트릭 비디오를 실시간으로 구현하기**

**"V³"** 논문은 3D 볼류메트릭 비디오를 모바일 기기에서 스트리밍할 수 있도록 하는 혁신적인 접근법을 제시했습니다. 핵심은 **3D Gaussian Splatting** 데이터를 **2D 비디오**로 변환해 하드웨어 비디오 코덱과 호환성을 확보하는 것입니다.

---

### **핵심 내용**

1. **3D 데이터를 2D로 변환**
   - **3D Gaussian Splatting**을 2D 이미지로 압축하고, 하드웨어 비디오 코덱을 사용해 **효율적으로 전송 및 재생**합니다.
   - 각 Gaussian 속성(위치, 색상, 회전 등)은 2D 이미지의 픽셀에 매핑됩니다.
2. **Temporal Consistency**
   - **Residual Entropy Loss**: 연속 프레임 간 속성의 변화량을 최소화해 압축 손실을 줄입니다.
   - **Temporal Loss**: 시간축에서 프레임 간 일관성을 유지합니다.
3. **저장 및 렌더링 최적화**
   - Morton Sorting을 통해 3D Gaussian 속성을 2D로 매핑하여 **공간적 일관성**을 확보했습니다.
   - **uint16**과 **uint8**을 사용해 정밀도를 유지하면서도 데이터 크기를 최소화했습니다.

---
