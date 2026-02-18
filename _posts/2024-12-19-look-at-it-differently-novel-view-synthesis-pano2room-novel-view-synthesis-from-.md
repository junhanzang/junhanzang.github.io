---
title: "[Look at it Differently: Novel View Synthesis] Pano2Room: Novel View Synthesis from a Single Indoor Panorama"
date: 2024-12-19 01:02:32
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://arxiv.org/abs/2408.11413>

[Pano2Room: Novel View Synthesis from a Single Indoor Panorama](https://arxiv.org/abs/2408.11413)

### **Pano2Room: 단일 파노라마 이미지를 활용한 새로운 시점 생성**

**"Pano2Room"** 논문은 단일 실내 파노라마 이미지를 기반으로 고품질 3D 장면을 재구성하고 새로운 시점을 생성하는 방법을 제안했습니다. 이 연구는 **3D Gaussian Splatting (3DGS)**와 **Stable Diffusion**을 활용하여 실내 공간의 구조와 텍스처를 사실적으로 재현합니다.

![](/assets/images/posts/421/img.jpg)

---

### **핵심 내용 요약**

1. **단일 파노라마에서 3D 메쉬 생성**
   - **Pano2Mesh 모듈**을 사용하여 파노라마 이미지를 초기 3D 메쉬로 변환합니다.
   - **Depth Edge Filter**를 활용해 객체 간의 경계를 명확히 구분하고 메쉬의 정확도를 높였습니다.
2. **Iterative Refinement**
   - **RGBD Inpainting**: 누락된 영역을 보완하기 위해 파노라마 RGB와 깊이 정보를 생성합니다.
   - **Geometry Conflict Avoidance**: 충돌이 발생하지 않도록 새로운 메쉬를 기존 메쉬에 통합합니다.
   - **카메라 탐색 최적화**: 적절한 카메라 뷰포인트를 선택해 반복 작업의 효율성을 높였습니다.
3. **Stable Diffusion 기반의 텍스처 생성**
   - **Stable Diffusion Fine-Tuning (SDFT)**: 파노라마 스타일과 일관성을 유지하며 텍스처를 보완합니다.
   - **Monocular Depth Fusion**: 여러 뷰에서의 깊이 정보를 융합해 더욱 정밀한 3D 깊이 맵을 생성합니다.
4. **Mesh to 3DGS 변환**
   - 최적화된 메쉬를 3DGS로 변환하여 사실적이고 매끄러운 새로운 시점을 생성합니다.
   - **Photometric Loss**를 사용해 텍스처 품질을 개선합니다.

---
