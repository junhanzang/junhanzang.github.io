---
title: "[Text, Texturing, and Stylization] InstanceTex: Instance-level Controllable Texture Synthesis for 3D Scenes via Diffusion Priors"
date: 2024-12-19 02:11:43
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687633>

[InstanceTex: Instance-level Controllable Texture Synthesis for 3D Scenes via Diffusion Priors | SIGGRAPH Asia 2024 Conference Pa](https://dl.acm.org/doi/10.1145/3680528.3687633)

### **InstanceTex: 텍스처 생성의 새로운 패러다임**

**InstanceTex**는 복잡한 3D 장면에서 객체별 텍스처 제어를 가능하게 하는 새로운 접근법을 제안합니다. 이 연구는 **Instance Layout Representation**을 도입하여 3D 씬의 텍스처를 생성할 때 개별 객체의 스타일과 장면 전체의 일관성을 동시에 유지할 수 있도록 설계되었습니다.

![](/assets/images/posts/429/img.jpg)

---

### **주요 내용**

1. **핵심 개념: Instance Layout Representation**
   - 3D 장면 내 각각의 객체에 대해 텍스트 프롬프트와 위치 정보를 정의한 **Instance Layout**을 사용합니다.
   - 이 레이아웃을 바탕으로 객체별 텍스처를 생성하며, 장면의 전체 스타일 일관성을 유지합니다.
2. **텍스처 생성 파이프라인**
   - **단일 뷰 기반 디퓨전**: Depth, Line Art, Position Map을 활용하여 객체별 텍스처 생성.
   - **다중 뷰 일관성 강화**: 로컬 다중 뷰 디퓨전을 통해 여러 시점의 텍스처를 조정하여 일관성을 개선.
   - **Neural MipTexture**: 멀티스케일 UV 맵핑을 도입해 앨리어싱 문제를 해결하고 고해상도의 텍스처 맵 생성.
3. **주요 결과**
   - 텍스처의 품질과 일관성 면에서 기존 방법들(Text2Tex, SyncMVD 등)을 능가.
   - 복잡한 장면에서도 높은 텍스처 품질과 정확한 스타일 구현 가능.

---
