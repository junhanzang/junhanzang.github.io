---
title: "[Text, Texturing, and Stylization] Compositional Neural Textures"
date: 2024-12-19 02:43:34
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687561>

[Compositional Neural Textures | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687561)

### **Compositional Neural Textures: 텍스처 편집을 위한 새로운 접근**

**Compositional Neural Textures**는 텍스처를 더 효율적으로 생성, 편집할 수 있도록 설계된 완전히 새로운 방식의 신경망 기반 텍스처 모델입니다. 이 논문은 텍스처를 **"Neural Textons"**라는 개념으로 분해하여 표현하며, 각 텍스처 요소를 2D Gaussian 함수로 나타냅니다. 이를 통해 텍스처의 구조와 외형을 명확히 분리하고 다양한 응용이 가능하게 합니다.

![](/assets/images/posts/433/img.jpg)

---

### **핵심 아이디어**

1. **Neural Textons의 도입**  
   텍스처를 구성하는 반복적인 패턴(텍스톤)을 2D Gaussian으로 모델링하여, 텍스처의 **공간적 구조**와 **외형**을 분리.
   - 텍스톤의 구조: Gaussian의 중심과 공분산으로 표현.
   - 텍스톤의 외형: 특징 벡터(feature vector)로 표현.
2. **완전 비지도 학습**  
   레이블 없이 학습하며, 텍스처를 편집 가능한 구성 요소로 분해.
3. **다양한 텍스처 편집 응용**
   - 텍스처 전이: 한 텍스처의 구조를 다른 텍스처의 외형과 결합.
   - 텍스처 다양화: 텍스톤의 외형을 임의로 섞어 새로운 텍스처 생성.
   - 텍스처 애니메이션: 텍스톤의 이동 및 변형을 통해 텍스처를 애니메이션화.

---
