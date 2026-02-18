---
title: "[Make it Yours - Customizing Image Generation] Customizing Text-to-Image Models with a Single Image Pair"
date: 2024-12-17 23:47:50
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687642>

[Customizing Text-to-Image Models with a Single Image Pair | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687642)

### **Customizing Text-to-Image Models with a Single Image Pair**

시그라프 아시아 2024에서 주목했던 논문 중 하나는 \*\*"Customizing Text-to-Image Models with a Single Image Pair"\*\*입니다. 이 논문은 **스타일과 내용의 분리**라는 오래된 문제를 효율적으로 해결하는 **Dual LoRA 구조**를 제안했습니다.

---

### **기존 방식의 한계: Overfitting 문제**

기존 Text-to-Image 모델들은 스타일을 학습할 때 **단일 이미지**에 Overfitting되는 문제가 있었습니다. 예를 들어:

- **스타일과 내용이 혼재**되어 스타일만 학습하려 해도 원본 이미지의 주제(Subject)나 구조까지 함께 학습되어버립니다.
- 그 결과 **내용의 구조를 유지하지 못하거나**, 새로운 컨텐츠에 스타일을 적용할 때도 **오버피팅**이 발생합니다.

---
