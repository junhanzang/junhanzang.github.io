---
title: "[Make it Yours - Customizing Image Generation] PALP: Prompt Aligned Personalization of Text-to-Image Models"
date: 2024-12-17 23:42:55
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687604>

[PALP: Prompt Aligned Personalization of Text-to-Image Models | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687604)

### **PALP: 데이터 확보의 중요성과 아쉬운 설명**

이번 시그라프 아시아 2024에서 흥미롭게 들었던 논문 중 하나는 PALP: Prompt Aligned Personalization of Text-to-Image Models입니다. 하지만 발표를 들을 당시에는 **데이터 확보**의 중요성에 대한 설명이 부족하다는 느낌이 들었습니다.

---

### **데이터 확보와 모델의 한계**

PALP는 기존 **Text-to-Image Personalization** 기법들이 겪는 문제, 즉 \*\*주제의 보존(subject fidelity)\*\*과 **텍스트 프롬프트 정렬(prompt alignment)** 사이의 **트레이드오프**를 해결하는 데 초점을 맞추고 있습니다.

- 기존 방식은 **주제**를 학습하는 과정에서 복잡한 프롬프트와의 정렬이 무너지는 경우가 많습니다.
- PALP는 이러한 문제를 해결하기 위해 **Score Distillation Sampling (SDS)**과 **Prompt-Aligned Loss**를 결합했습니다.

그러나 당시 발표에서는 **어떤 데이터**를 활용해 모델을 학습했는지에 대한 설명이 부족했고, 오히려 **Loss 구조**만을 강조하는 경향이 있었습니다.

---
