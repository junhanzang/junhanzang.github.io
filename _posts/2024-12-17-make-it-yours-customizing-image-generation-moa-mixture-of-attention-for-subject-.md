---
title: "[Make it Yours - Customizing Image Generation] MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation"
date: 2024-12-17 23:20:37
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687662>

[MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation | SIGGRAPH Asia 2024 Conference P](https://dl.acm.org/doi/10.1145/3680528.3687662)

### **MoA: Mixture-of-Attention for Subject-Context Disentanglement**

시그라프 아시아 2024에서 흥미롭게 들었던 논문 중 하나는 \*\*"MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation"\*\*입니다. 처음에는 Loss 구조에 집중해서 이해했지만, 논문을 다시 검토하면서 핵심이 다른 곳에 있다는 것을 깨달았습니다.

---

### **기존 Personalized Image Generation의 문제**

기존 Text-to-Image 모델들은 \*\*주제(Subject)\*\*와 \*\*맥락(Context)\*\*을 구분하지 못하고 함께 학습해버리는 경우가 많습니다. 예를 들어, 특정 인물이나 객체를 학습하려고 하면 배경과 같이 엉뚱한 부분까지 모델이 학습해버려, 새로운 장면에 해당 주제를 자연스럽게 삽입하지 못하는 문제가 발생합니다.

---
