---
title: "[(Don't) Make Some Noise: Denoising] Neural Kernel Regression for Consistent Monte Carlo Denoising"
date: 2024-12-19 22:47:46
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3687949>

[Neural Kernel Regression for Consistent Monte Carlo Denoising | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687949)

### **Neural Kernel Regression: Monte Carlo Denoising의 새로운 가능성**

Monte Carlo 경로 추적은 사실적인 렌더링에서 필수적이지만, 낮은 샘플링 비율(SPP)에서는 여전히 심각한 노이즈 문제가 발생합니다. 기존 딥러닝 기반 디노이저는 노이즈를 줄이는 데 효과적이지만, 높은 SPP 환경에서는 일관성과 정확도가 떨어지는 한계를 보입니다.  
이번 SIGGRAPH Asia 2024에서 소개된 **"Neural Kernel Regression for Consistent Monte Carlo Denoising"**는 이 문제를 해결하기 위한 새로운 접근 방식을 제안했습니다.

![](/assets/images/posts/443/img.jpg)

---

### **1. 기존 문제점**

1. **딥러닝 기반 디노이저의 한계**
   - Neural Networks(NN)를 사용한 디노이저는 낮은 SPP에서는 뛰어난 성능을 보이지만, 높은 SPP 환경에서는 **불일관성**(inconsistency) 문제로 인해 품질이 저하됩니다.
   - 이는 훈련 데이터의 한계와 네트워크 구조 자체의 제약에서 비롯됩니다.
2. **전통적 포스트-코렉션(post-correction)의 문제점**
   - 기존 방법은 **편향된 이미지(biased image)**와 **비편향 이미지(unbiased image)**를 조합해 일관성을 보장하지만, 노이즈가 심한 낮은 SPP 환경에서는 아티팩트가 발생하기 쉽습니다.

---
