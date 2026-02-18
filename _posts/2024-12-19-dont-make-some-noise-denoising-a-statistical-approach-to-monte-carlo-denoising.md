---
title: "[(Don't) Make Some Noise: Denoising] A Statistical Approach to Monte Carlo Denoising"
date: 2024-12-19 21:58:40
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687591>

[A Statistical Approach to Monte Carlo Denoising | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687591)

### **A Statistical Approach to Monte Carlo Denoising: 기존 방법의 세련된 개선**

Monte Carlo 기반 렌더링에서 발생하는 노이즈 문제는 오래된 과제입니다. 이번 SIGGRAPH Asia 2024에서 발표된 **"A Statistical Approach to Monte Carlo Denoising"** 논문은 딥러닝 대신 통계적 필터링을 활용하여 효율적이고 안정적인 노이즈 제거 방법을 제안했습니다. 이 연구는 흥미로운 접근법을 보여줬지만, 기존 방법론에서 크게 벗어난 혁신은 아니라고 느꼈습니다.

![](/assets/images/posts/439/img.jpg)

---

### **핵심 내용 요약**

1. **Monte Carlo 노이즈 제거의 통계적 접근**
   - Gaussian 필터 기반의 기존 방법론을 개선하여, Welch's t-test와 Box-Cox 변환을 사용한 통계적 노이즈 제거를 제안.
   - G-buffer 데이터를 활용하여 픽셀 간 관계를 분석하고, 적응형 필터링으로 노이즈를 줄임.
2. **Box-Cox 변환 도입**
   - Gaussian 필터가 특정 상황에서 발생하는 오버플로우 문제를 해결하기 위해 분포를 변환.
   - 이를 통해 다양한 샘플 분포를 안정적으로 처리.
3. **성능과 효율성**
   - 딥러닝 기반 디노이저와 비교해 사전 학습 없이도 빠르고 안정적인 성능을 제공.
   - Neural Denoising 방식과 비교해 계산 비용이 적지만, 복잡한 디테일 표현에서는 한계가 있음.

---
