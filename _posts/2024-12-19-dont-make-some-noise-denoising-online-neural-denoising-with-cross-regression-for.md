---
title: "[(Don't) Make Some Noise: Denoising] Online Neural Denoising with Cross-Regression for Interactive Rendering"
date: 2024-12-19 22:22:20
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3687938>

[Online Neural Denoising with Cross-Regression for Interactive Rendering | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687938)

# Interactive Rendering의 새로운 가능성: Online Neural Denoising

**Monte Carlo 렌더링**은 사실적인 조명 효과를 시뮬레이션할 수 있지만, 샘플링 제약으로 인해 노이즈 문제가 발생합니다. 특히, 실시간 렌더링 환경에서는 이 노이즈를 처리하기 위한 빠르고 효과적인 방법이 필수적입니다. 이번 논문에서는 기존의 방법과 차별화된 **하이브리드 노이즈 감소 프레임워크**를 소개합니다.

![](/assets/images/posts/440/img.jpg)

## 1. 주요 내용 요약

논문에서는 다음과 같은 접근 방식을 제안했습니다:

- **Cross-Regression**: 입력 이미지를 두 개의 파일럿 추정치로 나누고, 이를 바탕으로 뉴럴 네트워크 학습에 활용합니다.
- **Self-Supervised Loss**: 외부 데이터셋 없이, 런타임 이미지 시퀀스를 활용해 네트워크를 실시간으로 학습합니다.
- **Spatiotemporal Filter**: 공간 및 시간적 필터링을 통해 노이즈를 줄이고 디테일을 유지합니다.

---

## 2. 논문의 기여

이 방법은 기존의 두 가지 주요 방법론의 장점을 결합합니다:

- **클래식 회귀 기반 방법**: 단순하지만 비학습 기반으로 비선형 엣지 보존에 약점이 있음.
- **뉴럴 네트워크 기반 방법**: 데이터셋 준비가 필요하지만 복잡한 엣지 보존에 강점.

본 논문은 두 방법의 단점을 보완하고 데이터셋 준비 없이 실시간 학습을 가능하게 했습니다.

---
