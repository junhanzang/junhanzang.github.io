---
title: "Large Language Diffusion Models"
date: 2025-02-18 20:03:31
categories:
  - 인공지능
tags:
  - large language diffusion models
---

<https://ml-gsai.github.io/LLaDA-demo/>

[SOCIAL MEDIA TITLE TAG](https://ml-gsai.github.io/LLaDA-demo/)

<https://arxiv.org/abs/2502.09992>

[Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)

초록  
자기회귀 모델(ARM)은 대형 언어 모델(LLM)의 초석으로 널리 간주됩니다. 본 연구에서는 사전 학습과 지도 미세조정(SFT) 패러다임 하에 처음부터 학습된 확산 모델인 LLaDA를 소개하며, 이러한 인식을 도전합니다. LLaDA는 전방향 데이터 마스킹 과정과, 마스킹된 토큰을 예측하기 위해 기본 트랜스포머로 파라미터화된 역방향 과정을 통해 분포를 모델링합니다. 우도 경계를 최적화함으로써, 이는 확률적 추론을 위한 원칙적인 생성적 접근법을 제공합니다. 광범위한 벤치마크에서 LLaDA는 강력한 확장성을 보여주며, 자체 구성한 ARM 기준선보다 우수한 성능을 기록합니다. 특히, LLaDA 8B는 인컨텍스트 학습에서 LLaMA3 8B와 같은 강력한 LLM과 경쟁할 만하며, SFT 이후 다중 회차 대화와 같은 사례 연구에서 인상적인 명령 수행 능력을 나타냅니다. 더욱이, LLaDA는 역전 저주 문제를 해결하여, 역전 시의 완성 과제에서 GPT-4o를 능가합니다. 우리의 연구 결과는 확산 모델이 ARM에 대한 실행 가능하고 유망한 대안임을 확립하며, 앞서 논의한 주요 LLM 기능들이 본질적으로 ARM에만 국한된 것이 아님을 시사합니다.

프로젝트 페이지 및 코드: <https://ml-gsai.github.io/LLaDA-demo/>

기계 학습, ICML

---

우도 경계란 모델이 데이터를 생성하는 확률, 즉 로그 우도를 직접 최대화하는 대신에 최적화할 수 있는 계산 가능한 하한값을 의미합니다. 보통 이는 증거 하한(Evidence Lower Bound, ELBO)이라고도 불리며, 모델이 실제 데이터 분포에 근접하도록 유도하는 역할을 합니다.

확률 우도 경계라고 표현하는 것은, 모델의 학습 목표가 단순히 데이터가 발생할 확률을 높이는 것이 아니라, 그 확률의 하한값을 최적화함으로써 안정적인 확률적 추론을 가능하게 하는 접근법임을 강조하는 것입니다.

---
