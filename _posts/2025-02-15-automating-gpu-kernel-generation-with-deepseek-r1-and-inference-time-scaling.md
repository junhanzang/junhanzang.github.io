---
title: "Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling"
date: 2025-02-15 15:30:40
categories:
  - Article
tags:
  - deepseek-r1
  - automating gpu
---

<https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940>

[Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling | NVIDIA Technical Blog](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940)

![](/assets/images/posts/511/img.png)

**AI 모델이 점점 더 복잡한 문제를 해결할 수 있는 능력을 확장함에 따라**, 추론 시점 스케일링(test-time scaling, inference-time scaling)이라는 새로운 스케일링 법칙이 등장하고 있습니다. AI 추론 또는 롱-씽킹(long-thinking)이라고도 불리는 이 기법은 추론 과정에서 더 많은 계산 리소스를 할당하여 여러 가능한 결과를 평가한 뒤 최적의 결과를 선택함으로써 모델의 성능을 높입니다. 이 방법을 통해 AI는 복잡한 문제를 인간이 문제를 쪼개고 각 부분을 해결하여 최종 답안을 도출하는 방식과 유사하게 전략적으로 접근하고 체계적으로 문제를 해결할 수 있게 됩니다.

이 글에서는 NVIDIA 엔지니어들이 최신 오픈소스 모델 중 하나인 **DeepSeek-R1** 모델에, 추론 중 추가 계산 자원을 투입해 복잡한 문제를 해결하는 실험을 다룹니다. 이 실험의 목표는 주어진 주의(attention) 연산의 여러 변형에 대해, 명시적 프로그래밍 없이도 수치적으로 정확하고 최적화된 GPU 주의 커널(attention kernel)을 자동 생성하는 것이었습니다.

실험 결과, 어떤 경우에는 숙련된 엔지니어들이 개발한 최적화 커널보다도 더 나은 성능을 보였습니다.

## 최적화된 주의 커널의 필요성과 관련 과제

주의(attention)는 대규모 언어 모델(LLM)의 발전을 획기적으로 이끈 핵심 개념입니다. 모델이 작업을 수행할 때 입력 데이터에서 가장 중요한 부분에 선택적으로 집중할 수 있게 해주며, 이를 통해 모델은 더욱 정확한 예측을 하고 데이터에 내재된 패턴을 찾아낼 수 있습니다.

**NVIDIA GTC 2025**  
주의 연산의 계산 복잡도는 입력 시퀀스 길이에 대해 이차적으로 증가합니다. 이는 단순한 구현에서 발생할 수 있는 런타임 에러(예: 메모리 부족) 등을 방지하고 계산 효율성을 확보하기 위해, 하위 수준(GPU 커널)의 최적화된 구현을 개발해야 한다는 필요성을 일깨웁니다.

주의에는 인과(causal), 상대적 위치 임베딩(relative positional embeddings), alibi 등 다양한 변형이 있으며, 실제 작업에서는 이들 중 여러 변형을 혼합해서 사용하는 경우가 많습니다.

멀티모달 모델(예: 비전 트랜스포머)은 공간-시간(spatio-temporal) 정보를 유지해야 하는 컴퓨터 비전·영상 생성 모델 등에서 주로 다뤄지는 **공간 이웃 주의(Spatial Neighborhood Attention)** 같은 특수 주의 기법이 필요하기 때문에 또 다른 난관이 생깁니다.

---

공간 이웃 주의(Spatial Neighborhood Attention)는 이미지나 영상 같은 2D(또는 3D) 입력에서 **특정 공간 영역(이웃 범위)에 한정하여 집중하는** 주의 메커니즘입니다. 예컨대 영상 속 한 지점 주변의 픽셀만을 우선적으로 살펴보는 방식으로, 국소적으로 관련성이 높은 부분에 초점을 맞춰 효율적으로 정보를 처리할 수 있습니다.

일반적인(글로벌) 주의는 입력 전체를 대상으로 하기 때문에 시퀀스 길이가 길어지면 계산 복잡도가 급격히 증가합니다. 반면 이웃 주의는 주변 영역만 고려하여 연산하므로, **특히 영상·비디오 등에서 자주 나타나는 공간적·시간적 특성을 잘 포착**하면서도 비교적 계산 부담을 줄일 수 있습니다. 이런 이유로 멀티모달 모델(예: 비전 트랜스포머)에서 **이미지나 영상 내 공간적 구조를 유지**하고, 중요한 영역을 놓치지 않도록 하는 데에 자주 활용됩니다.

예를 들어, 여러 dilation(확장) 값을 적용해 입력 이미지를 단계적으로 더 넓은 이웃 영역까지 주의 범위를 확대함으로써, 처음에는 제한된 범위만 보다가 점차 전체 영상을 파악하는 방식으로 활용하기도 합니다. 이를 통해 모델은 영상 내 국소적 특징에서 시작해 전체 맥락까지 다룰 수 있게 됩니다.

---
