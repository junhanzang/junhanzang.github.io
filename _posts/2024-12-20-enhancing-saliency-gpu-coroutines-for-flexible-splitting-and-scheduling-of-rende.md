---
title: "[Enhancing, Saliency] GPU Coroutines for Flexible Splitting and Scheduling of Rendering Tasks"
date: 2024-12-20 18:26:59
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
tags:
  - gpu coroutines for flexible splitting and scheduling of rendering tasks
---

<https://dl.acm.org/doi/10.1145/3687766>

[GPU Coroutines for Flexible Splitting and Scheduling of Rendering Tasks | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687766)

#### 간단 요약 및 소개

GPU 코루틴을 활용하여 메가 커널 기반 GPU 렌더링의 비효율성을 해결하는 기술입니다. 렌더링 작업을 코루틴으로 분할하고 스케줄링하여 더 나은 성능과 관리 효율성을 제공합니다.

![](/assets/images/posts/467/img.jpg)

#### 기존 문제점

- **메가 커널 비효율성:** GPU의 병렬성에 적합하지 않은 거대한 커널 구조는 스레드 다이버전스와 높은 레지스터 사용으로 인한 성능 저하를 초래.
- **수작업 분할의 어려움:** 메가 커널을 수작업으로 분할하고 스케줄링하는 것은 복잡하고 오류가 발생하기 쉬움.
- **플랫폼 의존성:** 기존 GPU 스케줄링 방법은 특정 하드웨어 또는 소프트웨어 플랫폼에 제한적.

#### 해결법

1. **코루틴 도입:** GPU에 적합한 스택리스(stackless) 코루틴 모델을 사용하여 렌더링 작업을 유연하게 분할.
2. **스케줄러 설계:** 사용자 정의가 가능한 내장 스케줄러(예: Persistent Threads, Wavefront Scheduler)를 통해 다양한 작업에 적응.
3. **자동화된 코드 변환:** 사용자가 $suspend를 통해 코루틴 분할 지점을 지정하면 시스템이 자동으로 코드를 변환하고 상태 관리.
4. **효율적 데이터 관리:** SoA(Structure of Arrays) 메모리 구조를 사용하여 메모리 대역폭을 최적화.

#### 기여

- **유연한 분할 및 스케줄링:** 복잡한 렌더링 작업을 코루틴으로 분할하고 효과적으로 스케줄링.
- **플랫폼 독립성:** 하드웨어에 구애받지 않는 범용 GPU 코루틴 모델.
- **성능 개선:** 메가 커널 대비 더 나은 스레드 일관성과 낮은 메모리 트래픽으로 성능 향상.

#### 한계 및 개인적 생각

- **계산 비용:** 코루틴 분할 및 스케줄링 과정에서 추가적인 계산 비용 발생.
- **학습 곡선:** 새로운 언어 기능 및 스케줄링 인터페이스 학습 필요.
- **일부 한계:** 작은 서브루틴 간의 스케줄링이 과도하게 세밀할 경우, 오히려 오버헤드가 증가할 가능성.

실제로 이러한 방식이 GPU 병렬 처리의 새로운 가능성을 열어주었으며, 특히 물리 기반 렌더링이나 복잡한 데이터 흐름 관리가 필요한 작업에서 매우 유용할 것으로 기대됩니다.
