---
title: "Open-sourcing circuit tracing tools"
date: 2025-06-07 18:32:20
categories:
  - 소식
tags:
  - neuronpedia
---

<https://www.anthropic.com/research/open-source-circuit-tracing?_bhlid=f5b0e86a94268a7109932dbb05966007b57ea583>

[Open-sourcing circuit-tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing?_bhlid=f5b0e86a94268a7109932dbb05966007b57ea583)

회로 추적 도구의 오픈소스 공개  
2025년 5월 29일

우리는 최근 해석 가능성(interpretability) 연구에서 대형 언어 모델의 사고 과정을 추적하는 새로운 방법을 소개했습니다. 오늘, 이 방법을 오픈소스로 공개하여 누구나 우리의 연구 위에 기반해 새로운 작업을 이어갈 수 있도록 합니다.

우리가 제안한 접근 방식은 **어트리뷰션 그래프(attribution graph)**를 생성하는 것입니다. 이 그래프는 모델이 특정 출력을 결정하기 위해 내부적으로 거친 일부 과정을 드러냅니다. 이번에 공개하는 오픈소스 라이브러리는 널리 사용되는 공개 가중치(pretrained open-weights) 모델에서 어트리뷰션 그래프를 생성할 수 있도록 지원하며, **Neuronpedia**에서 호스팅하는 프런트엔드를 통해 이 그래프를 **인터랙티브하게 탐색**할 수 있습니다.

이 프로젝트는 **Anthropic Fellows 프로그램** 참가자들이 주도했으며, **Decode Research**와 협력하여 진행되었습니다.

![](/assets/images/posts/569/img.webp)

Neuronpedia에서 제공하는 인터랙티브 그래프 탐색 UI 개요

이제 여러분은 Neuronpedia 인터페이스에 접속하여 원하는 프롬프트에 대해 직접 어트리뷰션 그래프(attribution graph)를 생성하고 시각화할 수 있습니다. 보다 고급 사용이나 연구 목적이라면 코드 저장소(repository)를 확인해보세요. 이번 공개를 통해 연구자들은 다음과 같은 작업이 가능해졌습니다:

- 지원되는 모델에 대해 직접 어트리뷰션 그래프를 생성해 **회로(circuit)를 추적**할 수 있습니다.
- **그래프를 시각화하고 주석을 달며 공유**할 수 있는 인터랙티브 프런트엔드가 제공됩니다.
- **특징(feature) 값들을 수정하고 그에 따른 모델 출력 변화를 관찰**함으로써 다양한 가설을 검증할 수 있습니다.

우리는 이미 이 도구들을 사용하여 **Gemma-2-2b** 및 **Llama-3.2-1b** 모델에서 **다단계 추론(multi-step reasoning)**, **다국어 표현(multilingual representation)** 등 흥미로운 행동을 분석했습니다. 데모 노트북에는 여러 사례와 분석이 담겨 있으며, 이를 통해 자세히 살펴볼 수 있습니다. 또한 커뮤니티의 참여를 환영하며, 아직 분석되지 않은 어트리뷰션 그래프들을 데모 노트북과 Neuronpedia에 제공하여 새로운 회로 탐색에 대한 영감을 드리고자 합니다.

Anthropic의 CEO **Dario Amodei**는 최근 해석 가능성 연구의 시급성을 강조했습니다. 현재 우리는 AI의 능력 향상 속도에 비해 그 **내부 작동 방식에 대한 이해는 매우 뒤처져** 있습니다. 우리는 이러한 도구들을 오픈소스로 공개함으로써 **언어 모델 내부에서 무슨 일이 일어나는지를 더 넓은 커뮤니티가 연구**할 수 있도록 돕고자 합니다. 이 도구들을 통해 모델의 행동을 이해하고, 나아가 이 도구 자체를 개선하려는 다양한 확장이 나타나기를 기대합니다.

이번 **회로 탐색 오픈소스 라이브러리**는 Anthropic Fellows인 **Michael Hanna**와 **Mateusz Piotrowski**가 개발했으며, **Emmanuel Ameisen**과 **Jack Lindsey**가 멘토로 참여했습니다. **Neuronpedia 통합 구현**은 Decode Research가 맡았으며, 프로젝트 리드는 **Johnny Lin**, 과학 책임자 및 디렉터는 **Curt Tigges**입니다. Gemma 관련 그래프는 **GemmaScope 프로젝트의 transcoder 학습 결과**를 기반으로 생성되었습니다. 질문이나 피드백은 GitHub 이슈를 통해 남겨주세요.
