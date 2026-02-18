---
title: "Efficient Streaming Language Models with Attention Sinks"
date: 2024-01-07 13:06:36
categories:
  - 인공지능
tags:
  - Streamable Attention
---

**스트리밍 언어 모델의 효율성 향상과 Attention Sinks**

언어 모델링은 텍스트를 생성하고 이해하는 데 사용되는 인공지능 알고리즘입니다. 기존 언어 모델은 데이터 처리에 많은 자원을 필요로 하며, 특히 긴 텍스트를 다룰 때 비효율적일 수 있습니다. 긴 텍스트를 처리할 때 기존 모델들은 느려지고 많은 메모리를 필요로 합니다. 이는 특히 모델이 긴 텍스트를 처리하며 정확도를 유지하려고 할 때 더욱 심해집니다.

이에 대한 해결책으로 'Attention Sinks'라는 새로운 개념이 도입되었습니다. 이 기법은 언어 모델이 텍스트의 특정 부분에만 주의를 기울이도록 하여, 불필요한 계산을 줄이고 모델의 속도와 효율성을 향상시킵니다. 이 새로운 방식을 적용한 모델은 긴 텍스트를 더 빠르고 효율적으로 처리할 수 있으며, 실험 결과에 따르면 기존 방법보다 훨씬 나은 성능을 보였습니다.

**Dense Attention과 Window Attention**

'Dense Attention'은 모든 입력 단어 간의 관계를 고려하며, 이는 모델이 텍스트의 전체적인 맥락을 더 잘 이해할 수 있게 해줍니다. 하지만, 이 방법은 계산량이 많고 긴 텍스트를 처리할 때 비효율적일 수 있습니다. 반면에, 'Window Attention'은 주어진 '창' 내의 단어들만을 고려합니다. 이 방식은 계산량을 줄이고 처리 속도를 빠르게 하지만, 각 창 내에서만 관계를 고려하기 때문에 전체 텍스트의 맥락을 완전히 파악하기 어려울 수 있습니다.

**Streamable Attention의 역할**

'Streamable Attention'은 실시간으로 텍스트를 처리하는 데 중점을 둡니다. 이는 연속적으로 들어오는 데이터를 효율적으로 처리할 수 있도록 설계된 모델입니다. Streamable Attention은 긴 텍스트 스트림을 실시간으로 처리할 수 있게 하면서도, 중요한 정보에만 집중할 수 있도록 합니다. 이는 Attention Sinks와 함께 사용될 때, 모델의 효율성과 처리 속도를 극대화합니다.

'Efficient Streaming Language Models with Attention Sinks' 연구에서는 이러한 기법들의 차이점을 인식하고, 효율성과 정확성을 균형있게 유지하기 위해 Attention Sinks와 Streamable Attention을 도입했습니다. 이는 언어 모델이 긴 텍스트를 효율적으로 처리하면서도 필요한 정보에만 주목할 수 있도록 하는 데 중점을 둡니다.

참고 필요

**Big Bird 모델의 역할과 중요성**

Big Bird 모델은 Google Research에서 개발된 혁신적인 언어 모델입니다. 이 모델은 특히 긴 문서를 처리하는 데 효과적인 방법으로 설계되었습니다.

1. **개념**: Big Bird는 '스파스(sparse) 어텐션 메커니즘'을 사용합니다. 이는 모든 단어 간의 관계를 고려하는 대신, 중요한 단어들 사이의 관계에만 집중합니다. 이는 Dense Attention과는 달리, 메모리 사용량과 계산량을 크게 줄여줍니다.
2. **장점**: Big Bird는 긴 텍스트의 처리에 있어서 뛰어난 효율성과 정확도를 제공합니다. 이는 특히 대용량의 데이터를 처리하는 데 있어서 기존 모델들보다 뛰어난 성능을 보여줍니다.
3. **적용**: 이 모델은 자연어 처리의 여러 분야에서 사용될 수 있으며, 특히 긴 문서나 대화를 분석하는 데 유용합니다.

**Big Bird와 다른 언어 모델들과의 관계**

- Big Bird는 Dense Attention의 전체적인 맥락 이해 능력과 Window Attention의 효율성을 결합한 것으로 볼 수 있습니다.
- Streamable Attention과 마찬가지로, Big Bird도 연속적인 데이터 스트림 처리에 효과적입니다.
- 'Efficient Streaming Language Models with Attention Sinks'에서 소개된 Attention Sinks 기법과 함께 사용될 경우, 언어 모델의 효율성과 정확성을 더욱 향상시킬 수 있습니다.

마지막 의견

Streamable Attention은 cache로 이전 문맥을 가져오는 방식이고 Big Bird는 양방향 bert 형식이다. 논리적으로는 Big Bird가 더 맞지만, 가정과 실험결과에서는 Streamable Attention도 나쁘지 않음을 보인다.

Streamable Attention은 문제점은 0에 결국 cache학습이 잘 되는지에 대한 점인데, 지속적인 숙제로 남을 것이다.
