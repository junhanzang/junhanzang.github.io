---
title: "Leave No Context Behind： Efficient Infinite Context Transformers with Infini attention"
date: 2024-04-28 23:59:49
categories:
  - 일상생활
tags:
  - leave no context behind： efficient infinite context transformers with infini attention
---

<https://www.youtube.com/watch?v=r_UBBfTPcF0>

<https://arxiv.org/abs/2404.07143>

[Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)

[2404.07143v1 ko.pdf

1.01MB](./file/2404.07143v1 ko.pdf)

1.이들이 설계한 구조

![](/assets/images/posts/153/img.png)

그냥 더 간단하게 보면 다음과 같다.

![](/assets/images/posts/153/img_1.png)

그냥 이전 데이터와 현재 데이터의 데이터 불균형화를 막는 형식

실제로 이걸 여러겹 쌓으면 다음과 같이됨

![](/assets/images/posts/153/img_2.png)

이상하지 않은가?

Infini-Transformer가 linear하다는 것을?

즉, 이건 궁극적으로 활용되지 못하는 방식이다. softmax로 선형 조건을 비선형으로 짜치는 것이다.

이게 그 공식이고

![](/assets/images/posts/153/img_3.png)

그리고 이러한 방식은 1990년대에 시도되었고

![](/assets/images/posts/153/img_4.png)

  Reference에 보면 나와있다. 그리고 없어졌다. -> 잘안된거

Performance 성능에서는 좋게 나왔는데, 500k다 단어가. 그러면 infinite이라고 우리가 봐야하나?

![](/assets/images/posts/153/img_5.png)

500,000개 글자인데, 생각보다 너무 적다. 책한권 아닌가? 이 부분은 이미 Gemini나 다른 연구에서 attention 매커니즘으로 잘되고 있다는 것을 확인했다.

Linearity에 의해 이미 불안정한 것에 추가로 올란다면, 나는 부정적이다

나는 window context가 무한인건줄알고 읽었는데, 아니었다는 점이 더 실망감이 컸다.

저보다 잘 서술하신 분을 보시려면 아래로

<https://ostin.tistory.com/513>

[Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://ostin.tistory.com/513)
