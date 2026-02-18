---
title: "AlphaGeometry： Solving olympiad geometry without human demonstrations"
date: 2024-03-08 06:24:59
categories:
  - 인공지능
tags:
  - AlphaGeometry
---

<https://www.nature.com/articles/s41586-023-06747-5>

문제를 입력받아 다음과 같은 도메인으로 1차 처리

![](/assets/images/posts/144/img.png)

이후에 이를 바탕으로 이게 되는지 안되는지 계속 tree search후, 정답 조건식들을 만족하게 되면 정답을 맞추는 방식

trial and error를 계속 사용하되 그것이 llm을 사용하게 하는 것으로 오히려 강화학습에 가깝지 않나 싶음

결국 무한의 도메인을 더 좁은 도메인으로 좁히는 방식으로 푸는 것

하지만 무한한 trial and error도 풀지 못한 5개가 존재하는데, 이에 대해서 알려주었다면 더 좋겠음

좋은 발전이지만 아직 논리는 이해한다고 보긴 어려워보임
