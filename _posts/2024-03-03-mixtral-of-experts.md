---
title: "Mixtral of Experts"
date: 2024-03-03 23:27:01
categories:
  - 인공지능
tags:
  - Moe
  - Mixtral of Experts
---

<https://arxiv.org/abs/2401.04088>

[Mixtral of Experts](https://arxiv.org/abs/2401.04088)

간단하게 표현하면 x개의 모델이 존재하며 앞에 FFN을 추가해서 그중 높은 점수 2개를 결합하여 사용하는 방식

Routing에 대한 인공지능을 추가한 것과 동일

토큰이 들어오면 이 토큰이 어떤 인공지능을 사용해야되는지 길을 알려주는 것과 같음
