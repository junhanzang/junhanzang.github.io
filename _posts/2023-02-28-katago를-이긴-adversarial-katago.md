---
title: "Katago를 이긴 Adversarial Katago"
date: 2023-02-28 00:57:33
categories:
  - 일상생활
tags:
  - Adversarial attack
  - goattack
---

<https://goattack.far.ai/>

[Adversarial Policies in Go - Game Viewer](https://goattack.far.ai/)

참조 영상

<https://www.youtube.com/watch?v=Kv23z_9X3_M>

현재 바둑인공중 최강중 하나인 카타고가 아마추어 1단에게 15판중 14판을 졌다. 이는 켈린 펠린이라는 맥길대학교 학생이 Adversarial Attack이라는 기법을 활용하여 카타고 또는 현재 존재하는 바둑인공지능들의 작동방식과 약점을 보여준 것이다. 참조영상 13분 쯤에 김성룡님 말씀처럼 해당 인공지능은 결국 Value-Based이기 때문에 그 상황에서도 최대한 점수를 얻어내기 위한 방식을 취한다. 이는 Montecarlo-search에 기반하여 가장 점수를 높게 낼 수 있는 수를 추적하기 때문에 발생하는 약점이다. 이전에 설명했던 예시를 빗대보면, 개와 고양이 사이의 분포를 교묘하게 파고들듯 이번에는 죽은 돌과 산돌의 사이의 분포를 교묘하게 파고든 것이다.

앞으로 인공지능에서는 Attacker와 Defender간에 승부가 계속될 것이다. 특히 암호분야와 같이 지속적으로 Attacker와 Defender 분야는 더욱더 그럴 것이다.
