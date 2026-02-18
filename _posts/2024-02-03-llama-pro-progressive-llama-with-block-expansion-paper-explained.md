---
title: "LLaMA Pro： Progressive LLaMA with Block Expansion (Paper Explained)"
date: 2024-02-03 22:33:08
categories:
  - 인공지능
tags:
  - LLaMA Pro
---

<https://arxiv.org/html/2401.02415v1>

[LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/html/2401.02415v1)

짧은 요약

LLaMA Pro는 기존 LLaMA 언어 모델을 기반으로 새로운 블록 확장 기능을 통해 지속적인 학습을 가능하게 하는 개선된 모델입니다. 기존 지식을 유지하면서 새로운 데이터를 효율적으로 학습할 수 있도록 설계되었습니다. 이는 특정 작업, 예를 들어 코딩이나 수학 벤치마크에 대한 모델의 능력을 강화하기 위해 추가된 레이어를 미세 조정함으로써 달성됩니다. 이 과정은 기계 학습 모델이 이전에 배운 내용을 잊어버리는 문제를 최소화하면서 새로운 지식을 효과적으로 통합할 수 있게 합니다.

![](/assets/images/posts/139/img.png)

Zero - Linear를 통해 다른 expert에서 가져온 weight를 freeze해서 유지하는 것이 핵심

결과값이 0이라면 동일한 weight 및 학습결과라고 가정해서 이를 바탕으로 layer들을 추가로 쌓는 방식

차라리 이럴거면 adaptor랑 Lora 처럼한거네하고 이를 추가하는 형식을 바탕으로 MOE( mixture of expert )로 하는게 좋지 않을까?

단일 모델이어도 유지한다는 점에서는 좋지만 결국 학습되었던 것 이상의 성능을 내지 못하는 것은 아쉽다.
