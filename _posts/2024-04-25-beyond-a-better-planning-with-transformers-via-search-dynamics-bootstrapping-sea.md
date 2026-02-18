---
title: "Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping (Searchformer)"
date: 2024-04-25 22:58:13
categories:
  - 일상생활
tags:
  - searchformer
---

<https://www.youtube.com/watch?v=PW4JiJ-WaY4>

<https://arxiv.org/abs/2402.14083>

[Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083)

[Beyond A＊： Better Planning with Transformers via Search Dynamics Bootstrapping (Searchformer) (PW4JiJ-WaY4).srt

0.06MB](./file/Beyond A＊： Better Planning with Transformers via Search Dynamics Bootstrapping (Searchformer) (PW4JiJ-WaY4).srt)

1. 이건 planning에 대한 논문이다. ( 정답 주는 건 아니다 )

2. Prompt -> Trace -> Plan 으로 64개 뽑고 그중에서 제일 짧은 답을 Response로 준다

3. 이중에서 Plan이 제일 짧다면 이걸로 대체하고 이 후에 Trace가 짧은 순으로 대체한다

4. 이중에서 Planning 학습은 LLM 모델 1.5B정도되는 걸 사용하며 Trace -> Plan을 학습시키는거다.

5. A\*보다 짧은 답을 뽑아낼 수 있다는데, 결국 A\*의 휴리스틱을 줄이는 정도이다

6. 그래도 일반화라는 관점에서 좋은 논문이다. (환경은 그렇지 않지만)
