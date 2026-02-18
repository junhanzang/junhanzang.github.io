---
title: "Hogwild! Inference: Parallel LLM Generation via Concurrent Attention"
date: 2025-04-18 18:45:16
categories:
  - 인공지능
tags:
  - LLM
  - hogwild! inference
---

<https://github.com/eqimp/hogwild_llm?_bhlid=d0e43178b4d426d6b68fdb2f5c141720a923cacc>

[GitHub - eqimp/hogwild\_llm: Official PyTorch implementation for Hogwild! Inference: Parallel LLM Generation with a Concurrent At](https://github.com/eqimp/hogwild_llm?_bhlid=d0e43178b4d426d6b68fdb2f5c141720a923cacc)

<https://www.arxiv.org/abs/2504.06261?_bhlid=eead2d61bcbd2ce1f827d441dfc16b82f2fa7798>

[Hogwild! Inference: Parallel LLM Generation via Concurrent Attention](https://www.arxiv.org/abs/2504.06261?_bhlid=eead2d61bcbd2ce1f827d441dfc16b82f2fa7798)

**초록(Abstract)**  
대형 언어 모델 (Large Language Models, **LLMs**)은 고급 추론 (advanced reasoning), 장문의 콘텐츠 생성 (long‑form content generation), 그리고 도구 활용 (tool use)을 통해 갈수록 복잡한 과제까지 해결할 수 있음을 보여 주었다. 이러한 과제를 풀려면 종종 긴 추론 시간이 필요한 계산이 뒤따른다. 사람의 문제 해결 과정에서 작업 속도를 높이는 흔한 전략은 **협업(collaboration)**이다. 문제를 하위 과제로 나누거나, 여러 전략을 동시에 탐색하는 식이다. 최근 연구에 따르면 LLM 역시 명시적인 협력 프레임워크—예컨대 **투표 메커니즘(voting mechanisms)** 또는 병렬로 실행 가능한 독립적 하위 과제의 명시적 생성—를 통해 병렬로 동작할 수 있다. 그러나 이러한 각 프레임워크가 모든 과제 유형에 적합한 것은 아니어서 적용성이 제한될 수 있다.

본 연구에서는 다른 설계를 제안한다. 우리는 여러 LLM **“작업자(worker)”**를 병렬로 실행하면서, 동시에 업데이트되는 **어텐션 캐시(attention cache)**를 통해 동기화하도록 하고, 각 작업자가 어떻게 협업할지 스스로 결정하도록 프롬프트한다. 이렇게 하면 각 인스턴스가 서로의 부분 진행 상황을 공유 캐시로 “보면서” 당면 과제에 맞는 협업 전략을 스스로 고안할 수 있다.

이를 위해 우리는 **Hogwild! Inference**를 구현했다. 이는 동일한 LLM 인스턴스 여러 개가 동일한 어텐션 캐시를 공유하며 병렬로 실행되고, 서로가 생성한 토큰에 **즉각적으로(“instant”)** 접근할 수 있는 병렬 LLM 추론 엔진이다.<sup>1</sup> Hogwild! Inference는 **회전 위치 임베딩(Rotary Position Embeddings, RoPE)**을 활용해 재계산을 피하면서 하드웨어 병렬 활용도를 높인다. 우리는 최신의 고급 추론 능력을 갖춘 LLM이 추가 미세 조정(fine‑tuning) 없이도 **공유 Key‑Value 캐시(shared Key‑Value cache)**로 추론을 수행할 수 있음을 확인했다.

<sup>1</sup> 구현 코드는 <https://github.com/eqimp/hogwild_llm>에서 확인할 수 있다.

† 교신 저자(Corresponding author): rodionovgleb@yandex‑team.ru  
\* 공동 1저자(Equal contribution)  
‡ 책임 저자(Senior author)

### 1 서론(Introduction)

![](/assets/images/posts/543/img.png)

**그림 1**: Hogwild! Inference의 직관적 설명. 두 개의 워커(worker)가 병렬로 텍스트를 생성하고, 세 개의 공유 캐시 블록(shared cache block)을 사용한다. 색상마다 하나의 캐시 블록을 나타낸다. 실제 동작 예시는 ‘example generation’에서 확인할 수 있다.

최근 대형 언어 모델(**Large Language Models, LLMs**) 연구에서는 추론(inference) 단계에서 추가적인 계산을 수행해 성능을 높이는 방법이 주목받고 있다 (Suzgun et al., 2022; Snell et al., 2024; Beeching et al.; Muennighoff et al., 2025). 이러한 방법은 **고급 추론(advanced reasoning)** (Wei et al., 2022; Kojima et al., 2022; Zhang et al., 2022; Yao et al., 2023; Lightman et al., 2023), **장문 생성(long‑form generation)** (Bai et al., 2024), **외부 도구 활용(tool use)** (Schick et al., 2023; Qin et al., 2023; Yao et al., 2022; Shen et al., 2023) 등에서 두각을 드러낸다. 실제 서비스에서도 장문‑추론 능력을 갖춘 모델이 등장하고 있으며(OpenAI et al., 2024; Google DeepMind, 2025; Anthropic, 2024), DeepSeek‑AI et al. (2025), Qwen Team (2025), Yang et al. (2024), Touvron et al. (2023), Dubey et al. (2024), Muennighoff et al. (2025), Ye et al. (2025) 등 공개 모델도 속속 등장하고 있다.

그러나 복잡한 문제를 해결하려면 토큰을 하나씩 생성하는 **순차적(sequential)** 계산이 길어지는 경우가 많다. 한편 많은 추론 문제는 본질적으로 순차적이지 않다. 이를 활용해 여러 연구는 **병렬 추론(parallel inference)** 전략을 제안했다. 예를 들어, 여러 LLM이 독립적으로 문제를 풀고 **투표**(Wang et al., 2022)하거나 서로의 출력을 교차 검증(cross‑reference)하여(Du et al., 2023; Wang et al., 2024a) 정확도를 높일 수 있다. 다른 접근은 LLM이 문제를 여러 **독립적 하위 과제(sub‑task)**로 분할한 뒤 이를 병렬로 해결하고 결과를 합치는 방식이다(Ning et al., 2024; Kim et al., 2024; Jin et al., 2025). 이러한 전략은 하드웨어 병렬성을 활용해 품질과 효율을 동시에 개선한다.

하지만 단일 협업 전략이 모든 문제에 효과적인 것은 아니다. 예컨대 독립적인 병렬 “스레드”로 문제를 풀면, 특정 스레드가 특히 오래 걸릴 때 나머지 에이전트는 **지연(straggler)**을 기다리느라 계산 자원을 낭비할 수 있다(Wang et al., 2022, 2024a). 반면 하위 과제 기반 추론은 문제를 곧바로 여러 과제로 쪼갤 수 있어야만 동작한다. 계획이 잘못됐음을 발견해도 재계획(re‑plan)을 할 수 없어 불필요한 과제를 계속 수행할 위험도 있다(Ning et al., 2024; Ding et al., 2025; Jin et al., 2025).

이는 인간 협업 방식과 대조적이다. 인간 문제 해결자는 고정된 전략을 따르기보다는 상황에 따라 계획을 수정하거나, 진행 중인 작업을 포기하고 더 유망한 접근으로 전환하며, 전략을 토론·논쟁하기도 한다(Hutchins, 1995; Entin and Serfaty, 1999). 이러한 **동적 협업(dynamic collaboration)**은 정의하기 어렵지만, 참여자가 충분히 유기적이라면 더 유연하고 효율적일 수 있다.

본 연구는 이 원칙을 **인공 추론기(artificial reasoner)**에도 적용해 본다. 현대 LLM은 이미 어느 정도 **계획 수립(planning)**과 **추론(reasoning)** 능력을 갖추었으므로(Zhou et al., 2024; Gao et al., 2024; Wang et al., 2024c), 여러 인스턴스 간 동적 상호작용이 도움이 될 것이라는 가설을 세웠다.

이를 검증하기 위해 **Hogwild! Inference**를 제안한다. 이는 **사전 정의된 협업 프레임워크 없이(no pre‑defined framework of collaboration)** 수행하는 병렬 LLM 추론 프로토콜이다.² 여러 LLM 인스턴스가 병렬로 토큰을 생성하면, 각 인스턴스는 다른 인스턴스가 방금 생성한 토큰을 **즉시** 확인할 수 있다. 그런 다음, 최신 상황을 반영해 각자 다음 행동(하위 과제 해결, 상호 검증, 전략 토론, 계획 전환 등)을 스스로 결정하도록 프롬프트한다.

이를 위해 Hogwild! Inference는 동일한 가중치(weights)를 공유하는 여러 LLM 인스턴스를 실행하되, **맞춤형 Key‑Value 캐시(custom Key‑Value cache)**를 사용해 토큰 표현을 공유하고 동시에 교차 어텐션(cross‑attention)을 가능하게 한다. 구체적으로는 각 워커의 KV 메모리를 추적한 뒤, **위치 임베딩(position embeddings)**을 보정해 워커마다 다른 순서로 **“이어 붙여(stitch)”** 사용한다(그림 1 참조).

최신 공개 모델인 **QwQ**(Qwen Team, 2025)와 **DeepSeek‑R1**(DeepSeek‑AI et al., 2025)으로 실험한 결과, 별도 미세 조정 없이도 “협업 조정에 대한 추론(reason about coordinating)”이 가능했다. 구체적으로, 병렬 에이전트들은 계획을 세우고, 초기 계획이 실패하면 적응하며, 서로의 오류를 지적하고, 핵심 관찰을 공유한다. 어느 워커가 이미 끝낸 하위 과제를 다른 워커가 중복 수행하거나, 계획 변경 후 불필요해진 문제를 풀고 있을 때 “중복 작업(redundant work)” 여부를 확인하도록 지시하면, 종종(항상은 아님) 이를 감지하고 다른 전략으로 전환했다.

우리는 이러한 관찰을 기반으로 세 가지 **메모리 레이아웃(memory layout)**을 시험했다.  
i) **단순 레이아웃(naive layout)**: 각 인스턴스가 연속적인 메모리 구역에 진행 상황을 기록.  
ii) **채팅형 레이아웃(chat‑like layout)**: 각 인스턴스가 비공개 버퍼에 작성하다가 주기적으로 공유 메모리에 커밋(commit).  
iii) **하이브리드(hybrid) 전략**: 공유 채팅형 히스토리를 사용하되, 메시지를 보내기 전 서로의 현재 메시지를 볼 수 있음.

우리는 긴 추론 체인이 필요한 수학 문제에 대해 Hogwild! Inference를 평가하여 레이아웃별 효과를 비교했다. 예비 실험 결과, 모든 캐시 설정에서 병렬 인스턴스는 **자신만의 추론 경로**를 유지하면서도 다른 인스턴스의 진행 상황을 동적으로 통합했다. 또한 문제 유형에 따라 **협업 행동(collaborative behavior)**을 스스로 조정하는 징후를 보였다.

이 초기 결과는 **공유 Key‑Value 캐시(shared Key‑Value cache)**를 활용한 병렬 추론이 여러 LLM 인스턴스 간 효과적인 협업을 가능케 하는 유망한 방법임을 시사한다.

---

² 우리 접근은 비동기적으로 업데이트를 적용하는 **Hogwild! SGD** (Recht et al., 2011)에서 영감을 받았다. 느낌표는 원래 이름의 일부다(Stanford HAI, 2023).

---
