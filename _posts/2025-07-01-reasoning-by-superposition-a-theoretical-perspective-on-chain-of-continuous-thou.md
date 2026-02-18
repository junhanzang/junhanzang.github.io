---
title: "Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought"
date: 2025-07-01 21:38:41
categories:
  - 인공지능
---

<https://arxiv.org/abs/2505.12514>

[Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought

Large Language Models (LLMs) have demonstrated remarkable performance in many applications, including challenging reasoning problems via chain-of-thoughts (CoTs) techniques that generate ``thinking tokens'' before answering the questions. While existing th

arxiv.org](https://arxiv.org/abs/2505.12514)

**초록**  
대규모 언어 모델(Large Language Models, LLMs)은 다양한 응용 분야에서 뛰어난 성능을 보여주고 있으며, 특히 질문에 답하기 전에 '사고 토큰(thinking tokens)'을 생성하는 연쇄적 사고(chain-of-thought, CoT) 기법을 통해 어려운 추론 문제에서도 강력한 성능을 보이고 있다. 기존 이론 연구에서는 이산 토큰(discrete tokens)을 사용하는 CoT가 LLM의 추론 능력을 강화함을 보여주었으나, 최근 연속적 CoT(continuous CoT)에 대한 연구는 방향 그래프 도달성(directed graph reachability)과 같은 다양한 추론 과제에서 이산 방식보다 더 우수한 성능을 보임에도 불구하고, 이에 대한 이론적 설명은 부족한 실정이다.

본 논문에서는 연속 CoT를 사용하는 깊이 2의 트랜스포머가 그래프의 지름(diameter)을 D라 할 때, **D단계**의 연속 CoT만으로 방향 그래프 도달성 문제를 해결할 수 있음을 증명한다. 이는 현재까지 알려진 이산 CoT 기반의 고정 깊이 트랜스포머가 **O(n²)** 단계의 디코딩을 요구하는 것보다 훨씬 효율적이다 (여기서 n은 정점의 수, D < n). 우리의 구성에서 각 연속적 사고 벡터는 여러 탐색 경계를 동시에 인코딩하는 **중첩 상태(superposition state)**로 작동하여 병렬적인 너비 우선 탐색(BFS)을 가능하게 한다. 반면, 이산 CoT는 중첩 상태에서 단일 경로만을 샘플링하므로 순차적 탐색만 가능하며, 이는 더 많은 단계가 필요하고 지역 최적해에 빠질 위험이 있다.

우리는 이러한 이론적 구성과 학습을 통해 얻은 실험 결과가 잘 일치함을 실증적으로 확인하였다. 특히, **중첩 상태로서의 다중 탐색 경계 인코딩**은 명시적인 감독(supervision) 없이도 연속 CoT 학습 중에 자연스럽게 나타나는 현상임을 관찰하였다.

\* 공저자들의 기여는 동일함.

## 1 서론

대규모 언어 모델(Large Language Models, LLMs)은 특히 연쇄적 사고(chain-of-thought, CoT) 기법(Wei et al., 2022)을 적용했을 때, AIME나 수학 증명과 같은 어려운 문제를 포함한 다양한 추론 과제에서 우수한 성능을 보여주고 있다. 그러나 이러한 CoT를 활용하더라도, 보다 정교한 추론 능력을 요구하는 과제들—예컨대, 복잡한 규모의 추론 및 계획 문제(Zheng et al., 2024; Xie et al., 2024)—에서는 여전히 한계를 드러낸다(Kambhampati, 2024; Valmeekam et al., 2024; Zhou et al., 2025).

기존의 이산 CoT를 확장하여 더 복잡한 추론 문제를 해결하는 방법은 아직까지 명확히 해결되지 않은 연구 과제다. 최근 Hao et al. (2024)은 연속적인 잠재 사고(latent thought)를 활용하는 **Coconut**(chain-of-continuous-thought)을 제안하였고, 방향 그래프 도달성(directed graph reachability)과 같은 합성 과제뿐 아니라, GSM8K(Cobbe et al., 2021)와 같은 실제 수학 추론 벤치마크에서도 성능 향상을 보여주었다. 특히 Coconut은, 최종 해답에 도달하기 전에 여러 후보 탐색 경계를 잠재적으로 동시에 저장할 수 있음을 시사하는 초기 결과를 보인다. 이는 각 사고 토큰을 하나씩 샘플링하여 순차적으로 모델에 입력해야 하는 이산 CoT와는 뚜렷하게 대조된다. 그러나 연속적인 사고의 표현력과 작동 메커니즘에 대해서는 아직 명확한 이론적 이해가 부족하다.

본 연구에서는 Coconut의 메커니즘을 **그래프 도달성(graph reachability)** 문제를 통해 탐구한다. 이 문제는 방향 그래프 내에서 주어진 시작 노드와 도착 노드 사이에 경로가 존재하는지를 판단하는 과제이며, 매우 일반적인 형태의 문제로, 튜링 기계의 정지 문제와 같은 이론적 문제나 지식 그래프 등 실용적인 응용 사례들을 모두 포함한다(Ye et al., 2024; Hao et al., 2024; Zhou et al., 2025). 이러한 설정 하에서 우리는 그래프의 지름(두 노드 사이 최장 경로 길이)을 D라 할 때, **깊이 2의 트랜스포머**가 **D 단계의 연속 사고(continuous thought)**만으로도 **정점 수가 n인 그래프**에 대한 도달성 문제를 해결할 수 있음을 증명하였다(D < n). 반면, 이산 CoT를 사용하는 고정 깊이 트랜스포머의 경우에는 **O(n²)** 단계가 필요하다는 것이 현재까지의 최선 결과이다(Merrill and Sabharwal, 2023a).

직관적으로 말해, 우리의 구성에서는 각 잠재 사고 벡터가 여러 유효한 탐색 경로들의 **중첩(superposition)** 상태로 표현되므로, 매 자회귀(autoregressive) 단계마다 그래프 상에서 **암묵적 병렬 탐색**을 수행할 수 있다. 이러한 연속 사고는 양자역학에서의 중첩 상태(superposition state)처럼 작동하여, 다수의 탐색 경계를 동시에 저장하고 효율적인 **너비 우선 탐색(BFS)**을 가능케 한다(Böhm, 2013). 반면, 이산 사고 토큰은 중첩 상태에서 **붕괴된 상태(collapsed state)**로 볼 수 있으며, 모델이 탐색 분기를 하나만 선택하도록 강제한다. 이로 인해 잘못된 탐욕적 탐색이나, 되돌아가며 진행하는 깊이 우선 탐색 방식으로 흐를 수 있으며, 이는 더 많은 계산 비용을 초래한다. 또한 기존 이론적 접근은 주어진 문제나 입력 길이에 맞게 위치 인코딩을 맞춤 설계해야 했으나, 본 연구의 구성은 **사인파 위치 인코딩(sinusoidal positional encoding)**(Vaswani et al., 2017)이나 **회전 위치 임베딩(rotary position embedding, RoPE)**(Su et al., 2024)과 같은 실제 널리 쓰이는 위치 인코딩에도 적용 가능하다.

더 나아가, 우리는 이론적으로 설계한 구조가 실제 **기울기 기반 학습(gradient-based training)**에서도 잘 구현됨을 보였다. 구체적으로, **연속 CoT를 사용하는 깊이 2 트랜스포머**는 **이산 CoT를 사용하는 깊이 12 트랜스포머**보다 그래프 도달성 문제에서 더 뛰어난 성능을 보였다. 어텐션 패턴과 내부 표현을 분석한 결과, 연속 사고는 실제로 중첩 상태 내에서 다수의 가능성 있는 탐색 경계를 병렬로 인코딩하고 있다는 것이 확인되었다. 주목할 만한 점은, 이러한 **중첩 기반 표현**이 다른 탐색 경로들을 명시적으로 지도하지 않고도, **그래프 도달성의 최적 경로만을 학습한 상황에서 자발적으로 등장**한다는 것이다.

---

## ? 더 시각적으로, "탐색 방식" 비교

### 이산 CoT: 한 경로씩만 탐색

```
Start
  |
  v
 [A] --→ [B] --→ [C] --→ [D]
        (한 경로만 따라가며 생각함)
```

- 한 번에 하나의 경로만 따라감
- 실수하면 다시 돌아가야 함

깊이가 깊어질수록 느려짐
---

### 연속 CoT: 여러 경로를 동시에 고려

```
Start
  |
  v
 [A]
  | \
  v  v
[B] [C]
  |  |
  v  v
[D] [E]

(한 벡터 안에 여러 경로가 중첩돼 있음 → 병렬 BFS 느낌)
```

- 여러 경로를 동시에 '잠재적으로' 생각함
- 빠르게 도달성 판단 가능
- 적은 깊이로도 문제 해결 가능

## 결론

- **이산 CoT**: 깊이가 깊고, 생각을 한 토큰씩 생성 → 느림
- **연속 CoT**: 얕은 깊이에서도 병렬 탐색 가능 → 빠르고 효율적

> 그래서 이 논문에서는  
> **"깊이 2의 연속 CoT 트랜스포머가, 깊이 12의 이산 CoT 트랜스포머보다 성능이 더 좋다"**  
> 라는 결과가 나옵니다.
---

### 1.1 관련 연구

#### LLM의 텍스트 및 잠재 공간에서의 추론

대규모 언어 모델(LLM)의 추론 능력은 **연쇄적 사고(chain-of-thought, CoT)**(Wei et al., 2022)를 활용함으로써 크게 향상될 수 있다. CoT는 LLM이 최종 답변을 예측하기 전에 **중간 사고 과정(intermediate thoughts)**을 텍스트 형태로 명시적으로 출력하게 하여, 문제 해결력을 높인다. CoT 방식에는 프롬프트만 사용하는 방식(Khot et al., 2022; Zhou et al., 2022)과, 중간 사고가 포함된 샘플로 모델을 학습시키는 방식(Yue et al., 2023; Yu et al., 2023; Wang et al., 2023b; Shao et al., 2024)이 있다.

텍스트 기반 CoT 외에도, **중간 사고를 반드시 텍스트 토큰으로 나타낼 필요가 없는 잠재 공간(latent space)**에서의 추론에 대한 연구도 활발하다(Goyal et al., 2023; Wang et al., 2023c; Pfau et al., 2024; Su et al., 2025). 특히 Hao et al. (2024)은 LLM이 **연속적인 잠재 공간(continuous latent space)**에서 추론을 수행하도록 학습하는 방법을 제안하였으며, 이는 특히 **분기(branching) 수가 많은 그래프** 추론 과제에서 이산 CoT 방식보다 뛰어난 성능을 보였다. Hao et al. (2024)의 실증 사례 연구에 따르면, 연속 사고는 **여러 개의 유효한 탐색 경계를 동시에 인코딩**할 수 있다고 가정된다.

본 연구에서는 이러한 **연속 사고의 작동 메커니즘**을 이론적으로 분석하며, **연속 사고를 사용하는 트랜스포머는 추론 과정에서 중첩 상태(superposition states)**의 이점을 활용할 수 있음을 증명한다.

#### 트랜스포머의 표현력

트랜스포머의 표현력(expressivity)을 분석하는 연구는 오랫동안 이어져 왔다(Yun et al., 2019; Bhattamishra et al., 2020a, b; Pérez et al., 2021; Likhosherstov et al., 2021; Yao et al., 2021; Edelman et al., 2022; Akyürek et al., 2022; Merrill and Sabharwal, 2023b). 최근에는 **CoT를 통해 트랜스포머의 표현력을 향상시킬 수 있음**을 보여주는 연구들도 등장하고 있다(Liu et al., 2022; Feng et al., 2023; Merrill and Sabharwal, 2023a; Li et al., 2024).

예를 들어, Liu et al. (2022)은 **반자동기계(semi-automata)**를 대상으로 **얕은 깊이의 트랜스포머가 CoT 한 단계만으로도 어떤 표현력을 가질 수 있는지**를 분석했다. Feng et al. (2023)은 CoT가 포함된 **고정 깊이 트랜스포머**가 특정 **P-complete 문제**를 해결할 수 있음을 보였고, Li et al. (2024)은 **P/poly 클래스의 문제들**에 대해 CoT를 활용한 **고정 깊이 트랜스포머 구성법**을 제시하였다.

또한 Merrill and Sabharwal (2023a)은 CoT의 길이에 따른 표현력 차이를 분석하였고, **입력 길이의 로그만큼의 CoT 단계**가 주어질 경우, 고정 깊이 트랜스포머의 표현력 한계가 **TC₀에서 L(로그 공간)**로 확장되며, **선형 단계 수**를 허용하면 **NC₁-complete 수준까지 도달**할 수 있음을 밝혔다.

하지만 이러한 표현력 연구는 대부분 **이산 CoT**를 기반으로 하고 있으며, **연속 CoT**에 대한 이론적 연구(Hao et al., 2024)는 매우 드물다. 본 연구는 바로 이 **연속 CoT의 이론적 분석**에 중점을 둔다. 또한 기존 연구들은 문제 또는 입력 길이에 특화된 위치 인코딩(positional encoding)을 사용해야 했으나, 본 연구는 실제로 널리 사용되는 **사인파 위치 인코딩(sinusoidal)**(Vaswani et al., 2017)이나 **RoPE(rotary position embedding)**(Su et al., 2024)와 같이 일반적인 인코딩 방식에도 적용 가능하다는 장점이 있다.

#### 그래프 기반의 추론 문제

많은 추론 문제는 **계산 그래프(computational graph)** 형태로 추상화될 수 있기 때문에, **LLM의 추론 능력을 평가하는 데 그래프 문제는 매우 중요**하다(Ye et al., 2024; Zhou et al., 2025). 관계형 데이터를 입력받는 트랜스포머 모델은 이를 **그래프의 간선(edge)** 형태로 모델링할 수 있으며(Wang et al., 2024a, b; Guo et al., 2025), 여러 연구들은 사전학습된 LLM이 이러한 그래프 기반의 추론 문제를 어느 정도 처리할 수 있음을 보여주었다. 그러나 여전히 복잡한 그래프 문제에서는 성능이 제한적이라는 보고도 있다(Wang et al., 2023a; Guo et al., 2023; Fatemi et al., 2023; Sanford et al., 2024; Luo et al., 2024; Dai et al., 2024; Cohen et al., 2025).

한편, 트랜스포머가 **그래프 도달성(reachability)**이나 **최단 경로(shortest path)**와 같은 **고전적이며 이론적으로 중요한 문제**를 어떻게 해결하는지를 분석한 연구도 있다. 예를 들어 Cohen et al. (2025)은 **이층 트랜스포머(two-layer transformer)**가 **line graph의 스펙트럼 분해(spectral decomposition)**를 활용해 소규모 무방향 그래프의 최단 경로를 예측할 수 있음을 보였고, Merrill and Sabharwal (2025)은 **로그 깊이 트랜스포머(log-depth transformer)**가 **방향 그래프 도달성 문제**를 해결할 수 있으나, **고정 깊이 트랜스포머로는 불가능**하다고 밝혔다.

특히 Merrill and Sabharwal (2023a)은 고정 깊이 트랜스포머가 방향 그래프 도달성 문제를 해결하려면 **O(n²)** 단계의 CoT가 필요하다고 제시하였으며, 여기서 n은 정점 수다. 반면, **더 적은 수의 이산 CoT 단계만으로 이 문제를 해결할 수 있는지는 여전히 불명확**하다.

이에 반해, 본 연구는 **지름이 D인 그래프에 대해, 연속 사고 D단계만으로 도달성 문제를 해결할 수 있는 2층 트랜스포머**가 존재함을 보인다.
---

## ? 먼저, 트랜스포머가 풀 수 있는 "문제 난이도" 구분하기

### 계산 복잡도 클래스란?

- 우리가 어떤 문제를 풀 때, **얼마나 빠르게 계산할 수 있느냐**에 따라 문제를 분류해 놓은 거예요.
- 마치 게임 난이도 Easy / Medium / Hard처럼 생각하셔도 좋아요.
- 여기서 나오는 용어들은 그 난이도 구분을 의미합니다.

## ? 주요 용어 요약 (직관 위주)

용어 뜻 비유

|  |  |  |
| --- | --- | --- |
| **TC₀** | 아주 쉬운 문제들 (논리 회로로 계산 가능) | 계산기 버튼 몇 개 누르면 되는 문제 |
| **L (Logspace)** | 메모리를 거의 안 쓰고 푸는 문제들 | 손가락으로 세면서 풀 수 있는 수준 |
| **NC₁** | 아주 빠르게 병렬처리로 풀 수 있는 문제들 | 친구 여러 명이 동시에 계산해도 되는 문제 |
| **P** | 일반적인 다항시간 문제 (대부분 알고리즘 문제 포함) | 현실적인 시간 안에 푸는 문제들 |
| **P/poly** | 입력 길이에 따라 "맞춤형 회로"가 있다면 풀 수 있는 문제들 | 문제마다 공장 맞춤형 기계로 처리 가능 |

## ? 그럼 이 논문은 뭘 말하는 걸까?

- 기존 연구(Merrill and Sabharwal, 2023a)는 이산 CoT를 이용한 트랜스포머가:
  - \*\*"입력 길이의 log만큼의 CoT step"\*\*을 가지면, **TC₀ → L**까지 표현력이 늘어난다.
  - \*\*"입력 길이만큼 선형 CoT step"\*\*을 가지면, **NC₁ 수준 문제까지도 해결** 가능하다고 했어요.

? 즉, **이산 CoT의 step 수를 늘리면 더 어려운 문제까지 풀 수 있게 된다**는 뜻입니다.

## ? 이 논문이 새로 주장하는 건?

- **이산 CoT는 단계(step)를 많이 써야 복잡한 문제를 풀 수 있지만**,
- **연속 CoT는 같은 문제를 훨씬 적은 step 수로 풀 수 있다**는 걸 보인 겁니다.

즉,

> "NC₁ 수준 문제를 풀기 위해 이산 CoT는 수십 단계가 필요하지만,  
> 연속 CoT는 2단계 트랜스포머로도 해결 가능하다"

는 식의 \*\*"효율성 차이"\*\*를 이론적으로 설명하려는 거예요.

## ✅ 정리: 지금까지 내용을 간단히 요약하면

질문 답변

|  |  |
| --- | --- |
| TC₀, L, NC₁ 같은 건 뭔가요? | 문제의 계산 난이도를 나누는 복잡도 클래스입니다. |
| 왜 등장하나요? | 트랜스포머가 **얼마나 어려운 문제까지 풀 수 있나**를 구분하려고요. |
| 기존 이산 CoT는? | 단계 수를 늘려야 더 어려운 문제를 풀 수 있음 |
| 연속 CoT는? | **단계 수가 적어도** 여러 경로를 병렬로 표현해 성능이 더 좋을 수 있음 |
| 이 논문의 주장은? | 연속 CoT가 이산 CoT보다 추론 효율이 높을 수 있다는 **이론적 근거**를 제시함 |
---

## 2. 사전 지식 (Preliminaries)

![](/assets/images/posts/576/img.png)

![](/assets/images/posts/576/img_1.png)

### 토큰과 임베딩 (Tokens and embeddings)

![](/assets/images/posts/576/img_2.png)

### 트랜스포머 알고리즘 정의

![](/assets/images/posts/576/img_3.png)

![](/assets/images/posts/576/img_4.png)

### 알고리즘 2: 어텐션과 MLP 정의

![](/assets/images/posts/576/img_5.png)

### 위치 인코딩 (Positional Encoding)

![](/assets/images/posts/576/img_6.png)

### 정의 1: 사인파 기반 위치 인코딩 (Vaswani et al., 2017)

![](/assets/images/posts/576/img_7.png)

**주석 1 (Remark):**  
RoPE(Su et al., 2024) 기반의 위치 인코딩을 사용하는 이론 구성도 부록 B.6에서 다룹니다.

## 3. 문제 정의 (Problem Formulations)

### 그래프 도달성 (Graph reachability)

![](/assets/images/posts/576/img_8.png)

### 입력 구조 (Input structures)

![](/assets/images/posts/576/img_9.png)

### 연속 사고 체인 (Chain of continuous thought)

![](/assets/images/posts/576/img_10.png)

### 위치 인덱스 정의 (Position index)

![](/assets/images/posts/576/img_11.png)

### ? 이 장의 요점:

- 이 장에서는 그래프 도달성 문제를 **토큰 시퀀스 형태로 포맷팅**하고,
- 트랜스포머가 **연속 사고 벡터**를 자회귀 방식으로 생성하며 추론을 진행하는 구조를 설명합니다.
- \*\*다음 장(Section 4)\*\*에서는 이 구조가 **이론적으로 문제를 해결 가능함**을,
- **Section 5**에서는 **실험적으로도 잘 작동함**을 보여줍니다.

## 4. 이론적 결과 (Theoretical Results)

이 장에서는 **연속적 사고(continuous thought)**를 사용하는 **깊이 2의 트랜스포머**가 그래프 도달성 문제를 효율적으로 해결할 수 있음을 이론적으로 증명합니다. 우선 **4.1절**에서는 트랜스포머 구성에서 사용되는 핵심 구성 요소인 **attention chooser**를 소개합니다. 이후 **4.2절**에서는 연속 사고가 **여러 탐색 경로를 동시에 중첩 상태(superposition state)**로 유지한다는 핵심 결과를 제시하며, **4.3절**에서는 본 논문의 **주요 정리(main theorem)**를 증명합니다. 마지막으로 **4.4절**에서는 그에 대한 추가 논의를 진행합니다.

### 4.1 어텐션 선택자 (Attention chooser)

우리는 이론 구성에서 **attention chooser**를 하나의 빌딩 블록으로 사용합니다. 이 구성 요소는 현재 위치의 토큰에 따라 **어떤 위치를 어텐션으로 참조할지 선택하는 메커니즘**입니다. 이를 통해 입력 길이가 달라지더라도 **같은 파라미터 구성(parameter construction)**을 유지할 수 있게 됩니다. 즉, 입력 시퀀스가 길어지거나 짧아져도 모델 구조를 변경하지 않고 동작할 수 있게 해주는 역할을 합니다. 이 구성에 대한 자세한 증명은 **부록 B.1절**로 미루어집니다.

### 보조정리 1 (어텐션 선택자, 보조정리 3의 비공식 버전)

![](/assets/images/posts/576/img_12.png)

### 증명 개요 (Proof sketch)

![](/assets/images/posts/576/img_13.png)

![](/assets/images/posts/576/img_14.png)

### 4.2 연속 사고는 중첩 상태(superposition states)를 유지한다

![](/assets/images/posts/576/img_15.png)

![](/assets/images/posts/576/img_16.png)

![](/assets/images/posts/576/img_17.png)

![](/assets/images/posts/576/img_18.png)

![](/assets/images/posts/576/img_19.png)

![](/assets/images/posts/576/img_20.png)

![](/assets/images/posts/576/img_21.png)

![](/assets/images/posts/576/img_22.png)

### 4.3 예측으로서의 중첩 상태 측정

![](/assets/images/posts/576/img_23.png)

### 정리 1 (연속 사고 체인은 그래프 도달성 문제를 해결할 수 있다)

![](/assets/images/posts/576/img_24.png)

요약하면, 이 정리는 다음을 말합니다:

- 연속 사고는 도달 가능한 정점들을 중첩 상태로 유지하고,
- 최종 <A> 토큰은 이 중첩 상태를 "측정"해서,
- 더 강한 신호를 가지는 후보 노드를 정답으로 선택할 수 있으며,
- 이 모든 과정을 **그래프 구조에 독립적인 2층 트랜스포머**로 구현할 수 있다는 것입니다.

### 4.4 논의 (Discussions)

#### 버퍼 공간의 역할

버퍼 공간(buffer spaces)은 임베딩 내에서 **유용한 정보를 저장하기 위한 부분 공간(subspace)**이다.  
본 논문에서는 개념의 명확성을 위해, 임베딩을 **‘content’**, **두 개의 ‘buffer’ 공간**으로 구분하여 **서로 다른 차원에 분리**하여 구성하였다.

![](/assets/images/posts/576/img_25.png)

#### 중첩 상태에서 노드 간 가중치

이 논문에서 제안한 이론 구성에서는 **각 중첩 상태(superposition state)**가 **정점들을 균등한 가중치**로 유지한다.  
그러나 실제 모델에서는 다음과 같은 이유로 인해 **정점 간 가중치가 달라질 수 있다**:

- 학습 신호의 영향
- 모델이 어떤 정점이 최종 정답에 도달할 가능성이 높은지를 **내부적으로 추론하는 방식**

실제로 Section 5에서 보여주는 바와 같이,  
**학습 신호(training signal)**는 중첩 상태를 다음 방향으로 편향시킬 수 있다:

- 정확히 iii단계 내에 도달 가능한 **경계 정점(frontier nodes)**
- 도착 노드로 가는 **최적 경로에 있는 정점들(optimal nodes)**

이는 학습 과정에서 중첩 상태가 단순한 균등 평균 이상으로, **모델이 정답 도출에 유용하다고 판단한 정점들에 더 많은 비중을 두는 방향으로 진화**함을 의미한다.

## 5. 실험

본 절에서는 이론적 결과를 뒷받침하기 위한 다양한 실험을 수행하였다. 특히 Coconut이 적은 레이어 수만으로도 discrete CoT보다 뛰어난 성능을 보인다는 점(5.2절), 그리고 이 성능 차이가 추론 과정에서 연속적인 생각(continuous thoughts)이 **중첩 상태(superposition states)** 를 표현하기 때문이라는 점(5.3절)을 실증적으로 확인하였다.

![](/assets/images/posts/576/img_26.png)

Figure 4: Coconut, CoT, CoT\* (12 레이어, 헤드 수 = 12), No CoT의 전체 정확도 비교.

### 5.1 훈련 설정

**모델:**  
GPT-2 스타일의 디코더 아키텍처를 채택하였으며, Transformer 레이어 수는 2개, hidden dimension은 768, attention head 수는 8이다. 초기부터 scratch로 학습하였으며, AdamW 옵티마이저를 사용하였다 (β₁ = 0.9, β₂ = 0.95, weight decay = 10⁻²), learning rate는 고정값 1×10⁻⁴이다.

**데이터셋:**  
ProsQA (Hao et al., 2024)에서 reasoning hop이 3~4회 필요한 문제들을 추출하여 서브셋을 구성하였다. 그래프 내 각 노드는 고유한 토큰으로 vocabulary에 삽입하였다. 데이터셋의 분할 통계는 Table 4에 요약되어 있다.

**학습 방식:**  
Hao et al. (2024)의 방법을 따라, **chain-of-thought 데이터의 지도학습을 활용한 다단계 학습 전략**을 사용하였다.

- 학습의 **stage i**에서는 모델이 연속적인 생각(continuous thought)을 **i개 사용**한 후, chain-of-thought 상에서 **i번째 노드**를 다음 토큰으로 예측하도록 학습한다.
- 만약 학습 단계 인덱스가 주어진 CoT 해의 길이 **l**보다 크다면, 모델은 l개의 continuous thought 이후 <A> 토큰을 통해 최종 정답을 출력하도록 훈련된다.

모델은 각 스테이지마다 25 에폭 동안 학습되며, 전체 학습은 총 **300 에폭** 동안 진행된다. 각 단계에서 이전 단계의 데이터를 **0.1 확률로 무작위로 섞어** 사용하는데, 이는 초기 실험에서 성능 향상에 효과가 있는 것으로 확인되었다.

## 5.2 전체 결과

Hao et al. (2024)의 연구를 확장하여, 우리는 **Coconut**을 사용할 경우 **2개의 레이어만 가진 트랜스포머**도 ProsQA 문제를 효과적으로 해결할 수 있음을 추가적으로 보여준다. Figure 4에서 볼 수 있듯이, Coconut은 **거의 완벽한 정확도**를 달성한 반면, CoT와 No CoT 베이스라인은 각각 **약 75%**만을 해결하였다 (무작위 추정은 50%). 비록 CoT를 사용하는 모델의 크기를 12 레이어, 헤드 수 12로 크게 늘린 경우(\*로 표시됨) 정확도는 **83%까지** 증가하였지만, 여전히 안정적으로 문제를 해결하지는 못한다.

## 5.3 잠재 추론 과정 시각화

이론적 구조가 실제 학습된 모델에 구현되었는지를 확인하기 위해, 우리는 **attention 패턴**과 **연속적인 생각(continuous thought)**의 표현을 분석하였다.

### Layer 1의 attention

이론적 구조에 따르면, **Layer 1의 attention head의 가장 중요한 역할은** 각 엣지 토큰 <e>에 해당 엣지의 source 노드와 target 노드 정보를 복사하는 것이다. Figure 5는 대표적인 attention map을 보여주며, 모델이 실제로 이 복사 메커니즘을 구현했음을 확인시켜준다.

### Layer 2의 attention

Layer 2는 **노드 확장 역할**을 수행한다. 즉, 현재 도달 가능한 노드로부터 나가는 모든 엣지들에 attention을 집중한다.  
이를 정량적으로 확인하기 위해, 모델이 i번째 continuous thought를 생성할 때 각 엣지 토큰 (s, t, <e>) 삼중쌍이 받은 attention 점수를 집계하였다.

총 4가지 엣지 종류가 있다:

1. **Reachable:** source 노드가 현재 reachable set에 포함된 경우
2. **Not Reachable:** source 노드가 reachable set에 없는 경우
3. **Frontier:** reachable 중에서도 현재 탐색 경계에 위치한 노드 (루트로부터 정확히 i 단계 떨어짐)
4. **Optimal:** frontier 중에서 최적 reasoning chain으로 이어지는 엣지

![](/assets/images/posts/576/img_27.png)

**Figure 5:** Layer 1 attention map의 예시. y축의 <e> 토큰은 x축의 source와 target 노드에 대부분의 attention을 집중하고 있다. 이는 이론 구조와 일치한다.

Table 1은 위 그룹별 평균과 표준편차를 테스트셋 기준으로 보고한다. 결과적으로 모델은 이론 구조에 맞게 **Reachable 엣지에 집중된 attention**을 보여준다. 흥미롭게도, 추가적으로 **Frontier 엣지에 대한 편향**이 존재한다. 이는 학습 신호가 각 스텝에서 frontier 노드를 예측하도록 유도했기 때문일 수 있으며, 과거 노드에 대한 중요도가 점점 낮아지는 효과(decaying effect)도 포함될 수 있다. 또한, **Optimal 엣지**가 더 많은 attention을 받는 경향도 확인되는데, 이는 CoT 학습 과정에서 **정답 경로에 대한 감독(supervision)**이 주어졌기 때문으로 보인다.

### 연속적인 생각(continuous thought)의 표현

각 continuous thought가 탐색을 위한 **중첩 상태(superposition state)** 역할을 하는지 확인하기 위해, i번째 continuous thought [tᵢ]와 각 노드 임베딩 uᵥ 간의 내적을 계산하였다. 노드들은 앞서와 마찬가지로 Reachable / Not Reachable / Frontier / Optimal로 분류된다.

![](/assets/images/posts/576/img_28.png)

**Figure 6 (상단):** reasoning step i별로 그룹별 내적 분포를 히스토그램으로 나타낸 것.

**Figure 6 (하단):** i번째 continuous thought와 노드 임베딩 간의 내적 히스토그램. 범례에 각 그룹의 평균값이 표시됨. Frontier는 Reachable의 부분집합, Optimal은 Frontier의 부분집합이다.

예측대로, i hop 이내의 노드들은 더 높은 유사도를 보였다. 특히 **Frontier 노드가 [tᵢ]와 가장 유사한 경향**을 보이며, 이는 중첩 상태가 다음 탐색 대상에 집중하고 있음을 시사한다. **Optimal 노드**는 Frontier보다도 더 유사도가 높았으며, 이는 학습 과정에서 항상 최적 경로를 제공한 결과로 해석된다.

종합하면, 이 분석은 **이론에서 제시한 중첩 기반 탐색 구조가 실제 학습된 모델에 구현되어 있음**을 확인해준다:

- Layer 1은 탐색 컨텍스트를 설정하고,
- Layer 2는 탐색 경계를 확장하며,
- latent 벡터는 도달 가능한 상태 집합을 **soft하고 병렬적으로 표현**한다.

또한, Appendix C.2에서는 **다양한 random seed에서도 동일한 탐색 패턴이 재현**됨을 추가적으로 보여준다.

![](/assets/images/posts/576/img_29.png)

※ **설명**:

- Not Reachable: 해당 step에서 source 노드가 도달 불가능한 엣지
- Reachable: 도달 가능한 엣지 전체
- Frontier: Reachable 중에서도 현재 탐색 경계에 있는 엣지
- Optimal: Frontier 중에서도 최적 경로에 포함된 엣지

이 표는 모델이 실제로 Reachable 및 Optimal 엣지에 더 높은 어텐션을 집중하고 있음을 보여주며, 이는 논문 4장의 이론과 일치합니다.

**5.4 탐색 우선순위 (Exploration Priority)**  
5.3절의 시각화 결과에서 흥미로운 사실은, 모델이 **최적 엣지 및 노드에 불균형적으로 높은 어텐션을 할당**한다는 점이며, 이는 **우선순위 기반 탐색 전략**을 연상케 합니다. 이러한 행동은 각 단계에서 CoT 해답을 명시적으로 지도하는 **다단계 커리큘럼 학습의 산물**일 수 있다는 가설이 제기됩니다. 이 다단계 지도 방식의 효과를 분석하기 위해, 우리는 대안 감독 방식인 **Coconut-BFS**를 제안합니다. 이 방식에서는 훈련의 i-번째 단계에서 CoT 해답의 i-번째 노드 대신, 루트에서 정확히 i 단계 떨어진 **프론티어 노드 중 무작위로 하나**를 정답 토큰으로 지정합니다. 그 외 모든 하이퍼파라미터는 원래 Coconut과 동일하게 유지됩니다.

실험 결과, **Coconut-BFS 역시 ProsQA에서 거의 완벽한 정확도를 달성**하며, 원래의 Coconut과 동등한 성능을 보였습니다. Figure 6은 두 모델의 연속적인 latent thought와 각 노드 임베딩 간 내적 분포를 비교한 것입니다. 놀랍게도, **최적 노드에 대한 명시적 유도가 없음에도** 불구하고, 두 감독 방식은 유사한 탐색 전략으로 수렴했습니다. 반면, 원래의 Coconut은 학습 중간 단계에서 **오직 최적 노드만을 지도받았음에도**, 여전히 **비최적 프론티어 노드들에 더 높은 가중치를 부여**하며, 실제로는 해답을 찾기 전 **너비 우선 탐색(BFS) 확장을 수행하고 있음**을 보여줍니다. 이와 같은 행동의 원인을 **학습 동역학 관점에서 이론적으로 설명**하는 것은 향후 과제로 남깁니다.

**6 결론 (Conclusions)**  
본 논문에서는 **그래프 도달 가능성(graph reachability) 문제**를 중심으로, 연속적 사고 연쇄(chain-of-continuous-thought, Coconut)가 대형 언어 모델(LLM)의 추론 능력을 어떻게 향상시키는지를 연구했습니다. 우리는 **정확히 D번의 연속적 사고 단계만으로**, 꼭짓점 수 n, 지름 D인 방향성 그래프에서 도달 가능성 문제를 효율적으로 해결하는 **2-layer 트랜스포머의 이론적 구성**을 제시했습니다. 이는 기존의 이산 CoT 기반 고정 깊이 트랜스포머가 **O(n²)** 단계가 필요한 것에 비해 현저히 효율적인 결과입니다.

우리의 구성은, **다수의 탐색 경로를 동시에 인코딩하는 중첩(superposition) 상태**가 Coconut의 강력한 추론 능력의 핵심임을 밝혔으며, 학습을 통해 얻어진 모델이 실제로 이 이론 구조를 따르고 있음을 다양한 실험을 통해 검증했습니다.

앞으로의 흥미로운 연구 방향은 다음과 같습니다:

1. **그래프 도달 문제에서 이산 CoT가 필요한 최소 단계 수 하한선**을 도출함으로써, CoT와 Coconut 간 표현력의 명확한 구분 제시
2. **결정적 탐색 경로만 제공되었음에도**, 학습 도중 자발적으로 나타나는 탐색 행동의 이론적 이해
3. **보다 일반적인 문제 설정에서**, 연속 공간 상의 추론이 가지는 장점 탐구
