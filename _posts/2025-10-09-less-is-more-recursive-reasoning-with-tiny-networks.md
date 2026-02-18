---
title: "Less is More: Recursive Reasoning with Tiny Networks"
date: 2025-10-09 22:14:39
categories:
  - 인공지능
---

<https://arxiv.org/abs/2510.04871?ref=refetch.io>

[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871?ref=refetch.io)

**초록(Abstract)**  
**Hierarchical Reasoning Model (HRM)**은 서로 다른 주기로 반복(recursing)하는 두 개의 작은 신경망을 사용하는 새로운 접근 방식이다. 생물학적 영감을 받은 이 방법은 **Sudoku**, **Maze**, **ARC-AGI**와 같은 난이도 높은 퍼즐 과제에서 **대규모 언어 모델(LLM)**을 능가하며, 단 **약 1,000개의 예시와 2,700만 개의 파라미터를 가진 작은 모델**로 학습되었다. HRM은 작은 네트워크로 어려운 문제를 해결할 수 있는 잠재력을 지니지만, 아직 충분히 이해되지 않았고 최적화되지 않았을 가능성도 있다. 이에 우리는 훨씬 단순한 재귀적(reasoning) 추론 접근법인 **Tiny Recursive Model (TRM)**을 제안한다. TRM은 단 **2개의 층(layer)**을 가진 하나의 초소형 네트워크만을 사용하면서도 HRM보다 훨씬 높은 **일반화 성능(generalization)**을 달성한다. 700만 개의 파라미터만으로 TRM은 **ARC-AGI-1에서 45%**, **ARC-AGI-2에서 8%의 테스트 정확도**를 기록하며, 이는 **DeepSeek R1**, **o3-mini**, **Gemini 2.5 Pro**와 같은 대부분의 LLM보다 높은 수치이다. 그럼에도 TRM은 이들 모델의 **0.01% 미만의 파라미터**만을 사용한다.

**키워드:** reasoning, recurrent, ARC-AGI

**1. 서론(Introduction)**

강력한 성능을 지닌 **대규모 언어 모델(Large Language Models, LLMs)**조차도 어려운 **질문-답변형 문제(hard question-answer problems)**에서는 종종 어려움을 겪는다. 이유는 간단하다 — LLM은 **자가회귀(auto-regressive)** 방식으로 출력을 생성하기 때문에, 단 **하나의 잘못된 토큰(token)**이 전체 답변을 무효화할 위험이 매우 크다.

이러한 신뢰성 문제를 개선하기 위해 LLM들은 주로 **사고의 연쇄(Chain-of-Thought, CoT)** (Wei et al., 2022)와 **시험 시 계산(Test-Time Compute, TTC)** (Snell et al., 2024)에 의존한다. CoT는 인간의 추론 과정을 모방하려는 시도로, 모델이 최종 답변을 내기 전에 **단계별(step-by-step)** 추론 과정을 샘플링하도록 유도한다. 이 방식은 정확도를 향상시킬 수 있으나, **비용이 크고**, **고품질의 추론 데이터**(이는 항상 존재하지 않는다)가 필요하며, 생성된 추론 과정이 잘못될 경우 **불안정(brittle)**하다는 한계가 있다.

이 신뢰성을 더 높이기 위해, 시험 시 계산(TTC)은 **K개의 출력 중 가장 빈도가 높은 답변**이나 **보상이 가장 높은 답변**을 선택하는 방법(Snell et al., 2024)을 사용하기도 한다.

![](/assets/images/posts/600/img.png)

**그림 1: Tiny Recursive Model (TRM)**은 작은 네트워크를 사용하여 예측된 답변 **y**를 **재귀적으로(recursively)** 개선하는 모델이다. 이 모델은 임베딩된 입력 질문 **x**, 초기 임베딩된 답변 **y**, 그리고 잠재 벡터(latent) **z**로부터 시작한다. 최대 **N\_sup = 16**번의 개선 단계(improvement steps)에 걸쳐, TRM은 자신의 답변 **y**를 점진적으로 향상시키려 시도한다. 그 과정은 다음 두 단계로 이루어진다.  
i) 질문 **x**, 현재 답변 **y**, 현재 잠재 벡터 **z**를 바탕으로 **잠재 벡터 z를 n번 재귀적으로 갱신**한다 (**재귀적 추론**, recursive reasoning).  
ii) 이후, 현재 답변 **y**와 잠재 벡터 **z**를 기반으로 **답변 y를 갱신**한다.

이러한 재귀적 절차를 통해 모델은 이전 답변의 오류를 점진적으로 수정하며, **매우 효율적인 파라미터 사용(parameter-efficient)**으로 **과적합(overfitting)**을 최소화하면서 답변의 품질을 꾸준히 개선할 수 있다.

하지만 이것만으로는 충분하지 않다. **Chain-of-Thought(CoT)**와 **Test-Time Compute(TTC)**를 사용하는 **LLM**이라도 모든 문제를 해결할 수 있는 것은 아니다. LLM은 2019년 이후 **ARC-AGI(Chollet, 2019)** 과제에서 상당한 발전을 이루었지만, **여전히 인간 수준의 정확도(human-level accuracy)**에는 도달하지 못했다(이 논문 작성 시점 기준으로 6년이 지난 현재까지도). 더 나아가, **ARC-AGI-2**와 같은 최신 버전에서는 성능이 더욱 저조하다. 예를 들어 **Gemini 2.5 Pro**는 매우 높은 수준의 TTC를 적용하고도 **테스트 정확도 4.9%**에 그친다(Chollet et al., 2025; ARC Prize Foundation, 2025b).

이에 대한 **대안적 접근 방향(alternative direction)**으로 **Wang et al.(2025)**은 새로운 방법을 제시했다. 그들은 **Hierarchical Reasoning Model (HRM)**이라는 새로운 구조를 통해, LLM이 고전하던 퍼즐형 과제들(Sudoku, Maze, ARC-AGI 등)에서 높은 정확도를 달성했다. HRM은 **지도학습(supervised learning)** 기반 모델이며, 두 가지 핵심 혁신을 가지고 있다:

1. **재귀적 계층 추론(recursive hierarchical reasoning)**
2. **깊은 감독(deep supervision)**

**재귀적 계층 추론(recursive hierarchical reasoning)**은 두 개의 작은 네트워크를 반복적으로 호출하여 답을 예측하는 방식이다.

- 고주파(high frequency)로 작동하는 **f\_L**,
- 저주파(low frequency)로 작동하는 **f\_H** 두 네트워크가 존재하며, 각각은 서로 다른 잠재 표현(latent feature)을 생성한다. 즉, **f\_L**은 **z\_H**를 출력하고, **f\_H**는 **z\_L**을 출력한다. 이 두 특징(**z\_L**, **z\_H**)은 서로의 입력으로 다시 사용된다. 저자들은 인간의 뇌가 서로 다른 시간적 주파수(temporal frequency)에서 작동하며 감각 정보를 계층적으로 처리한다는 생물학적 근거를 들어, 이런 다단계 재귀 구조를 정당화하였다.

**깊은 감독(Deep supervision)**은 여러 단계의 감독(supervision step)을 통해 점진적으로 답변을 개선하는 방식이다. 이 과정에서 두 개의 잠재 특징(**z\_L**, **z\_H**)은 다음 단계의 초기값(initialization)으로 전달되지만, **계산 그래프(computational graph)**로부터 분리(detach)되어 **gradient가 역전파되지 않도록** 한다. 이 구조는 **잔차 연결(residual connections)**을 만들어, 하나의 순전파(forward pass)에서 메모리 한계로 구현하기 어려운 **매우 깊은 신경망의 효과**를 모사한다.

한편, **ARC-AGI 벤치마크**에 대한 독립 분석 결과에 따르면, HRM의 성능 향상의 주요 요인은 재귀적 구조가 아니라 **깊은 감독(deep supervision)**이었다(ARC Prize Foundation, 2025a).

- 단일 단계 감독(single-step supervision)에서 **정확도 19% → 39%**로 **2배 향상**되었지만,
- 재귀적 계층 추론은 단일 순전파 모델 대비 **35.7% → 39.0%**로 **소폭 향상**에 그쳤다. 이 결과는 “여러 단계의 감독 간 reasoning은 유의미하지만, 각 단계 내부의 recursion 자체는 큰 영향을 주지 않는다”는 것을 시사한다.

이에 본 연구에서는 **재귀적 추론(recursive reasoning)**의 효과를 훨씬 더 극대화할 수 있음을 보인다. 우리는 **Tiny Recursive Model (TRM)**을 제안하는데, 이는 단 **2개의 층(layer)**으로 구성된 매우 작은 네트워크를 사용하면서도 HRM보다 **훨씬 높은 일반화 성능(generalization)**을 다양한 문제에서 달성한다. 그 결과, 우리는 다음과 같이 **최신 최고 성능(state-of-the-art)**을 크게 갱신하였다.

- **Sudoku-Extreme:** 55% → **87%**
- **Maze-Hard:** 75% → **85%**
- **ARC-AGI-1:** 40% → **45%**
- **ARC-AGI-2:** 5% → **8%**

**2. 배경(Background)**

**HRM(Hierarchical Reasoning Model)**의 구조는 **Algorithm 2**에 기술되어 있으며, 아래에서 그 세부 내용을 설명한다.

### **2.1 구조 및 목표 (Structure and Goal)**

HRM의 중심은 **지도 학습(supervised learning)**이다. 즉, 주어진 입력(input)에 대해 적절한 출력을 생성하는 것이 목표이다.

입력과 출력은 모두 [B, L] 형태를 가진다고 가정한다. (만약 길이가 다를 경우, **padding token**을 추가하여 길이를 맞춘다.)  
여기서

- **B**는 배치 크기(batch size),
- **L**은 컨텍스트 길이(context length)를 의미한다.

### **HRM의 학습 가능한 구성 요소 (Learnable Components)**

HRM은 네 가지 학습 가능한 모듈로 구성된다.

1. 입력 임베딩 (Input embedding): f\_I(·; θ\_I)
2. 저수준 순환 네트워크 (Low-level recurrent network): f\_L(·; θ\_L)
3. 고수준 순환 네트워크 (High-level recurrent network): f\_H(·; θ\_H)
4. 출력 헤드 (Output head): f\_O(·; θ\_O)

입력이 임베딩된 후에는 텐서의 형태가 [B, L, D]가 되며, 여기서 **D**는 임베딩 차원(embedding size)이다.

각 네트워크는 **4층 Transformer(Vaswani et al., 2017)** 구조를 사용하며, 다음과 같은 특징을 갖는다:

- **RMSNorm** (Zhang & Sennrich, 2019)
- **Bias 없음(no bias)** (Chowdhery et al., 2023)
- **Rotary embedding** (Su et al., 2024)
- **SwiGLU 활성 함수(activation function)** (Hendrycks & Gimpel, 2016; Shazeer, 2020)

**HRM의 의사코드(Pseudocode)**

```
def hrm(z, x, n=2, T=2):  # 계층적 추론(hierarchical reasoning)
    zH, zL = z
    with torch.no_grad():
        for i in range(nT - 2):
            zL = L_net(zL, zH, x)
            if (i + 1) % T == 0:
                zH = H_net(zH, zL)
    # 1-step gradient 업데이트
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    return (zH, zL), output_head(zH), Q_head(zH)

def ACT_halt(q, y_hat, y_true):
    target_halt = (y_hat == y_true)
    loss = 0.5 * binary_cross_entropy(q[0], target_halt)
    return loss

def ACT_continue(q, last_step):
    if last_step:
        target_continue = sigmoid(q[0])
    else:
        target_continue = sigmoid(max(q[0], q[1]))
    loss = 0.5 * binary_cross_entropy(q[1], target_continue)
    return loss

# 깊은 감독(Deep Supervision)
for x_input, y_true in train_dataloader:
    z = z_init
    for step in range(N_sup):  # deep supervision 단계
        x = input_embedding(x_input)
        z, y_pred, q = hrm(z, x)
        loss = softmax_cross_entropy(y_pred, y_true)

        # Q-learning 기반 적응적 계산 시간(Adaptive Computational Time, ACT)
        loss += ACT_halt(q, y_pred, y_true)

        _, _, q_next = hrm(z, x)  # 추가 forward pass
        loss += ACT_continue(q_next, step == N_sup - 1)

        z = z.detach()
        loss.backward()
        opt.step()
        opt.zero_grad()

        if q[0] > q[1]:  # 조기 종료(early stopping)
            break
```

**그림 2:** 계층적 추론 모델(Hierarchical Reasoning Models, HRMs)의 **의사코드(pseudocode)**.

---

![](/assets/images/posts/600/img_1.png)

![](/assets/images/posts/600/img_2.png)

![](/assets/images/posts/600/img_3.png)

---
