---
title: "Chapter 2 Reinforcement Learning Overview"
date: 2023-05-21 00:08:53
categories:
  - 강화학습
tags:
  - Reinforcement Learning
---

Reinforcement Learning (RL)은 불확실성 하에서의 의사 결정 및 경험을 통해 학습을 모델링하는 기계 학습 유형이라고 정의할 수 있다.

강화 학습이 다른 기계 학습 패러다임과 다른 점은 다음과 같다.

• 그라운드 트루스가 없고 reward 신호만 있음  
• 피드백이 즉각적이지 않고 지연됨  
• 시간이 정말 중요함(순차적이고 독립적이지 않고 동일하게 분산된 데이터).  
• 에이전트의 작업은 수신하는 후속 데이터에 영향을 미침

그라운드 트루스가 없다는게, 나는 가장 큰 장점이라고 생각한다. 왜냐면 어떤 상황에서도 학습이 가능하기 때문이다. 예를 들어 일반적인 NN은 로봇보고 알아서 걸으라고 하면 걸을까? 절대 못한다. 왜냐면 정답을 모르니까. 하지만 강화학습은 이걸 반복시켜서 학습하기 때문에 시간이 무한정있다면 학습이 가능하다.

그래서 agent라고 명명되는 NN(설명하기 쉽게 NN이라고 하자)은 다음 그림처럼 environment에 어떤 행동을하고 경험을 학습하게 된다.

![](/assets/images/posts/76/img.gif)

어떤 행동을 우리는 action a라 말하고 경험을 state s, reward r로 정의한다.

따라서 action은 미래의 reward와 state에 영향을 미친다.

**reward**는 다음과 같은 특성을 가진다.

• reward r은 스칼라 피드백 신호이다.  
• 에이전트가 ?단계에서 얼마나 잘하고 있는지 나타낸다  
• 에이전트의 임무는 누적 보상을 최대화하는 것이다

reward에 대한 예시를 들어보면 다음과 같다. 알파고를 학습시킨다고 예를 들면 이기면 긍정적인 보상 r이 돌아오고, 지면 부정적 보상 r이 돌아온다. ex) +1, -1

이를 극대화하고자 순차적 의사 결정(Sequential Decision Making)에 대해서 알아보자.

이는 미래의 총 보상을 극대화하기 위해 행동을 선택하는 과정을 말한다.   
• 목표: 미래의 총 보상을 극대화하기 위해 행동을 선택  
• 행동의 결과는 장기적으로 돌아올 수 있다  
• 보상은 지연될 수 있다  
• 더 많은 장기적 보상을 얻기 위해 즉각적인 보상을 포기하는 것이 더 나을 수 있다  
• 예시:   
          금융 투자 (성숙까지 몇 개월이 걸릴 수 있음)   
          헬리콥터에 연료 공급 (몇 시간 후에 사고 예방할 수 있음)   
          상대방의 움직임 차단 (멀리 떨어진 시점에서 승리 기회를 높일 수 있음)   
순차적 의사 결정은 다양한 분야에서 적용된다. 예를 들어 금융, 게임 이론, 로봇 공학, 자율 주행 등의 영역에서 많이 사용된다. 이는 단일 결정이 아닌 연속적인 행동 선택을 통해 최적의 결과를 달성하기 위한 방법론이다.

**Agent와 Environment**에 대해서 알아보자.

Step t에 agent는 observation ??, scalar reward ??를 받는다. 그리고 action ?? 실행한다.

그러면 해당 Environment는 action ??를 받고 observation ??, scalar reward ??를 agent에게 전달한다.

그리고 t는 Environment에서 1 step 또는 정해진 시간만큼 증가한다.

reward를 극대화하기 위해서는 이전 단계들의 observation, scalar reward, action에 대해서 알면 좋다.

따라서 다음과 같은 형식으로 저장하여 사용한다.

![](/assets/images/posts/76/img.png)

이 저장된 정보를 사용하면 좋은 건 당연히 안다. 그렇다면 어디에 사용할까? 다음 state를 예측하는데 사용하면 좋을 것이다. 내가 이런 행동을 했고, 이전에 어떤 state와 비슷하기 때문에 state 형태는 이럴것이다라고 생각하는 것처럼 말이다.

![](/assets/images/posts/76/img_1.png)

환경 상태 S는 환경의 개인 표현이며, 일반적으로 상담원에게 보이지 않으며 보이는 경우에도 관련 없는 정보를 포함할 수 있다

에이전트 상태S는 에이전트의 내부 표현이며, 에이전트가 다음 작업을 선택하는 데 사용하는 정보이며 강화 학습 알고리즘에서 사용하는 정보이다.

**Markov State**

Markov State는 기록의 모든 유용한 정보가 포함되어 state ??는 Markov인 경우에 다음의 수식을 사용할 수 있다.

![](/assets/images/posts/76/img_2.png)

하지만 미래는 주어진 현재와 과거로부터 독립적이며, state는 미래에 대한 충분한 통계이다.

**Fully Observable Environments**

에이전트가 환경 상태를 직접 관찰하여 에이전트 상태 == 환경 상태인 것을 말한다. 공식적으로 이것은 Markov Decision Process(MDP)로 불린다.

**Partially Observable Environments**

에이전트가 간접적으로 관찰하며, 카메라 비전을 가진 로봇은 절대 위치를 알려주지 않는다. 이제 에이전트 상태는 환경 상태와 같지 않다. 따라서 이것은 부분적으로 관찰 가능한 마르코프 결정 프로세스(POMDP)입니다. 에이전트는 자체 상태 표현을 구성해야 한다.

**RL 에이전트의 주요 구성 요소**  
RL 에이전트에는 다음 구성 요소 중 하나 이상이 포함될어야 한다.  
• Policy : 에이전트의 행동 기능  
• Value Function: 각 상태 및/또는 행동이 얼마나 좋은지  
• Model : 에이전트의 환경 표현

**Policy**   
Policy는 agent의 action입니다. 이를 통해, state에서 action으로의 지도/함수입니다.  
Deterministic policy: a=?(?) - 바로 action이 나온다.  
Stochastic policy: ?(a|s) = P[At = a | St = s] - 확률에 따라 나온다.

**Value Function**

Value Function는 미래 보상에 대한 예측이다. state의 좋음/나쁨을 평가하는 데 사용됨. 따라서 작업 중에서 선택된다.

![](/assets/images/posts/76/img_3.png)

**Model**

Model은 환경이 다음에 무엇을 할지 예측한다. ? 다음 상태 예측, ? 다음(즉각적인) 보상 예측

![](/assets/images/posts/76/img_4.png)

![](/assets/images/posts/76/img_5.png)

![](/assets/images/posts/76/img_6.png)

일정 경험을 진행하면 우리는 Deterministic policy형식으로 해당 위치에서 에로우를 따라가게된다.

이를 Value 상태로 보면 다음과 같다.

![](/assets/images/posts/76/img_7.png)

즉, 옳은 방향으로 인도한다는 것이다.

**Categorizing RL agents**

RL agents 분류는 다음과 같다.

• Value Based   
     • No Policy (implicit)   
     • Value Function   
• Policy Based   
     • Policy   
     • No Value Function   
• Actor Critic   
     • Policy   
     • Value Function   
• Model free   
     • Policy and/or Value Function   
     • no Model   
• Model based   
     • Policy and/or Value Function   
     • Model

RL도 다음 그림과 같이 많이 엮여있다. 그 만큼 종류가 다양하다는 것이다.

![](/assets/images/posts/76/img_8.png)

이것은 RL을 분류하는 대표적인 그림이다.

![](/assets/images/posts/76/img_9.png)

**Learning and Planning**

sequential decision making의 두 가지 근본적인 문제가 있다

• Planning:   
     • 환경 모델이 알려져 있다  
     • 에이전트는 모델과 함께 계산을 수행한다(외부 상호작용 없이).  
     • 에이전트가 정책을 개선한다  
• Reinforcement Learning:   
     • 환경은 처음에는 알려지지 않았다  
     • 에이전트는 환경과 상호 작용한다  
     • 에이전트가 정책을 개선한다

Example: Planning

• 게임의 규칙은 알려져 있다  
• (완벽한) 모델이 있는 시뮬레이터를 쿼리할 수 있게 된다  
• 상태 ?에서 ?조치를 취하면 다음 상태는 무엇입니까? 점수는 어떻게 될까요?에 대한 정보를 알 수 있다.  
• 최적의 정책을 찾기 위한 사전 계획을 위해 트리 검색과 같은 것을 사용한다.

![](/assets/images/posts/76/img_10.png)

**Example: Reinforcement Learning**

1. 게임의 규칙은 알 수 없습니다  
2. 인터랙티브 게임 플레이에서 직접 배우기  
3. 액션을 선택하고 픽셀과 점수를 확인

이를 반복한다. 아래 그림은 이를 보여주고 있다.

![](/assets/images/posts/76/img_11.png)

**Exploration and Exploitation**

강화 학습은 시행 착오 학습과 같다. 따라서 에이전트는 좋은 정책을 발견해야 빠르게 좋아질 수 있다.  
환경의 경험에서 도중에 너무 많은 보상을 잃지 않는 방향의 좋은 정책을 말이다.

Exploration은 환경에 대한 더 많은 정보를 찾는다.  
Exploitation는 알려진 정보를 악용하여 보상을 극대화한다.  
일반적으로 Exploration하고 활용하는 것이 중요하다.

여기서 Exploration과 Exploitation의 정의를 꼭 알고 넘어가야 한다.

예시를 한번 들어보자

1. 레스토랑 선택  
Exploitation : 좋아하는 식당에 가기  
Exploration : 새로운 레스토랑에 도전  
2. 온라인 배너 광고  
Exploitation : 가장 성공적인 광고 표시  
Exploration : 다른 광고 표시  
3. 석유 시추  
Exploitation : 가장 잘 알려진 위치에서 드릴  
Exploration : 새로운 위치에서 드릴  
4. 게임 플레이  
Exploitation : 최선이라고 생각하는 수를 사용해  
Exploration : 실험적인 움직임을 재생

![](/assets/images/posts/76/img_12.png)

Exploration and Exploitation

**Prediction and Control**

Prediction은 주어진 정책으로 미래를 평가한다  
Control: 최상의 정책 찾아 미래를 최적화하다

**Gridworld Example: Prediction**

목표: 주어진 Gridworld 환경에서 각 상태의 가치(Value)를 예측하는 것이다.  
상태(State): Gridworld 환경에서 각 셀이 상태가 되며, 셀의 위치를 나타낸다.  
행동(Action): 행동은 Prediction 문제에서는 고려하지 않는다. 즉, 각 상태에서 가능한 행동에 대한 선택은 필요하지 않는다.  
보상(Reward): Prediction 문제에서는 목표 지점에 대한 보상이 주어지지 않는다. 대신, 각 상태의 가치를 정확하게 예측하는 것이 중요하다.  
목표(Goal): 목표는 주어진 상태에서 정확한 가치를 예측하는 것이다. 이를 위해 최적의 가치 함수를 학습하는 것이 목표다.

![](/assets/images/posts/76/img_13.png)

Gridworld Example: Prediction

**Gridworld Example: Control**

목표: 주어진 Gridworld 환경에서 최적의 정책을 학습하여 특정 목표 지점에 도달하는 것이다.   
상태(State): Gridworld 환경에서 각 셀이 상태가 되며, 셀의 위치를 나타낸다.   
행동(Action): 주어진 상태에서 가능한 행동은 상하좌우로 이동하는 것이다.   
보상(Reward): 목표 지점에 도달하면 양의 보상을 받고, 벽에 부딪히거나 비효율적인 경로를 선택할 경우 음의 보상을 받는다.   
목표(Goal): 목표는 최소의 행동으로 목표 지점에 도달하는 것이다. 즉, 최적의 정책을 찾아내는 것이 목표이다.

![](/assets/images/posts/76/img_14.png)

Gridworld  Example: Control

**Summary**  
• **State, Action, Reward**   
• **Policy, Value, Model**   
• Planning vs. Learning   
• **Exploration vs. Exploitation**   
• Policy Prediction & Control
