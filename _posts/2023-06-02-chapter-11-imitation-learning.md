---
title: "Chapter 11. Imitation Learning"
date: 2023-06-02 00:37:14
categories:
  - 강화학습
tags:
  - Imitation Learning
---

**References**   
• ICML 2018 Imitation Learning Tutorial   
<https://sites.google.com/view/icml2018-imitation-learning/>

[ICML2018: Imitation Learning](https://sites.google.com/view/icml2018-imitation-learning/)

• CVPR 2018 Tutorial: Inverse Reinforcement Learning for Computer Vision

<https://www.youtube.com/watch?v=JbNeLiNnvII&t=1773s>

Imitation Learning은 사람이나 다른 전문가의 동작을 모방하여 에이전트를 학습시키는 강력한 기법입니다. 그렇다면 Imitation Learning에서 보상 함수는 어디에서 나오는 걸까요?

우리는 이전 강의에서는 게임을 기반으로 했기 때문에 아래처럼 생각할 것입니다.

![](/assets/images/posts/86/img.png)

하지만 실제 세계의 시나리오는 다릅니다. Reward라고 부를 만한게 딱히 없습니다.

Imitation Learning은 이를 해소시켜줍니다. 그렇다면 Imitation Learning에 대해서 알아볼까요?

![](/assets/images/posts/86/img_1.png)

위의 그림처럼 사람이 하는 동작을 로봇이 따라하는 것이 Imitation Learning입니다. 사람의 상태와 해당 상태의 action을 기억해서 이를 학습하는 것이죠. 따라서 다음과 같이 박스와 테이블이 있으면 이를 학습하여 로봇이 의자를 넘을 수 있죠.

![](/assets/images/posts/86/img_2.png)

즉, Imitation Learning에서 보상 함수는 사람이나 전문가의 행동을 모방하는 데 사용됩니다. 일반적으로 전문가는 주어진 작업을 수행하는 동안 특정 목표를 달성하기 위한 보상 신호를 제공합니다. 예를 들어, 자율 주행 자동차를 학습시킨다고 가정해 봅시다. 이 경우, 전문가는 자동차가 도로를 안전하게 운전하고 목적지에 도달할 때 양수의 보상을 주고, 교통 규칙을 위반하거나 사고를 일으킬 때 음수의 보상을 줄 수 있습니다.   
  
이렇게 전문가가 제공하는 보상 신호는 보통 사람의 도메인 지식과 경험에 기반하여 설계됩니다. 전문가의 피드백을 통해 좋은 결과를 얻기 위한 보상 함수를 정의할 수 있습니다. 때로는 전문가가 동작을 직접 제공하기보다는 기존의 기록된 전문가의 동작 데이터를 사용하여 보상 함수를 정의하는 경우도 있습니다.   
  
전문가의 동작을 모방하는 학습 과정에서는, 에이전트는 전문가와 유사한 행동을 취하면서 최대의 보상을 얻기 위해 학습됩니다. 보상 함수는 에이전트의 행동을 평가하고, 효과적인 행동을 장려하거나 부정적인 행동에 대해 벌점을 부과함으로써 에이전트의 학습을 돕습니다.

따라서 이를 앞선 State, Action, Policy 형태로 정의하면 다음과 같다.

![](/assets/images/posts/86/img_3.png)

Notation & Setup

이를 Racing Game에 적용한다면 다음과 같을 것이다.

![](/assets/images/posts/86/img_4.png)

위의 도로를 달리는 것을 기반으로 아래의 게임에 적용 시키는 방식이다. 따라서 우리가 이전 notation을 따르면 다음과 같다.

![](/assets/images/posts/86/img_5.png)

이런 방식을 Behavior Cloning이라고 말합다. Behavior Cloning은 Imitation Learning을 감독 학습(Supervised Learning)으로 간소화하는 방법 중 하나입니다.  
  
Behavior Cloning에서는 문제를 일반적인 머신 러닝 문제로 정의합니다. 먼저, 정책 클래스를 고정합니다. 이 클래스는 주로 신경망(neural network), 의사 결정 트리(decision tree) 등과 같은 것들이 사용됩니다. 그리고 학습 예제인 (?0,?0),(?1,?1),(?2,?2),...를 사용하여 정책을 추정합니다.

![](/assets/images/posts/86/img_6.png)

Behavior Cloning

이러한 방식의 해석은 다음과 같습니다. 우선, 지금까지의 완벽한 모방을 가정하고, 완벽하게 모방을 계속하는 것을 학습합니다. 또는 전문가의 경로를 따라 1단계의 오차를 최소화하는 것입니다.

![](/assets/images/posts/86/img_7.png)

이러한 Behavior Cloning은 전문가의 동작을 모방하기 위해 데이터를 사용하여 정책을 학습하는 간단하고 직관적인 방법입니다. 하지만 몇 가지 주의할 점도 있습니다. 예를 들어, 전문가의 동작을 완벽하게 모방하는 것이 아니라 그들의 경로에서 벗어나는 경우에는 어떻게 해야할지 고려해야 합니다.

이러한 에러가 이러나는 이유는 간단하게 다음과 같이 설명됩니다.

![](/assets/images/posts/86/img_8.png)

한번 상태를 벗어나면 이를 학습하지 못했기 때문에, 새로운 상태로 인식하여 오차가 점점 커집니다.

Types of Imitation Learning은 다음과 같다. 우리는 Behavior Cloning을 살펴 보았으니, Inverse RL과 Direct Policy Learning을 알아볼 것 이다.

![](/assets/images/posts/86/img_9.png)

Types of Imitation Learning

![](/assets/images/posts/86/img_10.png)

Types of Imitation Learning

**DAGGER: Dataset Aggregation**

DAGGER에서는 Behavior Cloning으로 계산된 정책이 취한 경로에 따라 전문가의 행동 라벨을 더 수집합니다. 즉, 처음에는 Behavior Cloning을 사용하여 전문가의 동작을 모방하는 정책을 학습합니다. 그런 다음 이 학습된 정책을 사용하여 새로운 경로를 샘플링합니다.   
  
새로운 경로에서는 전문가에게 해당 경로에서의 행동 라벨을 얻습니다. 이 라벨은 전문가의 동작과 비교하여 행동 모델을 보정하는 데 사용됩니다. 이 과정을 반복하여 더 많은 경로에서의 전문가의 행동 라벨을 얻고, 보다 정확한 행동 모델을 학습하는 데 사용합니다.

![](/assets/images/posts/86/img_11.png)

DAGGER: Dataset Aggregation

DAGGER는 Behavior Cloning의 단점 중 하나인 경로에서 벗어나는 문제를 완화할 수 있습니다. Behavior Cloning은 처음 학습한 정책을 그대로 사용하기 때문에, 전문가의 경로에서 벗어날 수 있습니다. 하지만 DAGGER에서는 반복적으로 전문가의 피드백을 받으며 정책을 보정함으로써, 보다 안정적이고 정확한 행동 모델을 학습할 수 있습니다.   
  
즉, DAGGER는 데이터 집계를 통해 전문가의 지식을 적극적으로 활용하여 Imitation Learning의 성능을 향상시키는 강력한 방법입니다.

**Inverse Reinforcement Learning**

Inverse Reinforcement Learning은 보통의 강화학습과 반대로, 보상 함수를 찾는 문제로 접근합니다. 이제 Inverse Reinforcement Learning에 대해 자세히 알아보겠습니다.

![](/assets/images/posts/86/img_12.png)

강화학습에서는 일반적으로 환경으로부터 보상을 받아 최적의 정책을 학습합니다. 이는 "regular" 강화학습이라고 할 수 있습니다. 하지만 Inverse Reinforcement Learning에서는 정책이 주어지고, 그에 맞는 보상 함수를 찾는 문제로 접근합니다.

![](/assets/images/posts/86/img_13.png)

즉, Inverse Reinforcement Learning은 전문가에 의해 제공된 정책을 기반으로 최적의 보상 함수를 찾는 것입니다. 전문가의 정책은 이미 문제를 해결하는 데 효과적인 동작을 보여줍니다. 이제 이 정책과 일치하는 보상 함수를 찾기 위해 Inverse Reinforcement Learning을 사용합니다.   
  
Inverse Reinforcement Learning은 주로 전문가의 행동을 해석하고, 그 행동을 지원하는 보상 함수를 찾는 데 사용됩니다. 이를 통해 에이전트는 전문가의 행동을 모방하거나 전문가 수준의 행동을 배울 수 있습니다.

**Feature Based Reward Function**

Feature Based Reward Function은 하나 이상의 전문가의 데모 (?0,?0,?1,?1,...)를 사용하여 보상 함수를 유추하는 방법입니다.  
  
Feature Based Reward Function에서는 상태 공간, 행동 공간, 전이 모델 ?(?′|?,a)이 주어진 상황에서 보상 함수 ?이 없는 상태에서 시작합니다. 목표는 보상 함수 ?을 추론하는 것입니다. 이때, 전문가의 정책이 최적이라고 가정합니다.

Feature Based Reward Function은 주로 전문가의 데모를 기반으로 보상 함수를 추론하기 위해 사용됩니다. 데모는 상태와 해당 상태에서의 전문가의 행동으로 구성됩니다. 이를 통해 보상 함수를 추론하는 데 사용됩니다.   
  
특히, 전문가의 정책이 최적이라고 가정한다면, 보상 함수에 대한 유용한 정보를 얻을 수 있습니다. 최적의 정책은 보상을 최대화하는 행동을 선택하기 때문에, 해당 상태에서 보상이 높은 특징(feature)에 대한 정보를 유추할 수 있습니다.   
  
이러한 방식으로 Feature Based Reward Function은 보상 함수를 추론하고, 추론된 보상 함수를 사용하여 에이전트를 학습하는 데 활용됩니다. 이를 통해 에이전트는 전문가의 행동을 모방하거나 전문가의 수준에 도달할 수 있습니다.

**Linear Feature Reward Inverse RL**

Linear Feature Reward Inverse RL은 선형 특징 기반 보상 역강화 학습입니다. 이 알고리즘은 선형 특징 기반 보상 역강화 학습입니다. 주어진 전문가의 행동 데이터와 관련된 특징 벡터를 사용하여 보상 함수를 추론하는 과정을 거칩니다. 일반적으로 선형 모델을 사용하여 특징과 보상 함수 사이의 관계를 모델링하고 추론합니다. 이 알고리즘은 보상 함수의 선형성 가정을 기반으로 하며, 특징 벡터와 가중치 벡터 간의 선형 조합으로 보상을 추정합니다.

![](/assets/images/posts/86/img_14.png)

**Relating Frequencies to Optimality**

Relating Frequencies to Optimality는 빈도를 최적성과 관련시키는 개념입니다. 이 알고리즘은 특정 행동 또는 상태의 빈도를 최적 정책과 관련시키는 개념입니다. 주어진 전문가의 행동 데이터에서 특정 상태 또는 행동의 빈도를 분석하여, 해당 상태 또는 행동이 최적 정책과 어떻게 관련되어 있는지를 알아냅니다. 이를 통해 최적 정책과 관련된 빈도를 파악하고, 최적 정책을 학습하는 데에 활용할 수 있습니다.

![](/assets/images/posts/86/img_15.png)

**Feature Matching**

Feature Matching은 특징 일치를 의미합니다. 특정 상태에서 특징 카운트를 일치시키는 데 초점을 맞춥니다. 이를 통해 특정 상태에서 전문가와 유사한 행동을 수행하는 정책을 찾는 것입니다. (GAN에서 나왔던 내용입니다.)

이 알고리즘은 특징 일치를 중요시하는 개념입니다. 주어진 전문가의 행동 데이터에서 특정 상태에서의 특징 카운트를 일치시키는 것을 목표로 합니다. 즉, 전문가와 유사한 행동을 수행하는 정책을 찾기 위해 특정 상태에서의 특징 값이 일치하도록 정책을 조정합니다. 이를 통해 전문가의 행동을 모방하거나 유사한 행동을 수행하는 정책을 학습하는 데에 활용할 수 있습니다.

![](/assets/images/posts/86/img_16.png)

**Ambiguity**

Ambiguity에 대해 이야기해보겠습니다. Ambiguity는 불확실성을 의미합니다. 보상 함수를 추론하는 과정에서 동일한 최적 정책과 관련된 무수히 많은 보상 함수가 존재할 수 있습니다. 또한, 특징 카운트를 일치시키는 데에도 무수히 많은 확률 정책이 존재할 수 있습니다. 이 때, 어떤 보상 함수 또는 정책을 선택해야 하는지에 대한 문제가 발생합니다.   
  
Ambiguity는 Imitation Learning의 한계 중 하나이며, 이를 해결하기 위해 추가적인 제약이나 페널티를 도입하여 더 정확한 보상 함수나 정책을 찾는 방법을 고려할 수 있습니다.

**Max-Margin Planning**

Max-Margin Planning은 강화학습과 경계 최대화 사이의 관계를 활용하여 최적의 정책을 학습하는 방법입니다.   
  
Max-Margin Planning은 강화학습과 지지 벡터 머신(Support Vector Machine)의 개념을 결합한 방법입니다. 이 방법은 보통의 강화학습과는 다르게 경계 최대화를 통해 최적의 정책을 학습합니다. 이를 통해 전문가의 행동과 모델의 행동 간의 차이를 최소화하는 최적의 정책을 찾습니다.

![](/assets/images/posts/86/img_17.png)

Max-Margin Planning은 주어진 전문가의 행동 데이터를 기반으로 모델을 학습하는 동시에 경계 최대화를 통해 최적의 정책을 찾습니다. 이를 위해 지지 벡터 머신의 아이디어를 활용하여 모델의 행동과 전문가의 행동 간의 마진을 최대화합니다. 이러한 접근 방식을 통해 전문가의 행동과 유사한 행동을 선택하는 최적의 정책을 학습할 수 있습니다.   
  
Max-Margin Planning은 특히 데이터의 불확실성을 고려하여 모델의 학습과 정책의 결정을 조율하는 데에 사용됩니다. 전문가의 행동과 모델의 행동 간의 차이를 최소화하면서도 모델의 불확실성을 고려하여 최적의 정책을 결정합니다.

다음은 예시들이다.

![](/assets/images/posts/86/img_18.png)

![](/assets/images/posts/86/img_19.png)

Modeling Taxi Driver

**Generative Adversarial Imitation Learning (GAIL) [2016 NIPS]**  
GAIL은 Imitation Learning과 Generative Adversarial Network (GAN)을 결합한 방법입니다.  
  
GAIL은 전문가의 데모를 모방하는 정책을 GAN을 사용하여 학습하는 방법입니다. GAIL은 전문가의 행동 데이터를 사용하여 생성자(generator)를 학습시킵니다. 생성자는 전문가의 행동과 유사한 행동을 생성하도록 학습됩니다.

![](/assets/images/posts/86/img_20.png)

GAIL에서는 생성자와 구분자(discriminator)라는 두 개의 네트워크가 사용됩니다. 생성자는 전문가와 유사한 행동을 생성하려고 하고, 구분자는 생성된 행동이 전문가의 행동인지 아닌지를 판별합니다. 생성자는 구분자를 속이는 방식으로 학습을 진행합니다. 즉, 생성자는 구분자에게 생성된 행동이 전문가의 행동으로 인식되도록 학습하려고 노력합니다.   
  
GAIL은 GAN의 경쟁적 학습을 활용하여 전문가의 행동을 모방하는 최적의 정책을 학습합니다. 생성자는 구분자를 속이기 위해 행동을 조정하고, 구분자는 생성자를 학습하여 생성된 행동을 정확하게 식별하도록 노력합니다. 이러한 경쟁적 학습을 통해 생성자는 점차적으로 전문가의 행동을 모방하는 정책을 학습하게 됩니다.   
  
GAIL은 Imitation Learning에서 전문가의 행동을 잘 모방하면서도 생성자의 다양성과 새로운 행동을 도입할 수 있는 장점을 가지고 있습니다.

![](/assets/images/posts/86/img_21.png)

알고리즘은 GAN과 유사함을 알 수 있다.

Robotics Experiments

![](/assets/images/posts/86/img_22.png)

**One/Few Shot Imitation Learning**

One/Few Shot Imitation Learning은 매우 제한된 데이터로부터 효과적으로 학습하는 방법입니다.  
  
One/Few Shot Imitation Learning은 말 그대로 매우 적은 양의 데이터로부터 학습하는 방법을 의미합니다. 보통의 Imitation Learning은 많은 수의 전문가의 데모나 경험이 필요하지만, One/Few Shot Imitation Learning은 한 개 또는 소수의 데모만으로도 효과적으로 학습할 수 있습니다.   
  
One/Few Shot Imitation Learning은 전문가의 행동을 잘 모방하기 위해 제한된 데이터를 최대한 활용하는 방법을 사용합니다. 이를 위해 데이터 증강(data augmentation) 기법이나 메타 학습(meta-learning) 기법 등을 활용하여 데이터의 다양성을 높이고 일반화 능력을 향상시킵니다.

![](/assets/images/posts/86/img_23.png)

또한, One/Few Shot Imitation Learning은 전문가의 행동 패턴을 잘 이해하고 일반화할 수 있는 모델 구조를 사용하는 것이 중요합니다. 이를 위해 신경망 아키텍처나 메타-학습 알고리즘 등을 사용하여 제한된 데이터로부터 효과적으로 학습할 수 있도록 합니다.   
  
One/Few Shot Imitation Learning은 실제 환경에서 전문가의 도움 없이도 새로운 작업이나 도메인에 빠르게 적응할 수 있는 유용한 기법입니다. 이를 통해 학습 시간과 노력을 크게 절약하면서도 높은 성능을 달성할 수 있습니다.

다음은 적용된 예시들입니다.

![](/assets/images/posts/86/img_24.png)

**Learning by Imitating Animals [RSS 2020]**

"Learning by Imitating Animals" 연구는 동물의 행동을 모방하여 학습하는 방법에 대한 내용을 다루고 있습니다. 동물은 복잡한 환경에서 생존과 행동에 대한 훌륭한 전문가로 알려져 있습니다. 이 연구에서는 동물의 행동을 모방하여 로봇이나 인공 시스템의 학습과 제어에 활용하는 방법을 탐구하고 있습니다.

![](/assets/images/posts/86/img_25.png)

Learning by Imitating Animals [RSS 2020]

동물의 행동을 모방하여 학습하는 접근 방식은 다양한 도메인에 적용될 수 있습니다. 예를 들어, 동물의 움직임이나 사냥 기술, 소통 방식 등을 모방하여 로봇의 움직임 제어, 자율 주행 자동차의 운전 전략 개선, 의료 분야에서의 수술 기술 향상 등에 응용될 수 있습니다.   
  
이러한 연구는 동물의 행동을 이해하고 모방함으로써 자연의 지혜와 효율성을 인공 시스템에 적용하는 새로운 접근 방식을 제시합니다. 동물의 행동을 모방하는 것은 복잡한 환경에서의 효율적인 행동을 학습하는 데에 도움이 될 수 있습니다.   
  
"Learning by Imitating Animals" 연구는 Imitation Learning의 한 분야로서 동물 행동 모방을 통해 다양한 도메인에서의 학습과 제어에 새로운 가능성을 열어줍니다.

<https://www.youtube.com/watch?v=lKYh6uuCwRY>

**Summary**

이번 강의에서는 Imitation Learning에 대해 다양한 내용을 다루었습니다. 이를 요약하면 다음과 같습니다.   
  
Imitation Learning은 좋은 정책을 학습하기 위해 필요한 데이터 양을 크게 줄일 수 있는 방법입니다. 전문가의 데모를 통해 학습하는 것으로, 데이터를 효과적으로 활용하여 정확한 정책을 학습할 수 있습니다.   
  
Inverse RL은 전문가의 데모로부터 알려지지 않은 보상 함수를 추론하는 방법입니다. 전문가의 행동을 토대로 보상 함수를 유추하여 학습을 진행합니다.   
  
Imitation Learning은 다양한 응용 분야에서 활용될 수 있습니다. 로봇 제어, 자율 주행 자동차, 의료 분야 등에서 전문가의 행동을 모방하여 성능을 향상시키는데 활용될 수 있습니다.   
  
하지만 Imitation Learning에는 여전히 도전 과제가 남아있습니다. 크고 연속적인 상태와 행동 공간에서 효과적인 학습을 진행하는 것이 어렵습니다. 또한, 알려지지 않은 동적 환경에서 효과적인 학습을 수행하는 것도 도전적입니다.   
  
Inverse RL과 온라인 강화학습을 결합하는 방법도 존재합니다. 이를 통해 전문가의 행동을 모방하면서도 동적인 환경에서 효과적으로 학습할 수 있습니다.   
  
이상으로 이번 강의에서는 Imitation Learning에 대해 다양한 내용을 다루었습니다. Imitation Learning은 데이터 효율성을 향상시키고 다양한 응용 분야에서 활용할 수 있는 강력한 기법입니다. 하지만 여전히 도전 과제가 존재하며, 앞으로 더 많은 연구와 개선이 이루어질 것입니다.
