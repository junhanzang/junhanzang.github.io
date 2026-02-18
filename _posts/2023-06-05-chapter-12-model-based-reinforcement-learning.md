---
title: "Chapter 12. Model-based Reinforcement Learning"
date: 2023-06-05 15:20:48
tags:
  - Model-based Reinforcement Learning
---

이번 챕터에서는 **Model Based Reinforcement Learning**에 대해 알아보겠습니다. Model Based Reinforcement Learning은 경험을 통해 직접 모델을 학습하고, 이를 활용하여 계획(planning)을 수행하여 가치 함수나 정책을 구축하는 방법입니다.   
  
Model Based Reinforcement Learning은 경험 데이터를 통해 환경의 모델을 직접 학습하는 방법입니다. 이 모델은 주어진 상태에서 행동을 취하면 어떤 상태로 전이되는지 예측할 수 있는 도구입니다. 이 모델은 환경의 동작을 학습하는 데 사용되며, 이를 통해 계획 알고리즘과 함께 가치 함수나 정책을 구축할 수 있습니다.   
  
Model Based Reinforcement Learning은 계획 알고리즘을 활용하여 학습한 모델을 기반으로 가치 함수나 정책을 생성하는 방식입니다. 계획은 학습한 모델을 사용하여 미래의 보상을 예측하고 최적의 행동을 선택하는 과정입니다. 이를 통해 최적의 결정을 내리는 데에 활용될 수 있습니다.   
  
Model Based Reinforcement Learning은 계획과 학습을 하나의 아키텍처로 통합하는 것을 목표로 합니다. 모델 학습과 계획 과정이 서로 상호작용하면서, 보다 효율적인 학습과 결정을 이끌어낼 수 있습니다. 이는 계획과 학습 사이의 유기적인 통합을 통해 높은 성능과 효율성을 달성할 수 있는 장점을 가지고 있습니다.   
  
즉, Model Based Reinforcement Learning은 학습한 모델을 활용하여 계획과 결합하여 최적의 정책을 생성하는 강력한 기법입니다.

Model-based Reinforcement Learning에서 중요한 요소인 **Model, Planning, Learning**에 대해 알아보겠습니다.

1. 첫 번째로, Planning에 대해 알아보겠습니다. Planning은 모델과 상호작용하여 가치 함수나 정책을 계획하는 과정입니다. Model-based & dynamic programming은 환경의 모델을 활용하여 계획을 수행하는 방식입니다. 이를 통해 최적의 결정을 내리는데 사용됩니다.   
  
2. 두 번째로, Model-free RL에 대해 알아보겠습니다. Model-free RL은 모델 없이 경험 데이터로부터 가치 함수나 정책을 학습하는 방법입니다. 모델 없이 직접 경험을 통해 학습하므로 환경의 동작을 예측할 수 없지만, 데이터만으로도 최적의 정책을 학습할 수 있습니다.   
  
3. 세 번째로, Model-based RL에 대해 알아보겠습니다. Model-based RL은 경험 데이터로부터 모델을 학습하고, 이를 활용하여 계획을 수행하는 방법입니다. 모델 학습을 통해 환경의 동작을 예측하고, 이를 기반으로 최적의 가치 함수나 정책을 계획합니다.   
  
Model, Planning, Learning은 Model-based Reinforcement Learning에서 각각의 중요한 구성 요소입니다. Planning은 모델을 활용하여 최적의 결정을 내리는 과정을 의미하며, Model-free RL은 모델 없이 직접 경험을 통해 학습하는 방식입니다. Model-based RL은 모델을 학습하고 이를 활용하여 계획을 수행하는 방법입니다.

**Model-based RL vs. Model-free RL**

Model-based Reinforcement Learning의 중요한 두 가지 접근 방식인 Model-based RL과 Model-free RL에 대해 비교해보겠습니다.

![](https://blog.kakaocdn.net/dna/b1qhfi/btsiwjoaKcS/AAAAAAAAAAAAAAAAAAAAAAZvWSrbAqh5fKglGpgc13eLQcUVhO720-27-8S9MqC4/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=fvRyVWsRVrL2o0chG6uTq0dyMD0%3D)

Model-based&nbsp;RL

먼저, Model-based RL은 모델을 활용하여 환경의 동작을 예측하고 계획을 수행하는 방식입니다. 모델은 주어진 상태와 행동으로부터 다음 상태와 보상을 예측하는 함수로 사용됩니다. 이를 통해 최적의 가치 함수나 정책을 계획하여 결정을 내립니다. Model-based RL은 모델을 학습하기 위해 경험 데이터를 사용하고, 학습된 모델을 활용하여 계획을 수행합니다.

![](https://blog.kakaocdn.net/dna/W3Jb3/btsiOhCbW1R/AAAAAAAAAAAAAAAAAAAAAME4kGcW9Jwna6hDrTXwaTHOxtWPNWKbUE5xKOUfqVf0/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=UZGp3MkGf2mDwiV7nIx6HNlVtvE%3D)

Model-free&nbsp;RL

반면에, Model-free RL은 모델 없이 직접 경험 데이터로부터 가치 함수나 정책을 학습하는 방식입니다. 모델을 사용하지 않고 주어진 상태에서 행동을 선택하고 보상을 받아 학습을 진행합니다. Model-free RL은 경험 데이터만으로부터 학습하므로 모델의 동작을 예측할 수는 없지만, 데이터만으로도 최적의 정책을 학습할 수 있습니다.   
  
Model-based RL과 Model-free RL은 각자의 장단점이 있습니다. Model-based RL은 모델을 활용하여 계획을 수행하기 때문에 효율적인 학습과 결정을 이끌어낼 수 있습니다. 하지만 모델을 학습하는 과정이 필요하므로 추가적인 계산과 시간이 소요될 수 있습니다. 반면에 Model-free RL은 모델을 사용하지 않고 직접 경험을 통해 학습하므로 학습과 결정 속도가 빠를 수 있지만, 데이터만으로부터 학습하기 때문에 학습의 효율성이 낮을 수 있습니다.

**Model-based RL**

![](https://blog.kakaocdn.net/dna/k1l67/btsiFDF7mEh/AAAAAAAAAAAAAAAAAAAAAPWeS6hdm7S0Ut6hP2l0KzRSPZIhXRzPurQQJgHZKYgG/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=LO4ZnF7XncZLhTkLVWQywrnh0pU%3D)

Model-based&nbsp;RL

**Advantage(장점)과 Disadvantage(단점)**

첫 번째로, Model based RL은 모델을 효율적으로 학습할 수 있는 장점이 있습니다. 모델은 supervised learning 방법을 사용하여 학습할 수 있으며, 주어진 상태와 행동으로부터 다음 상태와 보상을 예측하는 함수로 학습됩니다. 이를 통해 모델이 환경의 동작을 예측하고 최적의 가치 함수나 정책을 계획하는데 활용될 수 있습니다.   
  
두 번째로, 학습된 모델을 활용하여 추가적인 샘플링 없이도 최적의 정책을 계획할 수 있는 장점이 있습니다. 모델이 환경의 동작을 예측하고 가치 함수나 정책을 계획하는데 사용되므로, 새로운 샘플링 없이도 최적의 결정을 내릴 수 있습니다. 이는 샘플링에 필요한 시간과 비용을 줄이고 효율적인 의사 결정을 가능하게 합니다.   
  
Model based RL의 장점은 위와 같습니다. 하지만 이 방법에는 몇 가지 단점도 있습니다. 첫 번째로, 모델을 학습한 후에 가치 함수를 구성해야 한다는 점입니다. 모델 학습과 가치 함수 구성으로 인해 근사 오차(approximation error)가 두 가지 소스로 발생할 수 있습니다.

![](https://blog.kakaocdn.net/dna/M6irg/btsiJ7fOSG2/AAAAAAAAAAAAAAAAAAAAAH3EnS_J-23ccp6gUf0OhT3jFk51V9F04NwUdVnTpFXP/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=jN1wDEqWcsj%2BW7UD08TO4PtFqds%3D)

그렇다면 Model은 무엇일까요?

Model은 상태 공간 𝑆, 행동 공간 𝐴, 상태 전이 확률 𝑃, 보상 함수 𝑅 등으로 구성된 MDP(Markov Decision Process) 𝑀의 표현입니다. 𝑀은 파라미터 𝜂로 매개화되며, 모델 𝑀𝜂은 상태 전이 확률 𝑃𝜂≈𝑃와 보상 함수 𝑅𝜂≈𝑅를 나타냅니다.

그렇다면 이 model을 어떻게 학습시킬까?

Model Learning은 경험 데이터 𝑆1,𝐴1,𝑅2,...,𝑆𝑇로부터 모델 𝑀𝜂을 추정하는 것을 목표로 합니다. 이는 지도 학습 문제로 볼 수 있으며, 𝑠,𝑎,→𝑟은 회귀 문제로, 𝑠,𝑎,→𝑠′는 밀도 추정 문제로 간주됩니다. 손실 함수(예: 평균 제곱 오차, KL 다이버전스 등)를 선택하고, 경험 데이터에 대한 경험적 손실을 최소화하는 파라미터 𝜂를 찾습니다.

이를 합쳐 Table Lookup Model라하며 Table Lookup Model은 명시적인 MDP, 𝑃,𝜂 𝑅으로 모델이 구성된 경우를 의미합니다. 각 상태-행동 쌍의 방문 횟수 𝑁(𝑠,𝑎)를 계산합니다.

![](https://blog.kakaocdn.net/dna/cfVoqn/btsivSqEsHh/AAAAAAAAAAAAAAAAAAAAAK-sc_8e-A7oP_1txuADLK82Xs4C6jBbDbcwupldjfM6/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=RoFiHr4jEvjDOOkZB9U2LWZ4muQ%3D)

또는 각 시간 단계에서 경험 튜플 𝑆𝑡,𝐴𝑡,𝑅𝑡+1,𝑆𝑡+1과 같은 튜플을 기록합니다. 모델을 샘플링하기 위해 𝑠,𝑎,∙,∙과 일치하는 튜플을 무작위로 선택합니다.

이를 **AB Example**에 대해 적용해보자.  
AB Example은 이전 chapter에서 사용했으니 설명은 넘어가겠습니다.

![](https://blog.kakaocdn.net/dna/80CDR/btsiNvt2C45/AAAAAAAAAAAAAAAAAAAAANl2TmQHr3LuvNfdu7ISf-raXfeKpC4yO7A8tDUdKE0E/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=AtmzUsaYtJnf650jdSxSZ89tjr0%3D)

AB Example

**Planning with a Model**

주어진 모델 𝑀𝜂=𝑃𝜂,𝑅𝜂을 활용하여 MDP를 해결하는 것이 목표입니다. 이를 위해 여러 가지 계획 알고리즘 중 선호하는 방법을 사용합니다. 예를 들어, Policy Iteration이나 Value Iteration과 같은 알고리즘을 활용하여 최적의 정책을 계획합니다. 모델을 사용하여 상태 전이 확률과 보상 함수를 이용하여 계획을 수행합니다.

**Sample-based Planning**

Sample-based Planning은 계획에 간단하지만 강력한 접근 방식입니다. 모델을 사용하여 샘플을 생성하는데에만 활용합니다. 모델로부터 경험 데이터를 샘플링하고, 이를 Model-free RL (MC, Sarsa, Q-learning 등)에 적용하여 학습합니다. 이러한 Sample-based Planning 방법은 종종 효율적인 방식으로 알려져 있습니다.

그래서 AB Example로 돌와봅시다.

AB 예제에 다시 돌아와서 실제 경험으로부터 Table Lookup Model을 구축하고, 샘플된 경험에 Model-free RL을 적용하는 방법에 대해 알아보겠습니다.   
  
AB 예제에서는 실제 경험 데이터를 사용하여 Table Lookup Model을 구축하는 것이 가능합니다. 이를 위해 각 상태-행동 쌍에 대한 방문 횟수를 카운트하거나, 각 시간 단계에서 경험 튜플을 기록할 수 있습니다. 이렇게 구축된 모델은 경험을 기반으로 상태 전이 확률과 보상을 추정하는 데 사용될 수 있습니다.   
  
한편, 구축된 Table Lookup Model을 사용하여 샘플된 경험에 Model-free RL을 적용할 수 있습니다. 모델로부터 경험 데이터를 샘플링하고, 이를 Model-free RL 알고리즘 (예: MC, Sarsa, Q-learning 등)에 적용하여 가치 함수나 정책을 학습합니다. 이렇게 모델로부터 샘플된 경험에 Model-free RL을 적용함으로써 최적의 결정을 내릴 수 있는 학습이 진행됩니다.   
  
AB 예제에서는 실제 경험 데이터를 활용하여 Table Lookup Model을 구축하고, 이를 통해 Model-free RL 알고리즘을 적용하는 방법을 사용합니다.

![](https://blog.kakaocdn.net/dna/qkb2t/btsiOgi9HDv/AAAAAAAAAAAAAAAAAAAAAIIkIfhzFUH-QlY6-ubPy-gDvplj-rJ0YXViIWS1nPZ_/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=3WTnwmvc3N5QTvXYP3A6i4z2QG4%3D)

**Planning with an Inaccurate Model**

주어진 부정확한 모델 𝑃𝜂,𝑅𝜂≠𝑃,𝑅을 가지고 있는 경우, 모델 기반 RL의 성능은 근사적인 MDP 𝑆,𝐴,𝑃𝜂,𝑅𝜂에 대한 최적 정책에 제한됩니다. 이 경우 모델이 부정확할 때, 계획 과정은 부적합한 정책을 계산하게 됩니다. 모델 기반 RL은 추정된 모델만큼 정확하다고 할 수 있습니다.   
  
이런 상황에서는 모델이 정확하지 않기 때문에 계획 과정은 최적 정책을 계산하지 못하고 부적합한 정책을 계산하게 됩니다. 모델 기반 RL은 추정된 모델의 정확성에 따라 성능이 결정되므로, 모델이 얼마나 정확한지에 따라 최적의 결정을 내릴 수 있는 능력이 제한될 수 있습니다.   
  
따라서, 모델이 정확하지 않은 경우, 모델 기반 RL은 최적의 정책을 구할 수 없으며, 추정된 모델에 따라 부적절한 정책을 계획하게 됩니다. 모델 기반 RL은 추정된 모델의 정확도에 의존하기 때문에, 모델의 추정 성능이 모델 기반 RL의 성능을 좌우하는 중요한 요소입니다.

**Real and Simulated Experience**

Real and Simulated Experience는 두 가지 경험의 원천에 대해 고려합니다.   
  
첫 번째는 Real Experience입니다. 이는 환경으로부터 샘플링된 실제 경험 데이터로서, 진짜 MDP(Markov Decision Process)에서 얻어진 데이터입니다. 실제 환경에서 행동을 선택하고 상태 전이가 발생하며 보상을 받는 과정에서 얻어진 데이터를 의미합니다.

![](https://blog.kakaocdn.net/dna/cdM7at/btsiOJ6tbtX/AAAAAAAAAAAAAAAAAAAAAKpXVe9nAqs0DXvuvVsrorPi4XGpjf-qk2DxJRnlZNR4/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=GrKJGaGDng7dbXVD49onRgikwCM%3D)

Real&nbsp;Experience

두 번째는 Simulated Experience입니다. 이는 모델로부터 샘플링된 경험 데이터로서, 근사적인 MDP(approximate MDP)에서 얻어진 데이터입니다. 모델을 활용하여 상태 전이와 보상을 모사한 데이터를 생성합니다. 이를 통해 모델로부터 생성된 가상의 환경에서 학습을 진행할 수 있습니다.

![](https://blog.kakaocdn.net/dna/b22tVo/btsiwjWcD0I/AAAAAAAAAAAAAAAAAAAAANMDPkCo1J24lZr0umZCbuykMLB90BD2CuAS5jl8S44N/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=jb%2FRvIbhAm%2BH6eNZJthOnxl8Hr8%3D)

Simulated&nbsp;Experience

Real Experience와 Simulated Experience는 각각 실제 환경과 모델로부터 얻은 경험 데이터로, 각각의 특징과 장단점이 있습니다. 실제 경험 데이터는 진짜 환경에서 얻어진 데이터이기 때문에 실제 상황에 가까운 학습이 가능합니다. 반면, 모델로부터 생성된 경험 데이터는 모델의 특성을 반영하며, 실험과 학습을 더 효율적으로 수행할 수 있습니다.   
  
Real and Simulated Experience는 실제 환경과 모델로부터 얻은 경험 데이터를 활용하여 학습과 실험을 진행할 때의 두 가지 선택지를 제시합니다. 이를 적절히 활용하여 모델 기반 RL의 학습과 결정을 진행할 수 있습니다.

**Integrating Learning and Planning**

학습과 계획을 통합하는 방법은 다음과 같이 여러 가지 방식으로 나타낼 수 있습니다.   
  
첫 번째로, Model Free RL입니다. 이 방법은 실제 경험 데이터로부터 가치 함수 (및/또는 정책)를 학습합니다. 모델 없이 실제 경험을 통해 가치 함수를 학습하여 최적의 결정을 내립니다.   
  
두 번째로, Model Based RL (using Sample Based Planning)입니다. 이 방법은 실제 경험 데이터로부터 모델을 학습하고, 가상의 경험 데이터를 생성하여 계획을 수행합니다. 실제 경험으로부터 모델을 학습한 후, 모델로부터 생성된 가상의 경험 데이터를 활용하여 가치 함수나 정책을 계획합니다.   
  
세 번째로, Dyna [AAAI 1991]입니다. 이 방법은 실제 경험 데이터로부터 모델을 학습하고, 실제 경험과 가상의 경험 데이터를 모두 활용하여 가치 함수 (및/또는 정책)을 학습하고 계획합니다. 실제 경험을 통해 모델을 학습한 후, 실제 경험과 모델로부터 생성된 가상의 경험 데이터를 모두 활용하여 가치 함수나 정책을 학습하고 계획합니다.

![](https://blog.kakaocdn.net/dna/vMVi1/btsiu1O3ff3/AAAAAAAAAAAAAAAAAAAAANH6jbCScgR_OmJIDCzq_snEjRAdltOjkSVDS-_H3DHB/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=9tOKNEuYlJH0G%2BvxptH3KwaxaBs%3D)

Dyna Architecture

Dyna Q-Algorithm에 대해 알아보겠습니다. Dyna Q-Algorithm은 Dyna Architecture에서 사용되는 알고리즘 중 하나입니다. 이 알고리즘은 실제 경험과 가상의 경험 데이터를 모두 활용하여 Q-함수를 학습하고, 이를 기반으로 최적의 정책을 결정합니다. 실제 경험 데이터로부터 모델을 학습한 후, 모델로부터 생성된 가상의 경험 데이터를 활용하여 Q-함수를 업데이트하고, 최적의 결정을 내리는 과정을 반복합니다.

![](https://blog.kakaocdn.net/dna/bnuX68/btsiFPtdARU/AAAAAAAAAAAAAAAAAAAAAClttM8U5MmwHxx3JMTRHTaiK1Eo2Q09MxIq0mndzy1g/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=EO11DprqJuJrRV%2BPdaQnawCiisU%3D)

Dyna Q-Algorithm

**Dyna-Q on a Simple Maze**

우리의 목표는 미로 환경에서 최단 경로를 찾는 것입니다. 이를 위해 미로의 상태와 행동을 기반으로 Q-함수를 업데이트하고 최적의 결정을 내립니다. 실제 경험 데이터로부터 모델을 학습한 후, 모델로부터 생성된 가상의 경험 데이터를 활용하여 Q-함수를 업데이트합니다.

![](https://blog.kakaocdn.net/dna/tuBy2/btsiBKlrUBS/AAAAAAAAAAAAAAAAAAAAALaWMEvpSprl-cPFiHHaza3qP74tH1mzvfcse5GC-_BX/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=YlQLh42cBDYm0WT2cmYEL9n2otA%3D)

Dyna Q-Algorithm은 실제 경험과 가상의 경험 데이터를 교차로 활용하며, 반복적으로 Q-함수를 업데이트하고 최적의 정책을 결정하는 과정을 수행합니다. 이를 통해 미로 환경에서 최단 경로를 학습하는 과정을 진행할 수 있습니다.

**Forward Search**

 Forward Search 알고리즘은 전방 탐색을 통해 최적의 행동을 선택합니다. 이를 위해 현재 상태 𝑠𝑡를 루트로 하는 탐색 트리를 구축하며, MDP의 모델을 사용하여 미래를 예측합니다. 전체 MDP를 해결할 필요 없이 현재 시점에서 시작하는 하위 MDP(sub MDP)만 해결하면 됩니다.   
  
Forward Search 알고리즘은 현재 상태를 루트로 하는 탐색 트리를 구축하며, 미래의 가능한 상태와 행동을 예측합니다. 이를 통해 더 나은 행동을 선택할 수 있습니다. 전방 탐색은 MDP의 모델을 사용하여 미래를 예측하므로, 미래 상태와 보상을 미리 계산할 수 있습니다.   
  
전방 탐색은 전체 MDP를 해결할 필요 없이 현재 상태에서 시작하는 하위 MDP만 해결하면 되기 때문에 계산 비용을 줄일 수 있습니다. 이는 효율적인 학습과 결정을 가능하게 합니다.

![](https://blog.kakaocdn.net/dna/7B2WU/btsiPi180Xj/AAAAAAAAAAAAAAAAAAAAAB_wj5JIEbY5cgQB9pt6CLAbBOXMdk71nJgI6jEQcYyu/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Yi97bXuQ7v7Xv5WDwe9HizpQJp8%3D)

Forward&nbsp;Search

**Simulation-Based Search**

Simulation Based Search은 샘플 기반 계획을 활용한 전방 탐색 패러다임입니다. 모델을 사용하여 현재로부터 경험의 에피소드를 시뮬레이션하고, 시뮬레이션된 에피소드에 Model-free RL을 적용합니다.   
  
Simulation Based Search은 모델을 사용하여 현재 시점에서부터 경험의 에피소드를 시뮬레이션합니다. 이를 통해 모델로부터 생성된 경험 데이터를 활용하여 Model-free RL을 적용합니다. 예를 들어, Monte Carlo 제어 알고리즘을 사용하여 Monte Carlo 탐색을 수행할 수 있습니다.   
  
시뮬레이션된 경험 데이터는 모델의 예측을 기반으로 생성되므로, 실제 환경에서 얻은 경험과는 다를 수 있습니다. 그러나 이러한 시뮬레이션 기반의 탐색은 계산 비용을 줄이고 효율적인 학습과 결정을 가능하게 합니다. 모델로부터 생성된 경험 데이터에 Model-free RL을 적용함으로써 최적의 정책을 학습하고 결정할 수 있습니다.

![](https://blog.kakaocdn.net/dna/erWry1/btsiFMQ8Xsy/AAAAAAAAAAAAAAAAAAAAAG3kUE1SKmtqWNs4TVOsEawq-qOzWmxKDFwwtzC_gogq/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=SKZCszb%2BM4VqO72jtA1h%2Bu%2BH3m0%3D)

Simulation Based Search은 전방 탐색을 기반으로 하며, 모델을 사용하여 경험의 에피소드를 시뮬레이션하고, 이를 Model-free RL에 적용함으로써 최적의 결정을 내릴 수 있는 방법입니다.

![](https://blog.kakaocdn.net/dna/cADMZQ/btsiPiVrPOc/AAAAAAAAAAAAAAAAAAAAAFuJgTiGFrZLmycfDYgRx94LgtcTPxq-jQ1qA6JxxddF/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=GtOL5YDLxtwKKcKfYiMQFpxZQ3g%3D)

Simulation&nbsp;Based&nbsp;Search

**Simple Monte-Carlo Search**

Simple Monte Carlo Search은 주어진 모델 𝑀𝑣와 시뮬레이션 정책 𝜋를 기반으로 합니다. 각 행동 𝑎 ∈ 𝐴에 대해 다음과 같은 과정을 수행합니다.   
  
먼저, 현재 상태 𝑠𝑡에서 시작하여 𝐾개의 에피소드를 시뮬레이션합니다. 실제 상태에서부터 모델을 사용하여 𝐾개의 에피소드를 생성합니다.

![](https://blog.kakaocdn.net/dna/bMMU1s/btsiwlmyEqI/AAAAAAAAAAAAAAAAAAAAAJ4NgVcfpDTA5f01-wDPmwa4SFovsFrP9KvMkiFjn72g/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=1lxak5t%2BSjLjVWuzp%2B6DFWAnhdM%3D)

그 후, 각 에피소드의 반환값의 평균을 계산하여 각 행동의 가치를 평가합니다. 이것을 Monte Carlo 평가라고 합니다. 가치 평가를 통해 현재 상태에서 각 행동의 예상 이익을 추정할 수 있습니다.

![](https://blog.kakaocdn.net/dna/c6bmVV/btsiBNCQ7Lj/AAAAAAAAAAAAAAAAAAAAAJdlBRe_-JMv-TQWfkU7-Gy3W1VsncLGT9LW8STOm_wc/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=QI1mdHmju%2FBpk1Mb75ha4TIXB40%3D)

마지막으로, 가치 평가를 통해 각 행동의 예상 이익을 추정한 후, 이 중 가장 높은 가치를 갖는 행동을 선택합니다. 이렇게 선택된 행동을 현재 상태에서 취하는 것이 Simple Monte Carlo Search의 결과입니다.

![](https://blog.kakaocdn.net/dna/DC8U8/btsiyWmquSa/AAAAAAAAAAAAAAAAAAAAAMpGijBuMHMmeZIgsHde1V0_vdi_gqkqClPsGeZmtF49/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=NgJ0wRH4Yfoapup9ksdgCPyHIs0%3D)

Simple Monte Carlo Search은 주어진 모델과 시뮬레이션 정책을 사용하여 가치 평가를 수행하고, 최적의 행동을 선택합니다. 이를 통해 모델 기반 RL에서 효과적인 학습과 결정을 가능하게 합니다.

**Monte-Carlo Tree Search (Evaluation)**

Monte-Carlo Tree Search는 주어진 모델 𝑀𝑣을 활용하여 트리를 구축하고, 각 상태와 행동의 가치를 평가하는 방법입니다. 이를 통해 최적의 행동을 선택합니다. 이제 각 단계에 대해 자세히 알아보겠습니다.   
  
먼저, 현재 상태 𝑠𝑡에서 시작하여 𝐾개의 에피소드를 시뮬레이션합니다. 현재 상태에서부터 모델을 사용하여 𝐾개의 에피소드를 생성합니다. 그리고 방문한 상태와 행동을 포함하는 탐색 트리를 구축합니다.

![](https://blog.kakaocdn.net/dna/dKs4yh/btsiJ7Hnk24/AAAAAAAAAAAAAAAAAAAAAJ0y7stRy-gxevtKiib0T8cDEMr-Bc9z95wyQw9ZmlZg/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=5iYGiXvjWY9jBvQKla89infrl74%3D)

다음으로, 구축된 탐색 트리를 기반으로 상태와 행동의 가치를 평가합니다. 각 상태-행동 쌍의 가치는 해당 상태에서 해당 행동을 선택하여 시작한 에피소드의 반환값의 평균으로 계산됩니다. 이를 Monte Carlo 평가라고 합니다. 가치 평가를 통해 상태와 행동의 예상 이익을 추정할 수 있습니다.

![](https://blog.kakaocdn.net/dna/FS3FM/btsiOjgfhUm/AAAAAAAAAAAAAAAAAAAAAD937pHPOjzH6NC5TU5IHhoaa_7TrYonV6NM6tLgHh4q/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=7aBNnGUpth2VlSI%2BzFobIne126c%3D)

마지막으로, 탐색이 완료된 후, 가치 평가를 통해 추정된 상태-행동 가치를 기반으로 현재 상태에서 가장 높은 가치를 가지는 행동을 선택합니다. 이를 통해 Monte-Carlo Tree Search (Evaluation)의 결과로서 현재 상태에서의 최적의 행동을 결정할 수 있습니다.

![](https://blog.kakaocdn.net/dna/ACpy5/btsiOHgUES1/AAAAAAAAAAAAAAAAAAAAAMhR4hGdWedym6cwdZHNGht1x84zc7cO31hL3mh4tLHX/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=L5%2FBeMLg6SPccd0wGqFygksyZ%2Fc%3D)

Monte-Carlo Tree Search는 주어진 모델을 활용하여 트리를 구축하고 가치 평가를 통해 최적의 행동을 선택합니다. 이를 통해 모델 기반 RL에서 효과적인 학습과 결정을 가능하게 합니다.

**Monte-Carlo Tree Search (Simulation)**

Monte-Carlo Tree Search에서는 시뮬레이션 정책 𝜋이 개선됩니다. 각 시뮬레이션은 두 단계로 구성됩니다. 이제 각 단계에 대해 자세히 알아보겠습니다.   
  
먼저, 시뮬레이션에서는 트리 정책(Tree policy)와 기본/롤아웃 정책(Default/Rollout policy)의 두 가지 단계로 진행됩니다. 트리 정책은 𝑄(𝑆,𝐴)를 최대화하는 행동을 선택하는 단계입니다. 기본/롤아웃 정책은 무작위로 행동을 선택하거나 다른 정책을 사용하는 단계입니다.   
  
각 시뮬레이션마다 다음을 반복합니다. 먼저, Monte Carlo 평가를 통해 상태-행동 가치를 평가합니다. 이를 통해 𝑄(𝑆,𝐴)를 추정할 수 있습니다. 그런 다음 트리 정책을 개선합니다. 예를 들어, 𝜖-greedy(Q)와 같은 방식으로 트리 정책을 개선할 수 있습니다.   
  
Monte-Carlo Tree Search에서는 시뮬레이션 정책을 개선하면서 트리 정책과 기본/롤아웃 정책을 사용하여 트리를 탐색하고 가치를 평가합니다. 이를 통해 최적의 행동을 선택하는 과정을 수행합니다.
