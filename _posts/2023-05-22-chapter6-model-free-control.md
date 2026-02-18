---
title: "Chapter6. Model-Free Control"
date: 2023-05-22 00:02:43
tags:
  - Model-Free Control
---

**Model-free Reinforcement Learning**

모델-프리 강화학습은 알려진 MDP(Model) 없이 가치 함수를 최적화하는 방법을 의미합니다. 이를 통해 어떻게 더 나은 정책을 학습할 수 있는지 알아보겠습니다.   
  
모델-프리 강화학습에서의 모델-프리 제어 (개선)은 다음과 같은 과정을 거칩니다:   
  
1. 현재의 정책에 따라 가치 함수를 추정합니다. 이를 통해 현재 정책의 성능을 알 수 있습니다.   
2. 추정된 가치 함수를 기반으로, 정책을 개선합니다. 개선된 정책은 더 높은 보상을 얻을 수 있는 방향으로 조정됩니다.   
3. 새로운 정책을 기반으로 가치 함수를 다시 추정하고, 정책 개선을 반복합니다. 이 과정을 반복하면서 점차적으로 더 나은 정책을 학습합니다.   
**모델-프리 강화학습은 MDP의 모델 정보가 없는 상황에서도 학습이 가능하며, 현재 정책을 개선하여 점진적으로 더 나은 정책을 학습할 수 있습니다.** 이를 통해 모델에 대한 사전 지식 없이도 최적의 정책을 학습할 수 있습니다.

**Uses of Model-Free Control**

모델-프리 강화학습은 다양한 응용 분야를 MDP(Markov Decision Process)로 모델링할 수 있습니다. 이러한 응용 분야에는 백감몬, 바둑, 로봇 이동, 헬리콥터 비행, 로보컵 축구, 자율 주행, 광고 선택, 환자 치료 등이 있습니다.   
  
이러한 문제들 중에서도 대부분은 다음과 같은 상황에서 사용됩니다:   
  
1. MDP 모델은 알려져 있지 않지만, 경험을 통해 샘플링할 수 있는 경우   
2. MDP 모델은 알려져 있지만 직접 사용하기에는 계산적으로 불가능한 경우 (샘플링을 통해서만 활용 가능한 경우)   
이러한 문제들을 해결하기 위해서 모델-프리 강화학습이 사용됩니다. 모델-프리 강화학습은 MDP 모델에 대한 사전 지식 없이도 경험을 통해 학습하고 문제를 해결할 수 있는 강력한 도구입니다.

**On and Off-Policy Learning**

온-폴리시 학습은 직접적인 경험을 통해 학습하는 방식입니다. 이는 정책 𝜋를 따르며 얻은 경험을 통해 정책 𝜋의 추정치를 학습하고 평가하는 것을 의미합니다. 즉, 현재 정책을 따르면서 경험을 수집하고 이를 기반으로 정책을 학습하는 방식입니다.   
  
반면, 오프-폴리시 학습은 다른 정책 𝜋′를 따르며 수집한 경험을 사용하여 정책 𝜋의 추정치를 학습하고 평가하는 방식입니다. 즉, 현재 정책과는 상관없는 다른 정책을 따라 경험을 수집하고, 이를 기반으로 원하는 정책을 학습하는 방식입니다.   
  
온-폴리시 학습은 직접적으로 현재 정책을 따르면서 학습하기 때문에 학습과정이 안정적이고 직관적입니다. 하지만, 오프-폴리시 학습은 다른 정책을 사용하기 때문에 초기에는 추정치의 변동성이 크지만, 장기적으로는 좀 더 안정적이고 효율적인 학습이 가능합니다.

**Generalized Policy Iteration - 이전에 설명했습니다.**

![](https://blog.kakaocdn.net/dna/cdYnLt/btsgEgAAOBn/AAAAAAAAAAAAAAAAAAAAAGa0pajLKSIZGo_z7IcVNmCo7RM65aNQFBYfFS8PPnRw/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=XnR%2BZXmpPGePLTUZwJ7VU%2B8FLg8%3D)

**Generalized Policy Iteration with MC Evaluation**

일반화된 정책 이터레이션은 다음과 같은 과정을 거칩니다:   
  
1. 초기 정책을 설정합니다.   
2. MC 평가를 사용하여 현재 정책의 가치 함수를 추정합니다. 이는 여러 에피소드의 경험을 통해 직접적으로 계산됩니다.   
3. 가치 함수를 기반으로 현재 정책을 개선합니다. 개선된 정책은 더 높은 가치를 가진 행동을 선택하도록 조정됩니다.   
4. 개선된 정책을 기반으로 다시 MC 평가를 수행합니다. 이를 통해 가치 함수를 업데이트합니다.   
5. 2단계부터 4단계까지를 반복하면서 점진적으로 더 나은 정책과 가치 함수를 학습합니다.

![](https://blog.kakaocdn.net/dna/Hao0v/btsgEbljUzl/AAAAAAAAAAAAAAAAAAAAAADl8FhVezG2y_RE9Kuj735xrGQn45ZKF74ASUvDQF-3/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=QenwwXKu5kK2dUUWZh6%2FAR4X0Pg%3D)

이러한 과정은 초기 정책을 기반으로 MC 평가를 수행하고, 그 결과를 토대로 정책을 개선하는 방식으로 진행됩니다. 이렇게 일반화된 정책 이터레이션은 정책과 가치 함수를 상호간에 조정하며 점진적으로 최적의 정책을 학습하는 방법입니다.

**Model-Free Policy Improvement**

모델-프리 정책 개선에서는 𝑉(𝑠)를 기준으로 한 탐욕적 정책 개선은 MDP 모델을 요구합니다. 즉, 상태-가치 함수 𝑉(𝑠)를 기반으로 더 높은 가치를 가진 행동을 선택하는 정책을 개선하려면 MDP의 모델 정보가 필요합니다.

![](https://blog.kakaocdn.net/dna/BOfRF/btsgQq2pKtT/AAAAAAAAAAAAAAAAAAAAADdUKJtRbP8j4HMUi1Q9XgZH9CuMJ6ShEQ066nf0xDIk/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=gUK%2BGqXv5pRGYrt1IiAkzRXGr4E%3D)

반면에, 𝑄(𝑠,𝑎)를 기준으로 한 탐욕적 정책 개선은 모델-프리 방식으로 진행됩니다. 즉, 행동-가치 함수 𝑄(𝑠,𝑎)를 기반으로 더 높은 가치를 가진 행동을 선택하여 정책을 개선하는 방식입니다. 이는 MDP의 모델 정보 없이도 진행할 수 있으므로 모델-프리 방식으로 정책 개선이 가능합니다.

![](https://blog.kakaocdn.net/dna/bBL4bp/btsgDK9jIqy/AAAAAAAAAAAAAAAAAAAAAG1yw8U5jYU_AoHfHima12IPBKAu7F_9hA31AgQRIODa/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=SHOpiUngODcknZuMOM%2Fruw%2BqGLY%3D)

**따라서, 모델-프리 정책 개선에서는 𝑄(𝑠,𝑎)를 사용하여 탐욕적으로 정책을 개선함으로써 최적의 정책을 찾을 수 있습니다.**

**Generalized Policy Iteration with Q-Function**

일반화된 정책 이터레이션에서 Q-함수를 사용한 방법은 다음과 같은 과정을 거칩니다:   
  
1. 초기 정책을 설정합니다.   
2. Q-함수를 사용하여 현재 정책에 대한 가치 함수를 추정합니다. 이는 여러 에피소드의 경험을 통해 직접적으로 계산됩니다.   
3. 가치 함수를 기반으로 현재 정책을 개선합니다. 개선된 정책은 Q-함수를 기준으로 더 높은 가치를 가진 행동을 선택하도록 조정됩니다.   
4. 개선된 정책을 기반으로 다시 Q-함수를 추정합니다. 이를 통해 가치 함수를 업데이트합니다.   
5. 2단계부터 4단계까지를 반복하면서 점진적으로 더 나은 정책과 가치 함수를 학습합니다.

![](https://blog.kakaocdn.net/dna/cuIpOx/btsgDMeYvFO/AAAAAAAAAAAAAAAAAAAAAJnASqC_GMzOIThFWHMFlkmWPPzjNIMMeCsiYfbHe8F1/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=CCDv3N8LGcFZG%2BLX8HIvm75RT4o%3D)

이러한 과정은 초기 정책을 기반으로 Q-함수를 사용하여 가치 함수를 추정하고, 그 결과를 토대로 정책을 개선하는 방식으로 진행됩니다. 이렇게 일반화된 정책 이터레이션은 Q-함수와 정책을 상호간에 조정하며 점진적으로 최적의 정책을 학습하는 방법입니다.

Example of Greedy Action Selection

![](https://blog.kakaocdn.net/dna/b6TFOr/btsgTiiLxYc/AAAAAAAAAAAAAAAAAAAAALrqFBtKOWxpiTI0EbprzZN-ZMU7Zri6y7y5T98YEB8E/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=o19KOua58kPaKdj3aHC%2BySdK%2Bt8%3D)

주어진 예시에서는 숫자들이 나열되어 있습니다. 각 줄은 특정 상황에서의 보상(return)을 나타냅니다. 이를 기반으로 탐욕적 행동 선택을 수행해보겠습니다.   
  
첫 번째 줄부터 시작합니다. 처음에는 보상이 1이므로 현재까지의 최고 보상을 1로 설정합니다. 다음 줄부터는 현재까지의 최고 보상과 비교하여 더 큰 보상이 나오면 최고 보상을 업데이트합니다. 이를 반복하여 최고 보상을 구하고 해당하는 행동을 선택합니다.   
  
두 번째 예시에서도 같은 과정을 수행합니다. 하지만 두 번째 예시에서는 보상 100이 나오는 순간 최고 보상이 업데이트되어야 합니다.   
  
이렇게 탐욕적 행동 선택은 주어진 보상들 중에서 가장 큰 보상을 가진 행동을 선택하는 방식입니다. 따라서 왼쪽문이 최고 보상이 업데이트되는 순간 열릴 것입니다.

**Exploration and Exploitation**

강화학습은 시행착오 학습(trial-and-error learning)과 유사합니다. 에이전트는 환경과의 상호작용을 통해 좋은 정책을 발견해야 하지만 동시에 보상을 크게 잃지 않아야 합니다.   
  
**탐색(Exploration)은 환경에 대해 더 많은 정보를 찾는 과정입니다**. 에이전트는 새로운 경험을 얻기 위해 다양한 행동을 시도하고, 미지의 영역을 탐험합니다.   
  
반면, **활용(Exploitation)은 알려진 정보를 활용하여 보상을 극대화하는 과정입니다.** 에이전트는 이미 알고 있는 좋은 정책에 따라 행동하며, 최대한 많은 보상을 얻기 위해 이전에 배웠던 지식을 활용합니다.   
  
일반적으로, 탐색과 활용은 모두 중요합니다. 탐색을 통해 미지의 영역을 탐험하고 새로운 정보를 얻을 수 있지만, 동시에 활용을 통해 이미 알고 있는 좋은 정책에 따라 보상을 극대화할 수 있습니다.

**𝝐-Greedy Exploration**

𝝐-탐욕적 탐색은 지속적인 탐색을 보장하기 위한 가장 간단한 아이디어입니다. 이를 위해 모든 가능한 행동을 0 이상의 확률로 시도합니다. 그리고 1-𝝐의 확률로 탐욕적 행동을 선택하고, 𝝐의 확률로 무작위로 행동을 선택합니다.

![](https://blog.kakaocdn.net/dna/bgVErk/btsgFvRhHSS/AAAAAAAAAAAAAAAAAAAAACIjgEuoHgzfv0s0NAw6LN0NE-TIkeYClBQA_pL-N-QI/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=yjlUbOCKsy9OHUCWLZTQmS%2BJuZM%3D)

이때, m은 가능한 행동의 수를 나타냅니다.  
𝝐의 값은 보통 0과 1 사이의 작은 값을 사용합니다. 예를 들어, 𝝐=0.1이라고 가정해봅시다. 이때, 𝝐-탐욕적 탐색은 다음과 같이 구현될 수 있습니다:

![](https://blog.kakaocdn.net/dna/bdtzpK/btsgMcDpID0/AAAAAAAAAAAAAAAAAAAAAJwrPneJ6AjEIrWcyXpbL736OFXMmq5x5ZFipa7I5vTU/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=b5jfiXjxq7Xbbgl0vUoI6GG5FiI%3D)

𝝐-탐욕적 탐색은 탐색과 활용의 균형을 유지하면서도 지속적인 탐색을 보장할 수 있는 방법 중 하나입니다. 알고 있는 최적의 행동을 선택하는 활용과 새로운 행동을 탐색하는 탐색을 조절하여 최적의 정책을 찾을 수 있습니다.

**𝝐-Greedy Policy Improvement**

𝝐-탐욕적 정책 개선은 탐욕적 행동 선택(𝝐-Greedy Exploration)과 관련된 개념입니다. 이를 통해 현재의 정책을 개선하여 더 나은 정책을 찾아나갑니다.   
  
𝝐-탐욕적 정책 개선은 다음과 같은 방식으로 수행됩니다:

![](https://blog.kakaocdn.net/dna/Vj1pR/btsgTjIKY1p/AAAAAAAAAAAAAAAAAAAAAEZFBqL6E7xDaKqD2gSWuIq7JxU6Zp_q5fp41ANQxbnt/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=rFortyD8w%2BlzyiekFVtEgUeU9kA%3D)

1. 각 상태에서 가능한 모든 행동의 가치를 추정합니다.   
2. 𝝐-탐욕적 탐색을 통해 탐색과 활용을 조절하면서 가장 큰 가치를 가진 행동을 선택합니다. 즉, 𝝐의 확률로 무작위 행동을 선택하고, 1-𝝐의 확률로 가치가 가장 큰 행동을 선택합니다.   
3. 선택된 행동을 현재의 정책으로 업데이트합니다.   
4. 1단계부터 3단계까지를 반복하면서 점진적으로 더 나은 정책을 학습합니다.

𝝐-탐욕적 정책 개선은 탐색과 활용을 적절히 조절하면서 현재의 정책을 개선하는 방법입니다. 𝝐-탐욕적 정책 개선을 통해 보다 최적의 정책을 발견하고, 학습된 가치 함수를 기반으로 탐욕적으로 행동을 선택할 수 있습니다.

**(Model-Free) Monte-Carlo Policy Iteration**

몬테카를로 정책 이터레이션은 모델-프리 강화학습의 한 방법으로, 정책 개선과 가치 평가를 번갈아가며 수행합니다.

![](https://blog.kakaocdn.net/dna/5DOy9/btsgC2WG0F9/AAAAAAAAAAAAAAAAAAAAACzMK32CUSIE648WTY6AWgQYP5HQgnabeAYIQa7zoiVd/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=t40LcZ7HogkFPCrhl%2FS%2B7EXDPeo%3D)

1. 초기 정책을 설정합니다.   
2. 정책을 따라 에피소드를 생성합니다. 에피소드는 상태, 행동, 보상의 연속적인 시퀀스로 구성됩니다.   
3. 에피소드에서 얻은 보상을 이용하여 각 상태-행동 쌍에 대한 가치를 추정합니다. 이는 몬테카를로 방법을 통해 직접적으로 계산됩니다.   
4. 가치 함수를 기반으로 정책을 개선합니다. 개선된 정책은 상태별로 가치가 더 높은 행동을 선택하도록 조정됩니다.   
5. 2단계부터 4단계까지를 반복하면서 점진적으로 더 나은 정책과 가치 함수를 학습합니다.

몬테카를로 정책 이터레이션은 에피소드로부터 직접적으로 가치 함수를 추정하는 방식으로 정책을 학습합니다. 이를 통해 보상을 최대화하는 최적의 정책을 찾을 수 있습니다.

**Variant Monte-Carlo Policy Iteration**

변형된 몬테카를로 정책 이터레이션은 몬테카를로 정책 이터레이션의 변형된 형태로, 가치 함수의 추정에 활용되는 방법을 조금 변경한 것입니다.

![](https://blog.kakaocdn.net/dna/bfIdUE/btsgEKAW2GI/AAAAAAAAAAAAAAAAAAAAANmgW2z8uhoheKgJ8PoKaE1H4ga21B2UfHo1pgOG9r-Y/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=nWpxEt8bPUQPpwi1dbROWY%2FEg9w%3D)

1. 초기 정책을 설정합니다.   
2. 정책을 따라 에피소드를 생성합니다. 에피소드는 상태, 행동, 보상의 연속적인 시퀀스로 구성됩니다.   
3. 에피소드에서 얻은 보상을 이용하여 각 상태-행동 쌍에 대한 가치를 추정합니다. 이는 몬테카를로 방법을 통해 직접적으로 계산됩니다.   
4. 추정된 가치 함수를 기반으로 정책을 개선합니다. 개선된 정책은 상태별로 가치가 더 높은 행동을 선택하도록 조정됩니다.   
5. 2단계부터 4단계까지를 반복하면서 점진적으로 더 나은 정책과 가치 함수를 학습합니다.

**변형된 몬테카를로 정책 이터레이션은 몬테카를로 방법을 통해 가치 함수를 추정하는 점은 기존의 몬테카를로 정책 이터레이션과 동일하지만, 가치 함수를 기반으로 정책을 개선하는 방식에 일부 변화가 있을 수 있습니다.**이는 보다 효율적인 학습을 위한 변형된 방법으로 사용될 수 있습니다.

**Greedy in the Limit with Infinite Exploration (GLIE)**

 "무한 탐색을 통한 극한에서의 탐욕 정책(GLIE: Greedy in the Limit with Infinite Exploration)"는 강화학습에서 사용되는 하나의 정책 개선 전략입니다. 이 전략은 최적의 정책을 찾기 위해 탐색과 활용을 균형있게 조절하면서 지속적인 개선을 추구합니다.   
  
GLIE 전략은 다음과 같은 방식으로 수행됩니다:

![](https://blog.kakaocdn.net/dna/oHRCJ/btsgDNdU4p9/AAAAAAAAAAAAAAAAAAAAABs7uJSAqkGnYrYiFQPPKdwFYKx16kSJwfeASvPSangI/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=dT%2BlQvSZG9eVf2LC6e4pjMjBEKs%3D)

1. 초기에는 모든 가능한 행동을 무작위로 선택하는 무작위 정책을 사용합니다. 이는 무한한 탐색을 보장하기 위한 초기화 단계입니다.   
2. 에이전트는 경험을 통해 가치 함수나 행동 가치를 추정하고, 이를 기반으로 정책을 개선합니다.   
3. 개선된 정책은 탐욕적으로 가치가 가장 높은 행동을 선택하도록 업데이트됩니다. 하지만 확률적으로 일정 비율의 무작위 행동을 선택하는 것을 포함하여 계속해서 탐색을 유지합니다.   
4. 탐색과 활용을 조절하며 정책을 개선하는 과정을 반복합니다.   
5. 점진적으로 더 나은 정책을 학습하고, 행동 가치나 가치 함수가 수렴할 때까지 반복합니다.

GLIE 전략은 초기에는 무작위로 탐색하면서 점진적으로 활용을 늘려가며 최적의 정책을 발견하는 전략입니다. 이를 통해 무한 탐색을 보장하면서 극한에서 최적의 정책을 찾을 수 있습니다.

**Monte-Carlo Control**

몬테카를로 제어는 강화학습의 한 방법으로, 모델-프리 방법 중 하나입니다. 다음과 같은 과정으로 수행됩니다:

![](https://blog.kakaocdn.net/dna/bxPbGd/btsgEdKCqUN/AAAAAAAAAAAAAAAAAAAAAJDRy-RTbhWP6yNg-zDuzKPbXi-fuOqDTJP1kQQDU3Db/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hgBy9kNjWlEo2VU8PbS0iKZeryU%3D)

1. 정책 𝜋를 기반으로 에피소드를 생성합니다. 에피소드는 상태, 행동, 보상의 연속적인 시퀀스로 구성됩니다.   
2. 생성된 에피소드를 사용하여 각 상태-행동 쌍에 대한 행동 가치를 추정합니다. 이를 위해 상태-행동 쌍이 등장한 횟수를 카운트하고, 이를 이용하여 행동 가치를 업데이트합니다.   
3. 행동 가치 함수를 기반으로 정책을 개선합니다. 개선된 정책은 각 상태에서 가치가 가장 높은 행동을 선택하도록 조정됩니다.   
4. 1단계부터 3단계까지를 반복하면서 점진적으로 더 나은 정책과 행동 가치 함수를 학습합니다.

몬테카를로 제어는 에피소드를 통해 직접적으로 행동 가치를 추정하고, 이를 기반으로 정책을 개선하는 방식으로 학습합니다. 이를 통해 보상을 최대화하는 최적의 정책을 찾을 수 있습니다.

![](https://blog.kakaocdn.net/dna/JkSk3/btsgEKVji6t/AAAAAAAAAAAAAAAAAAAAAGQG1GGqSmVlO6RdPyyFclHDuRlO4I6Kyz2o1W7XH3w8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=dRwRiAunnuR3qesxf1M4eWrl0j8%3D)

Monte-Carlo Control in Blackjack

**MC vs. TD Control**

TD 제어는 몬테카를로 제어와 비교하여 몇 가지 장점을 가지고 있습니다.   
  
1. 낮은 분산: TD 제어는 몬테카를로 제어보다 분산이 낮습니다. 이는 더 안정적인 학습을 가능하게 합니다.   
2. 온라인 학습: TD 제어는 온라인 학습으로 진행됩니다. 즉, 매 타임 스텝마다 업데이트를 수행하여 실시간으로 학습이 가능합니다.   
3. 불완전한 시퀀스: TD 제어는 불완전한 시퀀스에서도 학습이 가능합니다. 에피소드의 끝까지 기다리지 않고 중간 결과로부터 학습을 진행할 수 있습니다.

따라서 자연스럽게 TD 제어를 몬테카를로 제어 대신 사용할 수 있습니다. 이 경우 TD 제어를 𝑄(𝑆,𝐴)에 적용하고, 𝜖-탐욕적인 정책 개선을 사용합니다. 또한 매 타임 스텝마다 업데이트를 수행합니다.   
  
TD 제어는 몬테카를로 제어와 비교하여 장점을 가지고 있으며, 더 효율적인 학습이 가능합니다.

**Model-free Policy Iteration with TD Methods**

모델-프리 정책 개선에서는 시간 차(TD) 메소드를 정책 평가 단계에 활용합니다. 다음과 같은 단계로 진행됩니다:   
  
1. 정책 𝜋를 초기화합니다.   
2. 반복적으로 다음을 수행합니다:   
     - 정책 평가: 𝜖-탐욕적인 정책에 대해 시간 차 업데이트를 사용하여 𝑄𝜋를 계산합니다.   
     - 정책 개선: 몬테카를로 정책 개선과 동일하게 𝜋를 𝜖-탐욕적인 정책에 대입합니다. 즉, 𝜋를 𝜖-탐욕적인 정책으로 업데이         트합니다.   
이와 같이 TD 메소드를 활용하여 모델-프리 정책 개선을 수행합니다. 이는 정책 평가 단계에서 시간 차 업데이트를 사용하고, 정책 개선 단계에서는 몬테카를로 정책 개선과 동일한 방식으로 정책을 업데이트합니다.   
  
모델-프리 정책 개선에서 TD 메소드를 사용하는 것은 보다 효율적인 학습을 가능하게 합니다.

**Updating Action-Value Functions with SARSA**

SARSA는 상태-행동-보상-다음 상태-다음 행동(SARSA)을 나타내는 시퀀스를 사용하여 행동 가치 함수를 업데이트하는 방법입니다. 다음과 같은 단계로 진행됩니다:

![](https://blog.kakaocdn.net/dna/pNBCU/btsgMbYMLVu/AAAAAAAAAAAAAAAAAAAAAFDzAQlYaPCn-BaTfR-S9fv-e3RCsLYLV0lnnUrkMcoF/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=3qeh5rZwLl%2BgiRSoAcLWuM013iY%3D)

1. 초기 상태와 행동을 설정합니다.   
2. 다음 상태로 이동하고 다음 행동을 선택합니다.   
3. 선택한 행동에 대한 보상을 받습니다.   
4. 다음 상태와 행동을 사용하여 행동 가치 함수를 업데이트합니다. 이를 SARSA 업데이트라고도 합니다.   
- 𝑄(𝑆,𝐴)←𝑄(𝑆,𝐴)+𝛼[𝑅+𝛾𝑄(𝑆',𝐴')−𝑄(𝑆,𝐴)]   
  (여기서 𝛼는 학습률, 𝛾는 할인율입니다.)   
5. 다음 상태와 행동을 현재 상태와 행동으로 설정하고 2단계부터 다시 반복합니다.

SARSA를 사용하여 행동 가치 함수를 업데이트하는 과정에서는 현재 상태와 행동, 다음 상태와 행동, 그리고 보상을 사용합니다. 이를 통해 에이전트는 경험을 통해 행동 가치 함수를 학습하고 최적의 행동을 선택할 수 있습니다.

**Policy Control with SARSA**

SARSA를 사용한 정책 제어는 다음과 같은 과정으로 이루어집니다:

![](https://blog.kakaocdn.net/dna/brGScC/btsgL9fB1wk/AAAAAAAAAAAAAAAAAAAAAAJet27mlWmxJLuVGyMwKPHvRLh4OgbmVlgQU3KHyiMm/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=wexm7QnuSZxVvmfqWdYjfQzm44g%3D)

매 타임 스텝마다 다음을 수행합니다:   
1. 정책 평가: SARSA를 사용하여 행동 가치 함수 Q를 근사합니다. 이를 𝑞𝜋에 근사한 것으로 간주합니다.   
2. 정책 개선: 𝜖-탐욕적인 정책 개선을 수행합니다. 이를 통해 정책을 업데이트합니다.

정책 제어 과정에서는 SARSA를 사용하여 행동 가치 함수를 근사하고, 이를 바탕으로 𝜖-탐욕적인 정책 개선을 수행합니다. 이를 반복하면서 정책을 점진적으로 개선해나갈 수 있습니다.   
  
SARSA를 사용한 정책 제어는 매 타임 스텝마다 정책을 평가하고 개선함으로써 최적의 행동을 학습합니다. 이를 통해 에이전트는 경험을 통해 최적의 정책을 찾아나갈 수 있습니다.

**SARSA Algorithm**

![](https://blog.kakaocdn.net/dna/D0uTI/btsgC4fTRra/AAAAAAAAAAAAAAAAAAAAABBgC8ZgxACL3JxkjpCqFzslw4CM-8CvXbaLRG5OYUe_/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=EqZgW1SHc6AnEBn0pjYHK2Up97Q%3D)

SARSA는 상태-행동-보상-다음 상태-다음 행동(SARSA)을 나타내는 시퀀스를 사용하여 정책을 개선하는 알고리즘입니다. 다음은 SARSA 알고리즘의 단계별 설명입니다:

1. 초기 상태 𝑆를 설정하고 초기 행동 𝐴를 선택합니다.
2. 다음 상태 𝑆'로 이동하고 다음 행동 𝐴'를 선택합니다.
3. 선택한 행동에 대한 보상 𝑅을 받습니다.
4. 다음 상태와 행동을 사용하여 행동 가치 함수 𝑄를 업데이트합니다.
   - 𝑄(𝑆,𝐴) ← 𝑄(𝑆,𝐴) + 𝛼[𝑅 + 𝛾𝑄(𝑆',𝐴') - 𝑄(𝑆,𝐴)] (여기서 𝛼는 학습률, 𝛾는 할인율입니다.)
5. 다음 상태 𝑆'와 행동 𝐴'를 현재 상태 𝑆와 행동 𝐴로 설정하고 2단계부터 다시 반복합니다.

SARSA 알고리즘은 상태와 행동의 시퀀스를 통해 정책을 개선하는 과정에서 행동 가치 함수를 업데이트합니다. 이를 통해 에이전트는 경험을 통해 최적의 정책을 학습하고 최적의 행동을 선택할 수 있습니다. (위에서 본 것과 동일하다.)

**Windy Gridworld Example**

Windy Gridworld는 격자 형태의 환경에서 에이전트가 목표 지점에 도달하기 위해 움직이는 문제입니다. 목표에 도달할 때까지 시간 단계마다 보상이 -1로 주어집니다. 다음은 이 예제에서 사용되는 파라미터입니다:

![](https://blog.kakaocdn.net/dna/FIh5E/btsgMbLgPgh/AAAAAAAAAAAAAAAAAAAAADvHfWuExuYMsgsF7v1G94XC7oQcdP7JV7r5V9hBemfN/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=NieOmlcO2C5z9SyLWOhMwucfjc0%3D)

- 𝜖: 0.1 (탐욕적 정책 개선에서의 탐색 비율)   
- 𝛼: 0.5 (학습률)   
- 할인 없음 (Undiscounted)

이 예제에서는 에이전트가 현재 위치에서 움직이는데에 영향을 주는 바람의 세기를 고려해야 합니다. 바람으로 인해 에이전트의 이동이 어려워지는 격자도 있을 수 있습니다.   
  
Windy Gridworld 예제는 정책 개선과 같은 강화학습 알고리즘을 통해 에이전트가 최적의 경로를 학습하고 목표 지점에 도달하는 방법을 탐구합니다.

![](https://blog.kakaocdn.net/dna/P6xA6/btsgEbyPk3O/AAAAAAAAAAAAAAAAAAAAAPtMxOP-8_UZaltt300uHAT059mgQ5vfRgRxwLBjNu3C/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=a%2BvpRgFWMPXtnKfGtxnDHL1H9Uo%3D)

SARSA on Windy Gridworld

**n-Step SARSA**

n-Step SARSA는 𝑛=1,2,..∞와 같은 다양한 n-스텝 반환(n-step returns)을 고려합니다. 이 때, n-스텝 Q-반환(n-step Q-return)을 정의하고, n-스텝 SARSA는 𝑄(𝑠,𝑎)를 n-스텝 Q-반환에 대해 업데이트합니다.

![](https://blog.kakaocdn.net/dna/bVRCuG/btsgEgACpWE/AAAAAAAAAAAAAAAAAAAAAKvYzHuCdC_lsUSPbHC0SBc9jxPRrLv_uBEVzZmVFBY7/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=nKYf8mDuECcwJuHogGGAgbY93Vk%3D)

n-스텝 반환은 에이전트가 현재 상태에서 n개의 시간 단계 동안 수집한 보상들을 합친 값입니다. 이를 이용하여 n-스텝 Q-반환을 계산하고, n-스텝 SARSA 알고리즘은 이 값을 활용하여 𝑄(𝑠,𝑎)를 업데이트합니다.

![](https://blog.kakaocdn.net/dna/dsBjGb/btsgMbxJUhj/AAAAAAAAAAAAAAAAAAAAABE-JgEhl-5CUnvJ5MEmAcC-BjLk8mUlO4zul29kEQB8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=aSgeSPSNFixfEEusTbPxqTXQcvE%3D)
![](https://blog.kakaocdn.net/dna/KsNcW/btsgG55SeMu/AAAAAAAAAAAAAAAAAAAAAJOH_mDKlWdGRSplciUmowxCNyw1SqAgbxlaSvQQmoYS/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=KZnWBJ3hKoEiZD8vIIACRBMmMYs%3D)

n-Step SARSA는 장기적인 시간 스케일을 고려하여 학습하는 데 유용하며, 𝑛의 값에 따라 다양한 시간 차이 간격을 고려할 수 있습니다.

**Off-Policy Learning**

오프-정책 학습은 목표 정책(𝜋(𝑎|𝑠))을 평가하기 위해 행동 정책(𝜇(𝑎|𝑠))을 따르면서 𝑣𝜋(𝑠) 또는 𝑞𝜋(𝑠,𝑎)를 계산합니다. 즉, 𝜇으로부터 생성된 경험 데이터인 {𝑆1,𝐴1,𝑅2,…,𝑆𝑇}을 사용하여 𝜋의 가치 함수를 평가합니다.   
  
이는 다음과 같은 이유로 중요합니다:  
• 사람이나 다른 에이전트를 관찰하면서 학습할 수 있습니다.   
• 이전 정책(𝜋1,𝜋2,..., 𝜋𝑡−1)에서 생성된 경험 데이터를 재사용할 수 있습니다.   
• 탐색적 정책을 따르면서 최적 정책에 대해 학습할 수 있습니다.   
• 하나의 정책을 따르면서 여러 정책에 대해 학습할 수 있습니다.

오프-정책 학습은 다양한 환경에서 유용하며, 다양한 정책에 대한 학습과 탐색을 동시에 수행할 수 있습니다.

**Importance Sampling**

중요도 샘플링은 다른 분포에서의 기댓값을 추정하는 기법입니다. 주어진 분포와 다른 분포에서의 확률을 이용하여 추정하고자 하는 분포의 기댓값을 계산합니다.

![](https://blog.kakaocdn.net/dna/bVTbGR/btsgUs6JbY3/AAAAAAAAAAAAAAAAAAAAAG32B8uV-V1qmsk6TBUlkD_AAzqNt4-udLHiiXk2zSQm/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=HIphgkeuUgPKLOjUkYoLWosCBIw%3D)

중요도 샘플링은 다음과 같은 상황에서 유용합니다:  
• 특정 분포에서의 샘플링이 어려운 경우   
• 기존 분포에 대한 정보가 충분히 있는 경우   
• 기존 분포에서의 샘플을 이용하여 다른 분포에서의 기댓값을 추정해야 할 때

중요도 샘플링은 강화학습과 같은 다양한 분야에서 활용되며, 다른 분포에서의 추정을 효과적으로 수행할 수 있습니다.

**Importance Sampling for Off-Policy Monte-Carlo**

오프-정책 몬테카를로에서 중요도 샘플링을 사용하면, 𝜇에서 생성된 반환값을 𝜋를 평가하는 데 사용할 수 있습니다. 중요도 샘플링은 두 정책 사이의 유사성에 따라 반환값을 가중치로 조정합니다. 전체 에피소드에 걸쳐 중요도 샘플링 보정을 곱하여 반환값을 보정한 후, 보정된 반환값을 기준으로 가치를 업데이트합니다.

![](https://blog.kakaocdn.net/dna/bbQBpD/btsgC32rifm/AAAAAAAAAAAAAAAAAAAAAE_QrGxT-l0XCslKy25mDG8MzB9uKv0JxLq7xg51cqiE/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=g4uOjQ90AcVp%2B6ch9XYWzCe2eQ4%3D)
![](https://blog.kakaocdn.net/dna/KqP7R/btsgGj3Yo3d/AAAAAAAAAAAAAAAAAAAAAHzWW4M4MGmC0ZT0GQM7f2c5fcdVPLX49nYNARtiVrlp/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=cz5L%2F27TPjTUtArOahicb1JI41c%3D)

중요도 샘플링은 𝜇이 0인 경우에는 사용할 수 없으며, 분산을 크게 증가시킬 수 있다는 단점이 있습니다.   
  
중요도 샘플링은 오프-정책 몬테카를로에서의 평가를 수행하는 데에 유용하게 사용될 수 있습니다.

**Importance Sampling for Off-Policy TD**

오프-정책 TD에서 중요도 샘플링을 사용하면, 𝜋를 평가하는 데에 TD 타겟을 활용할 수 있습니다. 중요도 샘플링을 이용하여 TD 타겟인 𝑅𝑡+1+𝛾𝑉𝑆𝑡+1을 가중치로 보정합니다. 단일 중요도 샘플링 보정만 필요하며, 다음과 같이 가치를 업데이트합니다:

![](https://blog.kakaocdn.net/dna/cFiGvp/btsgJ1B9jgk/AAAAAAAAAAAAAAAAAAAAACCFbnLrRQK0DjypBIzs-gVzRITguwp1jkbpn75uFurM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=TSD3JfTg8MAWHL9tJi3Xk2NHmcE%3D)

이 방법은 Monte-Carlo 중요도 샘플링에 비해 훨씬 낮은 분산을 가지므로 유용하게 활용될 수 있습니다.

**Q-Learning : Off-policy TD Control**

액션-가치 함수인 𝑄(𝑆,𝐴)의 오프-정책 TD 컨트롤인 Q-Learning에 대해 조금 더 알아보겠습니다.   
  
우리는 이제 행동 정책과 목표 정책 모두 개선을 허용합니다. 목표 정책 𝜋는 𝑄𝑠,𝑎에 대해 탐욕적으로 선택하는 정책입니다:

![](https://blog.kakaocdn.net/dna/upMTM/btsgG5Sn4CU/AAAAAAAAAAAAAAAAAAAAADRWZWigkcmabi-hxVSWk7DZDMkwsk7tmE5KaToCrGZJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=vaO3WhcsBaQgKQPypb%2B7WnHvjiw%3D)

반면에 행동 정책 𝜇는 예를 들어 𝑄𝑠,𝑎에 대해 𝜖-그리디 정책으로 선택됩니다:

![](https://blog.kakaocdn.net/dna/CjU2i/btsgL9s8EKr/AAAAAAAAAAAAAAAAAAAAAK3NKjGBz99E0UwDsXb7TTbFpdqlTXgnR8sKhxjMONkt/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=69YEt4dMa6DYGgnPxFHyTBFteZM%3D)

그러면 Q-Learning의 타깃은 다음과 같이 단순화됩니다:

![](https://blog.kakaocdn.net/dna/lRWtu/btsgGk9CLoo/AAAAAAAAAAAAAAAAAAAAANAg5x-0TT8uGTaIjvmycE2eslt8ktX29p9Av8lZfzdG/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=GDZBpfgLmszcI%2Bf0JAhiTQByXro%3D)

이 방법을 통해 𝑄(𝑆,𝐴)를 학습하면서 행동 정책과 목표 정책을 개선할 수 있습니다.

**Q-Learning Algorithm**

![](https://blog.kakaocdn.net/dna/bzLVDK/btsgEf2M4m1/AAAAAAAAAAAAAAAAAAAAAKTHvRSSvW_dEL0g4IhMSvuLIpAjlk1buIH8nCFVkgnv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=GN7HsqAkyCNJPGz2P%2FvmAGe3izo%3D)

Q-Learning 알고리즘은 오프-정책 TD 컨트롤의 한 예입니다. 이제 Q-Learning 알고리즘에 대해 자세히 알아보겠습니다.

1. Q-Value 테이블 초기화: 모든 상태-액션 쌍 (𝑠,𝑎)에 대한 Q-Value를 초기화합니다.
2. 반복 수행:
   - 현재 상태 𝑠를 초기 상태로 설정합니다.
   - 𝜖-그리디 정책에 따라 현재 상태에서 행동 𝑎를 선택합니다.
   - 선택한 행동으로 환경과 상호작용하여 다음 상태 𝑠'와 보상 𝑅을 받습니다.
   - 다음 상태에서 최대 Q-Value를 찾아서 𝑄(𝑠',𝑎')를 계산합니다.
   - Q-Value를 업데이트합니다: 𝑄(𝑠,𝑎) ← (1-𝛼)𝑄(𝑠,𝑎) + 𝛼(𝑅 + 𝛾𝑄(𝑠',𝑎') - 𝑄(𝑠,𝑎))
   - 현재 상태를 다음 상태로 업데이트합니다.
   - 종료 조건을 확인하여 알고리즘을 종료합니다.
3. 반환: 최종 학습된 Q-Value 테이블을 반환합니다.

Q-Learning은 행동 정책과 목표 정책을 개선하는 오프-정책 TD 컨트롤 알고리즘입니다. 이 알고리즘을 사용하여 최적의 행동 정책을 학습할 수 있습니다.

**SARSA vs. Q-Learning**  
SARSA

![](https://blog.kakaocdn.net/dna/cguOcS/btsgE7JTHiN/AAAAAAAAAAAAAAAAAAAAAM_SKDgrbHNR3Pz-bIqpyEBsRXVgBswsq0ZiEkluKu0p/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=VYUn2R57IVGwY3D1i7a1rce1ppQ%3D)

Q-Learning

![](https://blog.kakaocdn.net/dna/bVXCPk/btsgGiqrRyw/AAAAAAAAAAAAAAAAAAAAANg0SGAMn6x1Ed0bOSxhjFbwUDr5NQ700RQLOd42JBBx/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=nfR%2BdxbbNkhdyfKkyegxi0xqOyU%3D)

SARSA와 Q-Learning은 모두 모델 없는 강화 학습의 일종으로서, Q-Value를 학습하여 최적의 정책을 찾는 것을 목표로 합니다. 그러나 두 알고리즘은 약간의 차이점이 있습니다. 이제 SARSA와 Q-Learning의 차이점을 살펴보겠습니다.

1. SARSA (State-Action-Reward-State-Action)
   - SARSA는 현재 상태에서 선택한 행동과 다음 상태에서 선택한 행동의 쌍에 대해 Q-Value를 업데이트합니다.
   - SARSA는 현재 상태에서 선택한 행동으로부터 얻은 보상과 다음 상태에서 선택한 행동의 Q-Value를 사용하여 Q-Value를 업데이트합니다.
   - SARSA는 행동 정책과 목표 정책이 같은 경우, 즉 동일한 정책을 따르는 경우에 사용됩니다.
2. Q-Learning
   - Q-Learning은 현재 상태에서 선택한 행동과 다음 상태에서 최대 Q-Value를 사용하여 Q-Value를 업데이트합니다.
   - Q-Learning은 현재 상태에서 선택한 행동으로부터 얻은 보상과 다음 상태에서 가능한 모든 행동의 최대 Q-Value를 사용하여 Q-Value를 업데이트합니다.
   - Q-Learning은 행동 정책과 목표 정책이 다른 경우, 즉 탐험과 이용의 trade-off가 필요한 경우에 사용됩니다.

두 알고리즘 모두 최적의 Q-Value를 찾는데 사용될 수 있지만, SARSA는 현재 정책을 따르는 상황에서의 Q-Value를 학습하고, Q-Learning은 미래의 최적 정책을 추정하기 위해 최대 Q-Value를 사용합니다.

**Example: Cliff Walking (<https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py>)**  
Cliff Walking은 전형적인 강화 학습 문제 중 하나로, 격자 형태의 환경에서 에이전트가 목적지까지 이동하는 문제입니다. 그러나 클리프라는 낭떠러지가 존재하여 에이전트가 떨어질 수 있는 위험한 영역이 있습니다.   
  
SARSA와 Q-Learning은 이러한 문제를 해결하는 데 사용될 수 있습니다. 초기에는 높은 탐험 비율인 𝜖로 시작하여 점진적으로 𝜖를 줄여나가면서 최적의 정책을 찾아갈 수 있습니다. 이러한 과정에서 SARSA와 Q-Learning은 각각의 방법에 따라 다른 행동 선택과 Q-Value 업데이트를 수행하게 됩니다.

![](https://blog.kakaocdn.net/dna/Ns6wp/btsgEgm5FOD/AAAAAAAAAAAAAAAAAAAAAEhPGewarPAirHG5o5vNJdfOBp4xEwoGEF4p7XrFgD8G/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=qrNdC1tDVP%2BeEjBxNYq%2BCm1g0kU%3D)

𝜖의 감소로 인해 에이전트는 탐험과 이용 사이의 균형을 조절하면서 최적 정책에 수렴하게 됩니다. 𝜖가 줄어들면서 탐험 비율이 감소하고, 이용 비율이 증가하게 되므로, 최종적으로 최적 정책으로 수렴하게 됩니다.

**Relationship Between DP and TD**

DP(Dynamic Programming)와 TD(Temporal Difference Learning)은 강화 학습의 두 가지 주요 접근 방식입니다. DP는 문제의 MDP(Markov Decision Process) 모델을 완전히 알고 있을 때, 최적 정책과 가치 함수를 찾는데 사용됩니다. 반면에 TD는 MDP 모델을 알지 못하고, 에피소드로부터 얻은 경험을 통해 가치 함수를 추정하고 최적 정책을 개선하는 데 사용됩니다.   
  
DP는 Bellman 방정식을 사용하여 가치 함수를 반복적으로 업데이트합니다. DP는 완전한 정보를 활용하며, 에피소드 경험이나 시간 차이 정보에 의존하지 않습니다. 이를 통해 DP는 최적 정책과 가치 함수를 확실하게 찾을 수 있습니다. 그러나 DP는 MDP 모델을 완전히 알아야 하므로, 실제 문제에서는 제한적으로 사용될 수 있습니다.   
  
TD는 에피소드로부터 얻은 경험을 사용하여 가치 함수를 추정하는 방법입니다. TD는 일부 정보에 기반하여 가치 함수를 업데이트하며, 에피소드 경험을 통해 실시간으로 학습할 수 있습니다. TD는 MDP 모델에 의존하지 않으므로, 실제 환경에서 모델을 알지 못하는 경우에도 사용할 수 있습니다. 그러나 TD는 추정 오차에 따른 분산이 있을 수 있습니다.   
  
결론적으로, DP는 완전한 정보를 활용하여 최적 정책과 가치 함수를 찾는 반면에, TD는 에피소드로부터 얻은 경험을 통해 가치 함수를 추정하고 최적 정책을 개선하는 방식입니다. 두 방법은 강화 학습의 다양한 측면에서 상호 보완적인 역할을 하며, 실제 문제에 따라 적절한 방법을 선택하여 사용할 수 있습니다.

![](https://blog.kakaocdn.net/dna/chfEql/btsgEamnQhB/AAAAAAAAAAAAAAAAAAAAAD1le4vten9pDg4Tnvj9KEsdc-zGMEWL1CUBCAtDEuaC/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Sf8OIOoFOne6Dmev3Fx0HpKteXE%3D)
![](https://blog.kakaocdn.net/dna/P6Uco/btsgEdi5igO/AAAAAAAAAAAAAAAAAAAAAKHHSmtUhs1L6J7-OIn9-Jf5YnQ362MFWSm0YI0zS1wB/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=AH5tuMbpgCqak4syJSCA2gC2dHk%3D)

**Maximization Bias**   
2개의 액션에 대한 보상이 모두 평균이 0인 단일 상태 MDP를 고려해봅시다. 즉, 𝔼(𝑟𝑎=𝑎1)=(𝔼𝑟𝑎=𝑎2)=0입니다. 이 경우 Q(𝑠,𝑎1)=Q(𝑠,𝑎2)=0입니다.   
  
𝑎1과 𝑎2를 선택하는 초기 샘플이 있다고 가정해봅시다. Q의 유한 샘플 추정치인 Q𝑠,𝑎1과 Q𝑠,𝑎2를 생각해봅시다. 𝜋=argmax𝑎′ Q𝑠,𝑎으로 추정된 Q에 대한 탐욕적 정책을 고려합니다. 추정된 Q의 최대값은 0이 아닌 양수가 될 수 있습니다.   
  
이것은 최대화 편향(maximization bias)의 예입니다. 표본 추정치는 불완전한 정보를 기반으로 계산되기 때문에 정확한 값과 다를 수 있습니다. 따라서 추정된 Q의 최대값이 실제 최적 행동 가치보다 높거나 낮을 수 있습니다.   
  
최대화 편향은 강화 학습에서 발생할 수 있는 일반적인 문제 중 하나입니다. 이는 표본 기반 추정을 사용하는 경우에 특히 더 중요합니다. 이러한 편향을 완화하기 위해서는 샘플링의 다양성과 탐색을 통해 보다 정확한 추정을 수행하는 것이 중요합니다.

**Double Learning**

유한 샘플 학습 중 추정된 Q 값에 대한 탐욕적 정책은 최대화 편향을 유발할 수 있습니다. 이는 최대화 편향을 발생시키는 이중 사용 때문입니다. 즉, 최대화 행동을 결정하는 데 사용되는 Q를 동시에 값의 추정에 사용하기 때문입니다.   
  
더블 러닝(Double Learning)은 이러한 최대화 편향을 완화하기 위한 방법 중 하나입니다. 이를 위해 독립적인 두 개의 편향되지 않은 추정치인 Q1과 Q2를 사용합니다. 하나의 추정치는 최대화 행동을 선택하기 위해 사용됩니다: 𝑎∗=argmax𝑎𝑄1𝑠,𝑎. 다른 추정치는 𝑎∗의 값 추정을 위해 사용됩니다: 𝑄2𝑠,𝑎∗.   
  
이 방법은 추정된 Q 값에 대한 더 정확한 평가를 제공하여 최대화 편향을 완화합니다. 두 개의 독립적인 추정치를 사용함으로써 하나의 추정치만을 사용하는 경우보다 더욱 정확한 결과를 얻을 수 있습니다.

![](https://blog.kakaocdn.net/dna/SsECk/btsgNWG9iOm/AAAAAAAAAAAAAAAAAAAAAEwhVPR_7k4xLzp_tkBFz6oShb-lQumUo9OdBp1Ac8r-/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=tZ2PZ4zlEkONZNf8FAkbOrf6S7I%3D)

**Double Learning Algorithm**

더블 러닝(Double Learning)은 최대화 편향을 완화하기 위한 알고리즘입니다. 다음은 더블 러닝의 기본적인 알고리즘입니다:

![](https://blog.kakaocdn.net/dna/cxAABq/btsgL9GImw8/AAAAAAAAAAAAAAAAAAAAAH8_TKGvazmr_41gXNly780Y-0PuDgd68RYaao21UzZ5/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=yRGPVc5x4uoMp05G88ov%2Bbl%2Bq5E%3D)

1. 두 개의 독립적인 추정치 Q1과 Q2를 초기화합니다.
2. 에피소드를 반복합니다:
   - 초기 상태에서 시작합니다.
   - 각 타임 스텝에서 다음을 수행합니다:
     - 현재 상태에서 가능한 모든 행동 중 하나를 선택합니다.
     - 선택한 행동으로부터 환경으로부터 다음 상태와 보상을 받습니다.
     - Q1과 Q2 중 한 추정치를 사용하여 다음 상태에서 최대화 행동을 선택합니다.
     - 선택한 행동에 대한 Q1과 Q2의 추정치를 업데이트합니다. 업데이트는 두 추정치 중 하나를 무작위로 선택하여 수행합니다.
   - 종료 상태에 도달할 때까지 위 단계를 반복합니다.

더블 러닝은 두 개의 독립적인 추정치를 사용하여 최대화 편향을 완화합니다. 각 추정치는 다른 추정치의 편향을 보완하여 보다 정확한 값 추정을 가능하게 합니다.
