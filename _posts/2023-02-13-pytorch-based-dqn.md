---
title: "Pytorch based DQN"
date: 2023-02-13 23:28:10
categories:
  - 프로젝트
tags:
  - DQN
  - Reinforcement Learning
  - pygame
  - pyTorch
  - PygameDQN
---

DQN 및 강화학습을 Tensorflow로만 작성하다 Pytorch이가 점차 올라오는 추세가 되어 Pytorch 공부를 겸하여 코드 작성을 하였다.

Cartpole environment에서 return해주는 float 값들을 활용하는 DQN 코드를 먼저 작성하였다. 해당 코드를 작성후 Tensorflow와 시간차이를 확인해보니 동일한 알고리즘으로 작성한 것 같은데, 연산 속도에서 훨씬 빠른 속도를 체감하였다.

Cartpole environment에서 return해주는 float 값이 아닌 Cartpole environment를 사람처럼 보면서 입력받으며 CNN을 활용하기 위해 CNN-DQN을 작성하였다. 이전 버전들에서는 env.render만으로도 작동되던 방식이 env.render('human')으로 작성되어야 Cartpole environment가 보이는 것을 확인했고, 이전 방식으로의 return은 안되는 것을 확인하였다.(GetImage 함수에서 CV2를 사용해서 return받는 방식이었다.) 따라서 대안을 찾아야되었고 Pygame을 통한 return을 하기로 하였다. Pygame이 워낙 생소하다 보니 이를 공부할 때 생각보다 많은 시간이 걸렸던 것 같다.

추가적으로 학습이 완료되면 Gausian noise를 return값에 추가하여 학습하고, 다음으로는 Denoising으로 noise를 지운 return 값을 학습하는 방식으로 진행될 것이다. 이 둘의 차이를 알아보기 위해 말이다.

해당 코드는 github에 공유한다.

<https://github.com/cs20131516/Torch_kr/tree/main/DQN>

[GitHub - cs20131516/Torch\_kr: Torch\_study](https://github.com/cs20131516/Torch_kr/tree/main/DQN)

Torch 튜토리얼을 공부하다보니 해당 폴더를 사용한 점은 양해바랍니다.

결과가 생각보다 좋지 않게 나와서 다른 사람들의 결과를 찾아봤더니 정상적이었다.

<https://github.com/lmarza/CartPole-CNN>

[GitHub - lmarza/CartPole-CNN: Project of the "fundamentals of artificial intelligence" first year master's degree course](https://github.com/lmarza/CartPole-CNN)

Image를 사용하면 잘 안나오는게 맞다.
