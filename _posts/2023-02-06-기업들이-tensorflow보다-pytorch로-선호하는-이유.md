---
title: "기업들이 Tensorflow보다 Pytorch로 선호하는 이유"
date: 2023-02-06 01:17:31
categories:
  - 일상생활
tags:
  - TensorFlow
  - pyTorch
  - 면접용
---

간단하게 Tensorflow와 Pytorch의 정의를 보자.

TensorFlow는 다양한 작업에서 데이터 흐름 및 차별화 가능한 프로그래밍을 위한 오픈 소스 소프트웨어 라이브러리다. 기호 수학 라이브러리이며 신경망과 같은 기계 학습 응용 프로그램에도 사용되며, Google Brain 팀에서 개발했으며 많은 Google 제품 및 서비스에 사용되고 있다.  
  
PyTorch는 Torch 라이브러리를 기반으로 하는 오픈 소스 기계 학습 라이브러리이며, Facebook의 AI 연구소에서 개발했으며 많은 제품에 사용되는 중이다. PyTorch는 유연성과 사용 편의성에 중점을 두고 연구 및 실험을 위한 플랫폼을 제공한다.

그렇다면 둘의 결정적인 차이는 무엇일까?

TensorFlow와 달리 PyTorch는 "실행별 정의" 접근 방식을 사용한다. **즉, 사용자가 모델 실행을 더 많이 제어할 수 있다.**

기업들은 파이토치를 선호하는 이유로는

- 오픈 소스로 무료로 사용 가능
- 쉬운 디버깅 기능 제공
- 다양한 모델 구축 가능
- 파이썬 기반으로 개발자들이 익숙한 환경
- 클라우드 환경에서도 지원
- 텐서플로우와 비교해 더 새로운 기술 지원

이 있다.

내가 생각하는 추가적인 이유는 많은 NLP, Image, 음성인식쪽 라이브러리가 Tensorflow보다 Pytorch가 더 잘되어있기 때문에 Pytorch를 더 사용하게 되는 것 같다. 특히 대용량 모델에서는 처리 속도가 중요한데, 자동 미분(Autograd)을 선언하고 API를 불러와야 사용되는 Tensorflow보다 기본적으로 지원하는 Pytorch가 이런 부분에서 강점을 가지기 때문이다.

![](/assets/images/posts/7/img.png)

Tensorflow와 Pytorch의 점유율 현황

점유율과 관련되서는 다음의 링크들을 참조하면 좋겠다.

<https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/>

[PyTorch vs TensorFlow in 2023](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/)

<https://velog.io/@freejack/PyTorch-vs-TensorFlow-in-2022>

[PyTorch vs TensorFlow in 2022](https://velog.io/@freejack/PyTorch-vs-TensorFlow-in-2022)

추가적으로 조금 더 깊게 들어가면 다음에서 Pytorch가 강점이 있다.

- 모델 구축 방법: 파이토치는 기존 텐서플로우와 달리 모델을 구축하는 과정이 더 쉬워 사용자 친화적
- 디버깅 기능: 파이토치는 보다 쉬운 디버깅 기능을 제공
- 자동 미분 기능: 파이토치는 자동 미분(Autograd) 기능을 지원하여, 사용자는 수동으로 미분을 계산할 필요 없이 자동으로 계산
- 인터렉티브 디버깅, 그래프 실행 등을 제공하여 디버깅을 쉽게 할 수 있고, 코드를 편하게 작성 가능

추가적으로 자동 미분(Autograd)에 대해서 코드를 남기도록하겠다.

Tensorflow의 Autograd

```
import tensorflow as tf

# Define a simple function
def f(x, y):
    return x**2 + y**2

# Define the inputs
x = tf.Variable(3.0)
y = tf.Variable(4.0)

# Use GradientTape to track the gradient
with tf.GradientTape() as tape:
    z = f(x, y)

# Compute the gradients
dx, dy = tape.gradient(z, [x, y])
print(dx) # 6.0
print(dy) # 8.0
```

Pytorch의 Autograd

```
import torch

# Define the inputs
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

# Define a simple function
z = x**2 + y**2

# Compute the gradients
z.backward()

print(x.grad) # 6.0
print(y.grad) # 8.0
```

파이토치는 자동 미분을 기본적으로 지원 하며, 코드도 짧고 간결한 것을 확인할 수 있다.

개인적인 사담인데, **처음 인공지능을 하시는 분들에게는 Tensorflow를 추천한다.** 생각보다 Pytorch에서 torch.cat과 같은 torch 변환을 많이 사용해야되는 이 장벽이 처음하시는 분들께는 꽤 진입장벽이 높다.
