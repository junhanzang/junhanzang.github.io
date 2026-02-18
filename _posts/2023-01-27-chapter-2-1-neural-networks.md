---
title: "Chapter 2-1 Neural Networks"
date: 2023-01-27 00:18:52
tags:
  - Ann
  - Artificial Neural Network
  - Machine Learning
  - neural network
---

간단하게 인공 신경망(Artificial Neural Network, ANN)의 탄생을 설명하고 넘어가도록하자.

우리의 뇌는 10<sup>11</sup>개의 뉴런들로 이루어져있다.

뉴런은 다음 그림과 같이 생겼다.

![](https://blog.kakaocdn.net/dna/NI61F/btrXhQFKihV/AAAAAAAAAAAAAAAAAAAAAGC8GhdOIf0ZD7afeLusSHCtu4ahrjnpSNyQDHIs3Bze/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=rkOC0EiTkmmGGzF1XQoJUhoJoEM%3D)

출처: https://bioinformaticsandme.tistory.com/233

간단하게 이 뉴런의 구조를 Cell body, Dendrite, Axon으로 나눌수 있다.

뉴런은 해당 구조를 활용하여 정보전달을 하는데, 이는 아래의 그림과 같이 나타난다.

![](https://blog.kakaocdn.net/dna/PMMw9/btrXe810q7x/AAAAAAAAAAAAAAAAAAAAAMn27nPypSnFBopCfnE30-dZDtIL7k3wj4WypwfZ7dTw/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=SIbMfSpcuRfgwOXu0XtwlFvBMaA%3D)

출처: https://en.wikipedia.org/wiki/Neurotransmission

Dendrite: 신경전달물질을 받음

Axon: 신경전달물질을 방출

Cell body: 신경전달물질저장 공간

이를 모방한 것이 ANN이다.

ANN을 위의 방식으로 모방하면 다음 그림과 같이 모방이 가능할 것이다.

![](https://blog.kakaocdn.net/dna/IzVi5/btrXiN4UVBx/AAAAAAAAAAAAAAAAAAAAALku2PxCHtViM7Uo2YATDd_iCsyAklr8wNLSPUQQVbQy/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=m5YaezBt98hHXb65y9XDEZwxrpQ%3D)

출처: https://www.quora.com/What-is-the-differences-between-artificial-neural-network-computer-science-and-biological-neural-network

초기 모델들이 Sigmoid function을 사용하는 것과 해당 function을 바탕으로 XOR문제를 풀 수 있는 것은 너무 상세한 설명임으로 넘어가도록 하겠다. ~~(실제로 사용하는 일도 없고 말이다.)~~

![](https://blog.kakaocdn.net/dna/r3bfN/btrXmk8uMZT/AAAAAAAAAAAAAAAAAAAAABa-h9mOCNARYzZGKa8zveEQgVxb8DacJr-6yWf8HiRO/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=FBLO%2FkVH%2BfKxGNAYOmgpdjZgCaQ%3D)

출처: https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png

간단한 ANN의 탄생을 알아보았고 이제 학습 알고리즘으로 가보자.

![](https://blog.kakaocdn.net/dna/eTUq0R/btrXl1uGA5t/AAAAAAAAAAAAAAAAAAAAANYS771HGePUMksWaqCKXCd_gbECXCLzSyaPJTiQ-9eX/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=X1nogST%2BY2sGHyD5JHnRI4bXi4U%3D)

출처: https://static.javatpoint.com/tutorial/deep-learning/images/deep-learning-example.png

사진을 주어진 데이터, Layer들이 이루는 구조를 **Neural Network**라고 한다.

그렇다면 Neural Network에 해당 데이터를 어떻게 학습시킬까?

이는 Neural Network의 구조를 이해하면 Machine Learning에 적용시켜 학습시킬 수 있다는 것을 알 수 있다.

![](https://blog.kakaocdn.net/dna/bUonLa/btrXnj8WL7c/AAAAAAAAAAAAAAAAAAAAABadnIzxN5osVy9huAgZyXza3mOM8iSfS6lz80ooGn_8/img.webp?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ZGQ8BrlYis6X4l8eI%2Bg%2Fe3ZxSvw%3D)

출처: https://python-course.eu/machine-learning/neural-networks-structure-weights-and-matrices.php

결국, chapter 1에서의 문제가 다음과 같이 변경될 것이다.

![](https://blog.kakaocdn.net/dna/c8RRxM/btrXm87BcPg/AAAAAAAAAAAAAAAAAAAAAIhrlUh_aVKxGTnyusfXg4uDe4dzpArsJ4nC_ys9GfTG/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=QKru1dRMmDSHTrkRC72z3C0QPj8%3D)

𝑁𝑁(𝒘, 𝒙) = 𝒕, 𝒘 = (𝑤<sub>1</sub> ,𝑤<sub>2</sub> ,…,𝑤<sub>n</sub>)으로 정의하게 된다면

**아래의 함수를 minimum스럽게 만드는 w를 찾는 문제로 변경하게 된다!**

![](https://blog.kakaocdn.net/dna/LQm0e/btrXm87CLuI/AAAAAAAAAAAAAAAAAAAAAEa9s8egnYiXMEwWfl1aphYTcDFfKbUYercAkUr_aSSZ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=PQF%2FvyE2pCJwyI%2F9rM4pHnrsPDc%3D)

따라서 chapter 1과 같이 경사하강법을 고차원 함수에 적용한다면 다음과 같이 나올 것이다.

![](https://blog.kakaocdn.net/dna/bcWZ1E/btrXmJUBchV/AAAAAAAAAAAAAAAAAAAAAEY8zUepR2WnEthLWP__pT68f_Sx7rJgwo6akfD9o29j/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hYViGqpEghbMKEUMyK4k2352BfQ%3D)

출처:https://box-world.tistory.com/7

수식적인 예시는 따로 하지 않겠다.

![](https://blog.kakaocdn.net/dna/ciQ3PY/btrXmKeUYcM/AAAAAAAAAAAAAAAAAAAAAFm9aLDfXaUUJzb_2PesGADn4cAxPpHaDKDCQ6IJ9knD/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=tHBLZm%2BmntDqO5o2%2BrNgx%2FdIgUA%3D)

결국 위의 식을 적용하는 것과 같기 때문이다.

그렇다면 위 식에서의 𝜂는 어떻게 구할까? 우리가 임의로 정하는 값이 될까?

결론부터 말하면 **Error Back Propagation**이라는 방식을 이용하여 **𝜂(가중치)**를 구한다.

Error Back Propagation이란 신경망의 출력과 원하는 출력 사이의 오차를 이용해 가중치를 조정하는 방법으로 정의된다.

하지만 이렇게 정의로만으로는 이해하기 어렵기 때문에, 어쩔수 없이 수식적으로 들어가보자.

![](https://blog.kakaocdn.net/dna/bjRGoM/btrXn6ap7FW/AAAAAAAAAAAAAAAAAAAAAI2nLMx0-4zPAaHXPISa04BUYhweK1YhFv-igLHvlrrw/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=HU%2Fxm3qb9aZPd4DbW21l%2FDFkY9I%3D)

데이터와 해당 데이터를 위한 최적의 함수를 구하기 위해서는 E(w)를 최소화시키는 w의 값을 구해야한다.

데이터와 Neural Network의 구조를 더 쉽게 보기 위해 다음의 그림으로 수식을 설명하겠다.

![](https://blog.kakaocdn.net/dna/3l8Mb/btrXn7NU8mX/AAAAAAAAAAAAAAAAAAAAAH9HY6m6CHrOB54Zt9cR6-fFquEo6N_NZzZP9SrFMX6j/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Sn9H2MgyXUDwnPS6J94GrWEotJ0%3D)

그림 1: 데이터와 Neural Network의 구조

Chapter 1에서의 내용을 기억해보자.

**우리는 경사하강법을 적용하기 때문에, E에 대한 미분 값들을 구해야된다.**

![](https://blog.kakaocdn.net/dna/cnMfrx/btrXnkmKeE5/AAAAAAAAAAAAAAAAAAAAAPMWHYI13kVSTXGuzeq6kavJZOtUJtrT90x2B37GvPYD/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=LPVKhdPusEj1mirvSckjkyc10i8%3D)

계산은 Output과 Hidden Layer의 weight을 사용하는 계산을 바탕으로 진행할 것이며 그렇게 되면 수식은 다음과 같이 정의할 것이다. (초기 모델은 Sigmoid function을 사용한다고 앞에서 설명하였다.)

![](https://blog.kakaocdn.net/dna/eHRMvX/btrXmkgWUrq/AAAAAAAAAAAAAAAAAAAAAMkJbHpto19iONwtfgyksgJp6jHoU4-2TnC6pfRwU0HK/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ZmrVKB%2FuSg6cAAlcX8lmjUA98Oo%3D)

그림 1에서 각 w에 대한 E미분 값을 구하려면 다음의 식이 되어야 한다.

![](https://blog.kakaocdn.net/dna/nzLqO/btrXm7nD5Nu/AAAAAAAAAAAAAAAAAAAAADJPCkW1gccbkCovEJMeZ0nkuNMVeUdlmPB1Ilzp_PMP/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=CYnaO%2BPdglOYX8EZLeGSHqU0SMs%3D)

Sigmoid의 정의와 미분은 다음과 같다.

![](https://blog.kakaocdn.net/dna/bOuZc2/btrXmbR38Zd/AAAAAAAAAAAAAAAAAAAAAN04UTBjTCQtkj7UsOwvdQefuhyI-QAexDCN177ULjGE/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=p5fgTOpdKvkzyfV996rBQ7lifxA%3D)

순차적으로 𝛼𝐸를 구하면 다음과 같다.

![](https://blog.kakaocdn.net/dna/bnUfRD/btrXmpvMCHZ/AAAAAAAAAAAAAAAAAAAAACHHnD9uarwCKthf1gRUQRp9yDlZN7MhFESIWtt3Wleg/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=mmwaEH5buLyoBIWhiWxeQX0bBpo%3D)
![](https://blog.kakaocdn.net/dna/EvWPj/btrXnllFmmj/AAAAAAAAAAAAAAAAAAAAAG8sbfFO-6Qw8tHH4WmtghdeJLZuKzowcIRxDxpXafFG/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=VtZd4VMGnEo9ksQEkoPr3SEptZk%3D)

따라서 모든 W에 대해서 계산이 완료된다면 다음의 식이 나온다.

![](https://blog.kakaocdn.net/dna/dASdeC/btrXEBn01oW/AAAAAAAAAAAAAAAAAAAAAMTvE5YM6N3H-HJhm38bv1ZSFvq4ZLHgF3ZNXZeb9yrJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=XRKl%2FpiPJ8sf7u%2FnmwPuU9wQM9E%3D)

**Input Layer과 Hidden Layer의 weight을 사용하는 계산을 해보길 추천한다.**

앞에서와 비슷한 전개과정을 사용하면 다음의 식이 나올 것이다.

![](https://blog.kakaocdn.net/dna/cuD6Ru/btrXBXZRWbH/AAAAAAAAAAAAAAAAAAAAAKMj_5Yg8W6Q6wJBzmpS3h1L5Vn2czTNR3g8KaWV9z8C/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=2JxZZKIy6sF8YKXfRf3CctSqj%2FQ%3D)

우리가 사용한 모델을 Shallow Network이며 일반적으로 Hidden Layer가 2~3개 이하일때로 정의가 된다.

그렇다면 Hidden Layer가 4개 이상이면 뭐라고 부를까?

이는 **Deep Network**라고 부르며 다음과 같이 이루져있다.

![](https://blog.kakaocdn.net/dna/b3gOvt/btrXGTWaVMC/AAAAAAAAAAAAAAAAAAAAABAtFFgjtia0e20cIuOiDl_R0Dj_NFW4iWwUTgGnnLq_/img.webp?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=npLRbWmwS3nO%2BVGD9rRtpFaytcM%3D)

출처:https://www.ibm.com/kr-ko/cloud/learn/neural-networks

Deep Network를 더 상세하게 들여다보게 된다면 다음과 같이 이루어져있다.

따라서 여러개의 layer에 적용하게 된다면 다음과 같이 전개될 것이다.

![](https://blog.kakaocdn.net/dna/bdCUOO/btrXGS4hPyK/AAAAAAAAAAAAAAAAAAAAAKVSaSzubLrEY-Jr5PMVz6cHkHMAQnkZ4dUcWAz1r4um/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=CQGBBzaKHNLE46bmvueCI5RusgE%3D)

net 다음에 새로 보는 h이 있다. 이는 Activation function으로 Neural Network에서 뉴런의 출력 값을 결정하는 함수이다.

활성화 함수는 입력 값을 받아서 뉴런의 출력 값을 결정하며, 각 뉴런이 활성화될지 비활성화될지 결정하는 중요한 역할을 한다.

Deep Network에서 E를 계산은 다음과 같다. (궁금한 사람은 펼치기로 보길 바란다.)

더보기

![](https://blog.kakaocdn.net/dna/cnK4Oo/btrXIPInwhq/AAAAAAAAAAAAAAAAAAAAAIQgLSFyGsC9_XpigC6q3NytgrMr6wcV_-uwCIkl5c3i/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=r44kxPr%2FMmnzU67sAgaAtLutBpk%3D)
![](https://blog.kakaocdn.net/dna/eddcnr/btrXLseCULV/AAAAAAAAAAAAAAAAAAAAAKjCsBJpjSdlEcyiZvX7ufh1MY_GN4rwnYwOw3kNxw77/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=oAgMNMQQYBWJ4CbuopnXkaeVunI%3D)
![](https://blog.kakaocdn.net/dna/bpZlhb/btrXK8mOZAl/AAAAAAAAAAAAAAAAAAAAAEIUPaVDHZSvHQSbYjvXn2ruMS0zVsoKw6qQG_IB7Sxo/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=JmPJAa9XFTHl%2FJETQhXueV2ssHM%3D)

h = simoid(net)이라면 다음과 같이 정의가 완료된다.

![](https://blog.kakaocdn.net/dna/eysSl3/btrXIzyJ6F5/AAAAAAAAAAAAAAAAAAAAAGYwQ9NZX-Zvj6JIMZPeSesVwx-JTsMAEqdofLqk896e/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Jki3T3Lue%2F%2BIQlErj7%2ByzCmVa%2BY%3D)

우리는 이제 Neural Network의 작동 방식과 구조에 대해서 알았다.

**그럼 좋은 Neural Network 모델은 무엇일까?**

머신러닝 때처럼, 다음의 데이터를 받았다. 이때의 가장 좋은 Neural Network 모델은 무엇일까?

![](https://blog.kakaocdn.net/dna/zWgBg/btrXCJnS6qF/AAAAAAAAAAAAAAAAAAAAAKmMXH9zvw_BOTveZTjXULB4ywEbS6xvntPlLUI16Ye3/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=agjhz8gWK0e15oPCjN1oy8ZrHlw%3D)

출처:http://mlwiki.org/index.php/Overfitting

다양한 정답이 있지만 일반적으로 1차항, 2차다항식, 고차다항식의 형태로 나올것이다.

![](https://blog.kakaocdn.net/dna/EISvR/btrXIxS6P6M/AAAAAAAAAAAAAAAAAAAAAD4WZsIf-I4rUMYuv47QGt3M343yIyCGgxyVXXASBK34/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=lxRDkTWhtO9D85MSdu16813TilQ%3D)

출처:https://kimlog.me/machine-learning/2016-01-30-4-Regularization/

주어진 데이터에 가장 잘 맞는 형태는 고차다항식이지만, 이를 우리가 추정가능 함수로 사용가능 할까?

답은 사용할 수 없다이다. 제일 오른쪽의 형태는 학습 데이터에 너무 과도하게 최적화되어 새로운 데이터에 대해 적용되지 않을 가능성이 훨씬 높다. 이를 **Overfitting**이라고 하고 일반적으로 큰 변화(Variance)를 가지고 있다.

제일 왼쪽의 형태는 데이터의 증감을 잘표현했지만 너무 큰 error를 발생시킨다. 이렇게 모델이 너무 단순해서 데이터의 복잡성을 포착할 수 없는 모델을 **Underfit**이라고 말하며 높은 편향(Bias)를 가지고 있다.

가장 적절한 모델은 중간 모델이며 새로운 데이터에 대해 예측을 가장 정확하게 할 수 있기 때문이며, 이를 **Generalization**이라고 한다.

즉, 좋은 모델은 **Generalization**이 잘되는 모델인 것이다.

우리는 지금까지 Activation function을 sigmoid로 사용해왔다. 그렇다면 sigmoid는 만능일까? 초창기의 Neural Network가 사용했기 때문에?

정답은 당연하게 아니다.

[Regression](https://ko.wikipedia.org/wiki/%ED%9A%8C%EA%B7%80_%EB%B6%84%EC%84%9D)을 아래의 형태의 Neural Network로 만들수 있을까?

![](https://blog.kakaocdn.net/dna/dzsQRG/btrXC0JRtAB/AAAAAAAAAAAAAAAAAAAAANdgmcwywH-IQc96AaPaj86A9KVRHJq4iaZd_evTPZul/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=o1kg1Mxwrpp9aUdv0RA325peN%2Fk%3D)

Activation function이 sigmoid이기 때문에 이는 불가능하다.

이를 해결하기 위해서는 간단하게 Output에 있는 Activation function을 제거하면 된다.

![](https://blog.kakaocdn.net/dna/cXG0Up/btrXIyxOvQF/AAAAAAAAAAAAAAAAAAAAAOwAzkY8mw7YTVzQHI0UroivXv8r2D6cQmSkQvl7FUzp/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=QYNgfjpOTn25K8R9jKeY21V7slI%3D)

그렇다면 모든 경우에서 E function도 MSE로 고정형태일까?

정답은 당연히 아니다.

다음의 문제를 풀어보도록하자.

![](https://blog.kakaocdn.net/dna/z6SJT/btrXHInPQMZ/AAAAAAAAAAAAAAAAAAAAAE7O9tbIFqNxuUDlFJcOD_qwCjSmRDdbHDrlxoa-t58-/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=%2B6GgpGuHPWXQUBhotARrOwl4hGQ%3D)

점을 검정색과 붉은색으로 분류하는 Binary-Class Classification이라고 불리는 문제이다.

만약 이 문제에 대해서 Activation function과 Output 모두 sigmoid 함수로 진행하게 된다면 어떻게 될까?

먼저, 우리가 이 문제를 풀기위해서는 Red와 Black을 숫자 형태로 labeling을 진행해야 될 것이다.

즉, 주어진 방정식이 다음과 같이 변화하게 된다.

![](https://blog.kakaocdn.net/dna/bVndx3/btrXN4xkU33/AAAAAAAAAAAAAAAAAAAAACiS-Vbk535C_ASA_JL2MM70enUvQHGGVmBoWmljQKII/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ueobGg%2F6Dyh1HVJoXYy8HwJba%2Fo%3D)

상당히 그럴듯해 보인다. Sigmoid 함수는 0 ~ 1 사이 값을 출력해주기 때문이다.

그렇다면 추가적으로 E function을 MSE를 그대로 사용해도 될까?

MSE는 실제 Label과 예측된 출력 간의 평균 제곱 차이를 측정하지만 오류의 크기는 고려하지 않는다. 이로 인해 신뢰할 수 있는 예측이 중요한 이진 클래스 분류 문제에 대해 모델을 교육할 때 문제가 발생할 수 있다.

해당 답변이 어려울 수도 있다. (추가적인 공식을 예시로 작성해두었다.)

더보기

Deep Network 공식에서 정리를 하면 다음의 식이 나온다.

h = sigmoid(net)

![](https://blog.kakaocdn.net/dna/bJYLVB/btrXM30czh0/AAAAAAAAAAAAAAAAAAAAAIZPALxQZtHtXabNRzDI33kGG7tDmMESx8-FkjLSo47e/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=H4GdLj%2FBIlFRkna3qzAJpcnoxZQ%3D)

h<sub>nk</sub>가 1 또는 0에 가까우면 n번째 훈련 데이터의 모든 기울기는 0이 된다.

따라서 h<sub>nk</sub>가 1이나 0에 가깝지만 틀리게 된다면, Neural Network는 잘못된 학습에서 벗어날 수 없게된다.

**그러면 어떤 E function를 사용해야될까?**

각각의 확률을 최대화하고자 하는 공식을 사용하면 되지 않을까?

이를 수식적으로 보면 다음과 같다.

![](https://blog.kakaocdn.net/dna/cC7JDC/btrXPUtTLas/AAAAAAAAAAAAAAAAAAAAAP9evSLPFARrmb3xI_-wUiAmzGg0q8bSGYcyUIG08BZx/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Z8m7RSg4zOIUO4mcnk9dgzT2tnU%3D)

유도된 식은 Cross Entropy라고 불리며, 이를 활용하면 Red, Blue에 대한 확률을 최대화하는 경향으로 가는 것을 확인할 수 있다. 더 정확한 설명으로는 Cross Entropy는 정확하지 않은 신뢰할 수 있는 예측에 대해 모델에 더 심각한 페널티를 주기 때문에 분류 문제에 평균 제곱 오차(MSE)보다 더 적합한 손실 함수로 이해하면 된다.

2가지 분류 문제에서 더 많은 분류 문제로 넘어가보자.

![](https://blog.kakaocdn.net/dna/34KKn/btrXM4dNu4n/AAAAAAAAAAAAAAAAAAAAAEYwSAXED6OGvJcWFxCtaEniGhKojX4FiK78MgT1_ROc/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Uam%2F%2BaWV19%2Bc78qhYZaOu1PN%2BLE%3D)

이런 분류 문제를 Multi-Class Classification이라고 한다.

같은 Class Classification 문제이니 이전과 같이 Labeling을 진행해보자.

그렇다면 다음과 같은 형식으로 우리가 변형시킬수 있을것이다.

![](https://blog.kakaocdn.net/dna/but0IS/btrXOVUk8R6/AAAAAAAAAAAAAAAAAAAAACE2GnUSaBd6A3xR8cR76Lym-lf-u6jCWkDaIM39ySxl/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=a7vn1JO6phO2QFCPZ%2BAulmAxXnw%3D)

하지만 이는 좋지 않은 Labeling 방식이다.

Red, Yellow, Blue 사이에는 순서가 없기 때문에, Red > Yellow > Blue이라고 말할 수 없다.

따라서 우리는 새로운 형태의 Labeling 또는 Output 형식을 만들어야된다.

앞선 0,1 방식을 추가적으로 전개시키는 방식으로 진행하면 되지 않을까?

![](https://blog.kakaocdn.net/dna/6EgpI/btrXMSqS7Ah/AAAAAAAAAAAAAAAAAAAAAOHJaIFfEWXQ6vbygtRSL_Lz3Ys3FGmwpmzpKP9DRpce/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=WOa%2FPHOeJDE7Bx4ji0%2FOJoaAAiE%3D)

하지만 이렇게 된다면 지금까지 써왔던 아래의 모델은 어울리지 않는다.

![](https://blog.kakaocdn.net/dna/dzsQRG/btrXC0JRtAB/AAAAAAAAAAAAAAAAAAAAANdgmcwywH-IQc96AaPaj86A9KVRHJq4iaZd_evTPZul/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=o1kg1Mxwrpp9aUdv0RA325peN%2Fk%3D)

왜냐면 출력이 최대 1개이 때문이다. 따라서 우리는 Output을 늘려줘야한다.

그렇게 되면 Output Node를 늘려 다음과 같은 Neural Network가 된다.

![](https://blog.kakaocdn.net/dna/mokyd/btrXN3ljRRH/AAAAAAAAAAAAAAAAAAAAAOodf7g_JfZZpmzwb4dMtxc5UPpsjjGXJIqsL9G9XOjJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=8aAWVEP4VZev05LRk%2FkTjbkqgbY%3D)

다중 형태로 변경되었기 때문에 E function 또한 추가적으로 아래와 같이 변경될 것이다.

![](https://blog.kakaocdn.net/dna/b5AC5S/btrXOj2ADic/AAAAAAAAAAAAAAAAAAAAAHC9vRbQohDPzqv6gcIqLurkP20GGQ8VPNnQXHpIr7WI/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=DA9FMnj5TqJnvQl1c0yvXm0p0Cg%3D)

하지만 Activation function을 sigmoid로 유지하면 다음의 조건식을 만족시킬 수 없다.

![](https://blog.kakaocdn.net/dna/cpsC8a/btrXOjuJLmU/AAAAAAAAAAAAAAAAAAAAAHwmTA_PwHddDDKc7AcFKhqxOT_siKOXiOCvKw93YI_O/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=b2OrSjFytFL2mF25XKeqAym9860%3D)
![](https://blog.kakaocdn.net/dna/bo37dm/btrXM30j776/AAAAAAAAAAAAAAAAAAAAAJsWwCXeQUxbLEjkKeEU_O185iCRI-vYq-mLQPluB7vM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=da9V9RPYXrKR4Q0QJP0txiK4gno%3D)

따라서 우리는 합이 1이되는 새로운 Activation function을 찾아야한다.

이 조건을 만족시킬수 있는 Function은 Softmax로 각 layer들의 Output들을 확률화한 것과 같다.

![](https://blog.kakaocdn.net/dna/mVi65/btrXOtYbEfC/AAAAAAAAAAAAAAAAAAAAAPlIT88PvRE6RXQ1ytXa6J90N7MkxHewayyFOv22FI0z/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=L%2B9VeDmroiDcx040rtpwqCg7CVU%3D)

Softmax Function

더 자세한 변형은 더보기에 있다.

더보기

![](https://blog.kakaocdn.net/dna/zc4RK/btrXOlsAgxH/AAAAAAAAAAAAAAAAAAAAABI-OuBZIc2DPc2N6UJ4rlgGfmQjo9UKJDuLZVmPmNkH/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=%2BeRFwlgYOQ%2FoLnqJKaQjUoQpsOg%3D)

따라서 E function은 다음과 같이 변형될 것이다.

![](https://blog.kakaocdn.net/dna/UpJja/btrXOjardU5/AAAAAAAAAAAAAAAAAAAAAG-iIF1sdh_hDCNa9eSyRXZ0wHxvI1AoMrIvjN8Ay4Fc/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=TtDoX2xs5%2BJvFAh3NUP3M0LpZ4s%3D)

지금까지는 하나의 항목에 대한 분류를 했다.

그렇다면 여러가지 항목(label)에 대한 것은 어떻게 해야될까?

![](https://blog.kakaocdn.net/dna/KMnRJ/btrXPrMomuz/AAAAAAAAAAAAAAAAAAAAAOtNvJpe2ZIo2SRFccz7E9TTJMdzzOgbpRqVbN3LKhut/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=gY1x%2FV0VYRQjRz08bEvIiScSRp0%3D)

우리를 괴롭혔던 조건인 합이 1이 되어야함을 만족시키지 않아도 되기에 Activation function에 sigmoid를 써도 된다.

따라서 다음과 같이 될 것이다.

![](https://blog.kakaocdn.net/dna/bplwv5/btrXPUHzRuE/AAAAAAAAAAAAAAAAAAAAADF20KtNkQG_TtkR1tnzYy_VyUwhZDjv7GyZuX4YfH2i/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=mXpnqaZIC%2FRIoTodwRlPJIQFY4M%3D)

우리가 정리한 Regression, Classification, Multi-Label을 요약하면 다음과 같다.

![](https://blog.kakaocdn.net/dna/b28oIg/btrXPU1SskU/AAAAAAAAAAAAAAAAAAAAADlyWdbzq5TGqjXuMwBAZtTBlb7KeU-krB3gHB6RT7FV/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=nXs7FMgW5xJeFt%2FqESC0ODk%2FQMY%3D)

우리가 지금까지 공부했던 내용 중 Error Back Propagation에 대해서 생각해보자.

모든 E의 미분 값에 대해서 한번에 계산하고 이를 업데이트에 활용하였다.

생각해보면 이는 큰 데이터에서 비효율적일 수 있다. 모든 E의 미분에 대핸 계산을 진행해야되기 때문이다.

우리가  Error Back Propagation에 사용했던 weight 업데이트 방식을 도식화하면 다음과 같다.

![](https://blog.kakaocdn.net/dna/y9UrS/btrX1ly5H82/AAAAAAAAAAAAAAAAAAAAAJyEZ6yiTq8NoBTw3TAgWUNUPQ9Ex_ZSZjIni3zTxNP2/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=CdV4xKtIt%2Fbh%2B8F6jw3Cme1%2BS%2FQ%3D)

Batch Gradient Descent

그렇다면 큰 데이터 세트에서는 E의 미분 값을 한번에 계산하지 않고 나누어서 계산하면 되지 않을까?

![](https://blog.kakaocdn.net/dna/biaGDp/btrX0McDImt/AAAAAAAAAAAAAAAAAAAAADaRjxVkpAi_8hEC85UYZYJQ7xGkZfOaLYAivMN6TQXz/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=n0dUDWsyel8JAtzxYYpgjm9E64c%3D)

Stochastic Gradient Descent

이는 수학적으로 거의 동일하며, 궁금하면 다음의 사이트에서 확인하면 좋겠다. (<https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e>)

이를 Stochastic Gradient Descent라고 하며, 한번에 모든 gradient가 계산되는 방식은 Batch Gradient Descent라 명명된다. 그렇다면 Stochastic Gradient Descent는 어떤면에서 좋을까?

1. 하나의 업데이트에 대해 하나의 샘플에 대해 그래디언트가 계산됨  
2. 일반적으로 더 빠르고 온라인 학습에 사용할 수 있음  
3. 변동: 좋거나 나쁠 수 있음  
4. 작은 학습률로 비슷한 성능을 보임

더보기

![](https://blog.kakaocdn.net/dna/nvM3I/btrX2pmZQLK/AAAAAAAAAAAAAAAAAAAAAFsAaCkkKXUkeoUr3xcJ96BDa5OZKkc0xMLz0VHVOuwN/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=4XP%2B8GYfYuJVZmybRmBU1JvXPvI%3D)

Stochastic Gradient Descent의 식과 수렴 과정

Batch Gradient Descent의 특징은 다음과 같다.

1. 한 번의 업데이트에 대해 전체 데이터 세트에 대한 그래디언트가 계산됨  
2. 배치 경사하강법은 로컬 최소값으로 수렴되도록 보장됨  
3. 대규모 중복 데이터 세트에 대한 중복 계산됨

**따라서 큰 데이터 세트는 Batch Gradient Descent, 작은 데이터 세트는 Stochastic Gradient Descent를 사용하면 된다.**

Batch Gradient Descent와 Stochastic Gradient Descent 사이 방법도 있지 않을까?

![](https://blog.kakaocdn.net/dna/byDT4t/btrXXysSw8d/AAAAAAAAAAAAAAAAAAAAAMwu_byxmFJACUcc8-pIwtO7tkaNTTKiXdL1APhxNPJV/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=k0tRQtacT11U5x4ESyaPTbG8gvo%3D)

Mini-batch Gradient Descent - 1

![](https://blog.kakaocdn.net/dna/ce9azQ/btrX2WkJIzE/AAAAAAAAAAAAAAAAAAAAAIvMa2yaMAHUKiguCjUlCQdkZBorb1Krxjwu2JkMhlBp/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=tL%2F6P0R78NC1pUcG7V9muyYv2YU%3D)

Mini-batch Gradient Descent -2

위의 그림과 같은 방식으로 예시를 들 수 있으며, 이를 **Mini-batch Gradient Descent**라고 한다.

더보기

![](https://blog.kakaocdn.net/dna/t7aMq/btrX1lMPwPG/AAAAAAAAAAAAAAAAAAAAAH4Y-h-U-TcXmNKCMNlVilY1zR12oZaZFt9_6K3UujDi/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=R0L8Lj0S9jCARGiMhSDa%2Blo18NQ%3D)

Mini-batch Gradient Descent 수식

Mini-batch Gradient Descent의 특징은 다음과 같다.

1. 일반적인 배치 크기 : 수천에서 수십에 이르는 데이터 세트에 따라 다름  
2. 장점  
 - 실제 기울기의 좋은 추정  
 - 높은 처리량: GPU에서 한 번에 많은 수의 코어를 사용할 수 있습니다.  
 - 더 빠른 수렴: 좋은 예측 + 높은 처리량  
3. 단점  
 부정확: 분산이 큰 데이터 세트

위의 방법들은 Gradient Descent Method 개선하는 방법에 대해서 논의했다. 그렇다면 추가적으로 어떻게 해야 더 좋은 Gradient Descent Method이 될 까?

더 좋은 Local Optimum을 찾는 방법과 학습이 시작되면 큰 학습률이 선호되지만, 학습이 완료되면 작은 학습률이 선호되기 때문에 상황에 따른 Adaptive learning rates이 된다면 Better Gradient Descent Method라고 칭해도 되지 않을까?

더 좋은 Local Optimum을 찾는 방법은 어떤 방법이 있을까?

더 나은 곳으로 가기 위해 어떤 힘이 있으면 좋지 않을까? 이를 Momentum, 즉 운동량을 추가해준다라고 생각해주면 좋다.

![](https://blog.kakaocdn.net/dna/bJJ6SP/btrX41AcdGd/AAAAAAAAAAAAAAAAAAAAAPOT7VF5TEHXBU_ABehrmy427N5zDZBNuS_So9yISV3B/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ZNrPfrd8YZrwlg44YM3r83t6ql0%3D)

Momentum

모멘텀에 대한 설명은 생각보다 간단하니 다음을 참고하면 좋겠다.

<https://deepestdocs.readthedocs.io/en/latest/002_deep_learning_part_1/0021/#momentum>

[0021 Gradient Descent & Momentum - Deepest Documentation](https://deepestdocs.readthedocs.io/en/latest/002_deep_learning_part_1/0021/#momentum)

<https://towardsdatascience.com/gradient-descent-with-momentum-59420f626c8f>

더보기

결론적으로 간단한 식으로 표현하면 다음의 식으로 표현된다.

![](https://blog.kakaocdn.net/dna/vDL71/btrX7Y37Q8u/AAAAAAAAAAAAAAAAAAAAAFUBFo6rsh9P_NGtsS2F6DWnedRuAkN6NFyOjRNGlHWr/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=tbkQ926jtqHahOy2J8xYuwizhOk%3D)

모멘텀은 과거 그래디언트의 지수 평균으로 표현된다.

모멘텀도 장점만 존재하는 것은 아니다. 단순히 모멘텀만 추가하면 과도한 업데이트가 발생할 수 있고 최적의 local minimum 위치를 놓칠 수 있다.

![](https://blog.kakaocdn.net/dna/cq6BL1/btrX31gqgGg/AAAAAAAAAAAAAAAAAAAAAId0-IQFk1WyoSfqk8Ipf2gzOvsAtq_qn7WLZShayruO/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=H3aM9dyQ%2BPCR0IjHf0tJ0pheM3w%3D)

Momentum 단점

그래서 나온 Momentum이 Nesterov Accelerated Gradient (NAG)이다.

더 자세한 정보를 얻고자 하면 : <https://m.blog.naver.com/sw4r/221231919777>와 <https://tensorflow.blog/2017/03/22/momentum-nesterov-momentum/>를 참고하면 좋겠다.

[[Deep Learning] 최적화: Nesterov Accelerated Gradient (NAG) 란?](https://m.blog.naver.com/sw4r/221231919777)

나는 NAG의 Momentum을 기본 Momentum을 바탕으로 간단하게 설명하고자 한다.

![](https://blog.kakaocdn.net/dna/cJw8y7/btrX2XMmvEL/AAAAAAAAAAAAAAAAAAAAAKiu8Me9x2ns7DXaLxr0Re5kZxtbmL_lDC22rLcTqe8c/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=WjbgGJ07apRzAlRbbNwChN30FXg%3D)

NGA Momentum

1. Momentum을 고려하여 현 위치 업데이트를 업데이트 한다.

2. 새로 업데이트한 위치에서 기울기를 평가한다.  
3. 추가적으로 평가된 기울기를 바탕으로 위치 업데이트 한다.

Adaptive Learning Rates는 다음의 의문들로부터 파생되었다.   
모든 가중치에 대해 동일한 학습률을 사용하는 이유는?   
다른 학습 속도를 사용할 수 있을까?   
왜 다른 학습률이 필요할까?   
왜 일부 가중치는 자주 업데이트되고 일부는 업데이트가 안될까?   
일부 입력이 0일 수 있지만 일부는 대부분의 교육 데이터에서 0이 아닌 값을 가질 수 있는 경우 어떻게 해야 할까?   
  
**이를 해결하기 위해서, Adaptive Learning Rates는 덜 업데이트된 매개변수에 대해 대규모 업데이트를 만들기 위해 나왔다.**

종류는 Adagrad, RMSProp, Adam, 등이 있으며 일반적으로 Adam과 RMSProp이 가장 많이 쓰인다.

더 자세히 알고 싶다면 다음을 참고하면 좋다. (<https://dev-jm.tistory.com/10>, <https://cs.kangwon.ac.kr/~leeck/AI2/RMSProp2.pdf>)

**Adagrad**

장점

1. 훨씬 덜 업데이트된 매개변수 업데이트  
2. 덜 업데이트된 매개변수 업데이트

단점

1. 결국 𝐺<sup>t</sup><sub>i</sub>는 시간이 지날수록 커짐

2. 매개변수는 때때로 거의 업데이트되지 않음

![](https://blog.kakaocdn.net/dna/9KLA3/btrX16QBkux/AAAAAAAAAAAAAAAAAAAAADWKechB0pgt3M4PTQwnAOQbS7aujzPuaFHnavjviVGL/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=s9NOJhomEAp%2BUauXOfY2nAMzFkI%3D)

Adagrad

**RMSProp**

Adagrad를 보완한 형태로 업데이트의 총량을 고려하는 대신 최근 업데이트의 양을 고려

![](https://blog.kakaocdn.net/dna/b0jLN7/btrX2poT4yv/AAAAAAAAAAAAAAAAAAAAAFxger-NGjNWOCs7KVsug0AqofkcfYkYNN1WFBTj_WaM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hYR1jmUzSqJ5lk%2FfL5DCJUEW0Ew%3D)

RMSProp

Adam

RMSProp과 Momentum이 적절히 석인 방법이다.

더보기

![](https://blog.kakaocdn.net/dna/uJch9/btrX2nR3Frv/AAAAAAAAAAAAAAAAAAAAAORQk7V0LihE4LhacX0xqyGcXZ4fAuWsKaosRvF7ByxB/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=er%2FMYGAIsK%2Bqh67KhcDU3r6Hzw8%3D)
![](https://blog.kakaocdn.net/dna/CN5Oi/btrX8hJksRH/AAAAAAAAAAAAAAAAAAAAANV25GXzamX64aTJNljVmVEtNKnVO3CbGWnHWVWmEun-/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=4B5YdwrKUpD0sx5Gu3eRbJuKYAY%3D)

![](https://blog.kakaocdn.net/dna/rLtww/btrX2m6GI9j/AAAAAAAAAAAAAAAAAAAAAFJOKhXKNqK8BWBliLA7fXQRRiDFgx1iFjCxcqpOWAe8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ac06U6T7BMIGFbo%2FLcQlnPVeeYM%3D)

Adam

다음의 그림을 통해 지금까지 배웠던 걸 쉽게 알아볼 수 있다.

![](https://blog.kakaocdn.net/dna/x7J9g/btrX32Nfgk2/AAAAAAAAAAAAAAAAAAAAAG2JEQG-bkbd0QS-qa8Xug3EB5g1nU9sG9zV6hO5M8Ms/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=89Y5YklXi658Xaj1Mcy7fIvwehg%3D)

Comparison: Long valley

![](https://blog.kakaocdn.net/dna/z2DtJ/btrX3txyfBN/AAAAAAAAAAAAAAAAAAAAAP90K06m0Ax_Sf7g-paDq6qAUe3mrxTNM1F5dsC-K--0/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=qDo32Z1QZ%2FvplQoIVMz6OHeA7cg%3D)

Comparison: Beale&rsquo;s Function

![](https://blog.kakaocdn.net/dna/lyIPT/btrX16weZmp/AAAAAAAAAAAAAAAAAAAAAH84NLnvgPzLbW5vmgkguW2_NxqOrXOC612VYpzrqAGl/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=R%2FrdQLkpBE3xGjFp2Mgzvidqyrc%3D)

Comparison: Saddle Point
