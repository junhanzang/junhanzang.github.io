---
title: "Chapter3. Markov Decision Process"
date: 2023-05-21 17:04:39
tags:
  - Markov Decision Process
---

Markov Decision Processes (MDP)는 강화 학습을 위한 환경을 공식적으로 설명하며, 환경을 완전히 관찰할 수 있는 경우 거의 모든 RL 문제는 MDP로 공식화할 수 있다. 그렇다고 부분적으로 관찰 가능한 문제를 MDP로 변환할 수 없는 것은 아니다.

여기서부터는 action, state, reward를 행동, 환경, 보상으로 섞어쓸 것이다.

**Markov Property**

• **미래는 현재 주어진 과거와 독립적이다**  
• state는 기록에서 모든 관련 정보를 캡처한다  
• state가 알려지면 기록을 버릴 수 있다  
즉 . state는 미래에 대한 충분한 통계이다

![](https://blog.kakaocdn.net/dna/yRKAr/btsgKocA5Rw/AAAAAAAAAAAAAAAAAAAAAI5exKq8x6dKsz0RY0HmfMkITeZOt12G5TfDJJ6ClwvZ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=2QmzsV4AJ4KVpsq9JUAuqEe42r8%3D)

Markov&nbsp;Property

**State Transition Matrix(상태 전이 행렬)**

• Markov 상태 𝑠 및 후속 상태 𝑠′의 경우 상태 전이 확률은 다음과 같이 정의된다.

![](https://blog.kakaocdn.net/dna/cbwWE6/btsgDhNhkPg/AAAAAAAAAAAAAAAAAAAAAOkv1fvTAAAKeh5nSc5zvZPnqgcvmxvECb_JA54IWclW/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=PtPnJ1YvdKWQXjvOZ3E6B31ZRfs%3D)

• state 전환 매트릭스 𝑃는 모든 전환 확률을 정의한다  
• state s는 행렬의 각 행의 합이 1인 모든 후속 state s'이다

![](https://blog.kakaocdn.net/dna/btUe70/btsgE7v3hIw/AAAAAAAAAAAAAAAAAAAAAO5DtqUlwwJzguWuI51n_SFUnk665tcqv6NkufJDuHfF/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=tPJ5MUesg9F3YYlY%2BAIpJoKNbkw%3D)

**Markov Process**

마르코프 과정은 메모리가 없는 무작위 과정이다. 예를 들어, 마르코프 특성을 갖는 무작위 상태인 𝑆1, 𝑆2, . . . 의 순서대로 발생하는 시퀀스로 나타낼 수 있다.

![](https://blog.kakaocdn.net/dna/23b5K/btsgFuYSQgk/AAAAAAAAAAAAAAAAAAAAACLKSIRj8jl-PQQPM5xiIT-VllGJrTlWNMV-4_j6HG5X/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=nmJg9Npq%2FSdVvCZkPMBLW00wdak%3D)

**Example: Student Markov Chain**

아래의 마르코프 체인이 있다고 가정하자

![](https://blog.kakaocdn.net/dna/bXCAG8/btsgDOKQ726/AAAAAAAAAAAAAAAAAAAAADPfrDoMYodHvCAJFtUFTiA0Y1lmYDWO_zdSHanuSjIh/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hs%2FjJ8o0%2FSMTMjEXakFkkLuDcYA%3D)

전체가 관찰 가능하기 때문에, 우리가 정의했던 P를 통해 다음의 행렬을 얻을 수 있다.

![](https://blog.kakaocdn.net/dna/bUVRAi/btsgC14m4QK/AAAAAAAAAAAAAAAAAAAAAN0LCEwgelNLqFska-isco6IGfJgH-4q6PWH3TeXV3eq/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=lRmLvFZkdMFaepzJNJUucLC1vZM%3D)

**Markov Reward Process**

Markov 보상 프로세스는 값(Reward)이 있는 Markov 체인이다. 다음으로 정의된다.

![](https://blog.kakaocdn.net/dna/ba1pi2/btsgDOYjbes/AAAAAAAAAAAAAAAAAAAAAIuBKnomSbvt0TjQ9_kFTenIcRJL-JAdkpkW7ZgB6wQT/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=cGbU%2FA10cb7XBX9GX166JkodnFE%3D)

이 정의로 보면 state에 따른 discount factor를 통해 R을 구하는 방식이다. 일단 첫 step만 나타낸다면, 다음과 같이 나타낼수 있을것이다.

![](https://blog.kakaocdn.net/dna/XiiRG/btsgMbjT1y0/AAAAAAAAAAAAAAAAAAAAAH5ebsAZDJ4fYovg9xdsFwPTuq01LavmHPFqc5N8EBOV/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Z4judSfZ5mQIZY%2BcAzarPgpUT9g%3D)

**(Discounted) Return**

discount factor를 통해 R을 구하는 방식을 알아야되기 때문에 이를 확인하면 아래와 같다.

![](https://blog.kakaocdn.net/dna/moTaB/btsgEf9dkKN/AAAAAAAAAAAAAAAAAAAAAM3atapJxpXN3O-ABtY2wsyo49IujSaYYqdbyqW_RYks/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=MUTS6A3i2cVlyUuyMaTVmq%2F0VD4%3D)

• 할인율 𝛾∈[0,1]은 미래 보상의 현재 가치다  
• 𝑘+1단계 후 보상 𝑅을 받는 값은 𝛾𝑘𝑅이다  
• 이것은 지연된 보상보다 즉각적인 보상을 중요시한다  
• 𝛾이 0에 가까울수록 "근시" 평가  
• 𝛾이 1에 가까울수록 '원시적' 평가로 이어짐

왜 discount factor를 적용하는 것일까?

대부분의 Markov 보상 및 결정 프로세스는 할인된다 이는 순환 Markov 프로세스에서 무한 반환 방지를 위해서 그렇다. 추가적으로 미래에 대한 불확실성이 완전히 표현되지 않을 수 있고, 보상이 금전적인 경우 즉각적인 보상은 지연된 보상보다 더 많은 이자를 받을 수 있다. 일반적으로 동물/인간의 행동은 즉각적인 보상을 선호하기 때문에, 때때로 할인되지 않은 Markov 보상 프로세스를 사용할 수 있다.(예: 𝛾= 1 )

**Value Function**   
가치 함수 𝑣(𝑠)는 상태 𝑠의 장기적인 가치를 제공한다. 이는 아래 수식으로 확인할 수 있다.

![](https://blog.kakaocdn.net/dna/mluW8/btsgEdpEL7Y/AAAAAAAAAAAAAAAAAAAAANCiCQh-3chPEYgoe8wEyC09vfjOTpIpO_yKpSU7iDTp/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=azxWEqmOg9xW24PNLffqBgak94U%3D)

이전의 마르코프 체인에 discount factor와 value function을 적용하면 다음과 같다. 여기서 discount factor는 0.5이다.

![](https://blog.kakaocdn.net/dna/CQuiP/btsgEcEjP1P/AAAAAAAAAAAAAAAAAAAAANUimqOEJ8v8dLKtB6Ph9H1ucC19aOZeNnc-ER6SNveH/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=IM1lZzmPZlCHghsbmcaztUxifoM%3D)

Example: Student MRP Returns

discount factor에 따라 다시 구해보면 아래와 같은 그림이 된다.

![](https://blog.kakaocdn.net/dna/SwsIE/btsgJ0C0PvU/AAAAAAAAAAAAAAAAAAAAACCam5PuHhbJ_zVfyiR-GbRWymjN57xsESHMZZneiQ0x/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=xwE6kCrNA1bh3faVh9%2BaP2SWnG4%3D)

Example: State Value Function for Student MRP

**Bellman Equation for MRPs**

• 가치 함수는 두 부분으로 분해될 수 있다  
• 즉각적인 보상 𝑅𝑡+1  
• 후속 상태의 할인된 값 𝛾𝑣(𝑆𝑡+1)

이를 식으로 표한하면 다음과 같다.

![](https://blog.kakaocdn.net/dna/7wyd2/btsgQp95sZL/AAAAAAAAAAAAAAAAAAAAAGPt_gJIp-0yuzs7NpZnjMRKR14-c5VZJTJmzF7HklMu/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Y1ZuQLjblIgmfbMd47ayZFw7nLg%3D)

그림으로 표현하면 다음과 같이 표현될 것이다. 두 부분으로 분해되기 때문에

![](https://blog.kakaocdn.net/dna/qw02u/btsgDN6fHhE/AAAAAAAAAAAAAAAAAAAAANrpAT9MDRCHQpnUkS9RRWCbzmJhUSfrok168aXENwP4/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ZsoLFjwrIU2D2GXWuYYYjRXmhw4%3D)

**Markov Decision Process**

• Markov 결정 프로세스(MDP)는 결정이 포함된 Markov 보상 프로세스이다다. 따라서 모든 상태가 Markov인 환경이다.

![](https://blog.kakaocdn.net/dna/ddijd3/btsgQpI0Tk6/AAAAAAAAAAAAAAAAAAAAAOPibniivqOXqQGFrPkUpwSNobL7JtDn7u7qfezcLbTv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=MKP3yHZzsaPZF2slNX%2F3%2BQf7o44%3D)

Markov&nbsp;Decision&nbsp;Process

아까 예시를 들었던 student는 MDP로 볼 수 있다.

![](https://blog.kakaocdn.net/dna/OWQAE/btsgFeIr6t7/AAAAAAAAAAAAAAAAAAAAABr4rLMHbcE59G7xVf3jz_yyD5dC94KUVF44OA3yMuIA/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=%2F633JgBbBEl7%2F%2Fm%2B7L1XMUT6%2B9Y%3D)

Example: Student MDP

Policies

• 정책은 에이전트의 행동을 완전히 정의한다  
• MDP 정책은 현재 상태에 따라 달라진다(이전에 했던 행동이 아님)  
• 즉, 정책은 고정적(시간 독립적), 𝐴𝑡~𝜋(∙|𝑆𝑡),∀𝑡>0

![](https://blog.kakaocdn.net/dna/ZgUoi/btsgDKagudK/AAAAAAAAAAAAAAAAAAAAALS7ppgirt6IDJdj0vYAgqfkQ2HUKb4gOx6kiCAv_Tk1/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Tgqqd7DARAE3p1AdbjAK4EBDi5s%3D)

Policies

• 주어진 MDP =𝑆,𝐴,𝑃,𝑅,𝛾 및 정책 𝜋  
• 상태 시퀀스 𝑆1,𝑆2. . .는 마르코프 프로세스 <𝑆,𝑃𝜋>  
• 상태 및 보상 시퀀스 𝑆1,𝑅2,𝑆2, . . . Markov 보상 프로세스< 𝑆,𝑃𝜋,𝑅𝜋,𝛾>

따라서 식으로 표현하면 다음과 같다.

![](https://blog.kakaocdn.net/dna/coYzLa/btsgTipnBTR/AAAAAAAAAAAAAAAAAAAAAIqkHjaVJfIaEvlSDDtx3WLHIzFjdtoLuSRb__Kdv5cw/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=v2SCQrdggfi9f5wwaqzSQUGG6rI%3D)

**Value Function**   
value function은 state-value function과 action-value function이 있다.

1. 상태-가치 함수(State-Value Function): 상태-가치 함수는 주어진 상태에서 에이전트가 평균적으로 얻을 수 있는 기대 보상을 나타냅니다. 즉, 특정 상태에서 정책(policy)에 따라 행동을 선택하고 실행했을 때 얻을 수 있는 보상의 기댓값입니다. 상태-가치 함수는 다음과 같이 표현됩니다: V(s) = E[R | s], 여기서 V(s)는 상태 s에서의 가치를 나타내며, E는 기댓값을 의미하고, R은 에피소드(episode)에서 얻는 보상을 나타냅니다. 상태-가치 함수는 강화학습에서 가장 기본적이고 중요한 개념 중 하나입니다.
2. 행동-가치 함수(Action-Value Function): 행동-가치 함수는 특정 상태에서 특정 행동을 선택했을 때 얻을 수 있는 기대 보상을 나타냅니다. 즉, 상태와 행동의 조합에 따라 얻을 수 있는 보상의 기댓값입니다. 행동-가치 함수는 다음과 같이 표현됩니다: Q(s, a) = E[R | s, a], 여기서 Q(s, a)는 상태-행동 쌍 (s, a)에서의 가치를 나타내며, E는 기댓값을 의미하고, R은 에피소드에서 얻는 보상을 나타냅니다. 행동-가치 함수는 상태와 행동의 조합에 따른 가치를 평가하는 데 사용되며, 최적의 정책을 찾는 강화학습 알고리즘에서 중요한 역할을 합니다.

![](https://blog.kakaocdn.net/dna/tlylx/btsgG4MupdZ/AAAAAAAAAAAAAAAAAAAAALp_IVJoI0-HI8HjjizGTl8mZRlRUKSWjdDJBsaOyKxu/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=P6JMPQgRh50r2jXgHC%2BjEwFb0Lo%3D)

**Bellman Expectation Equation**   
• 상태 가치 함수는 다시 즉각적인 보상과 후속 상태의 할인된 가치로 분해될 수 있다.

![](https://blog.kakaocdn.net/dna/bIXMnL/btsgEDIRq1v/AAAAAAAAAAAAAAAAAAAAAEfhDcGAxdz2jaRs-MhLS1JhGmMkWni2kp7PE8hJB8KJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=b4QbUPGdjhMiOG%2Fig%2FU2eqU0M%2FI%3D)

• 액션 가치 함수도 유사하게 분해할 수 있다.

![](https://blog.kakaocdn.net/dna/bplo0a/btsgEDvkwnY/AAAAAAAAAAAAAAAAAAAAALZilgqHNwS9v250IqBeAP-H6OZAEhgfq4byKpN92-iR/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=LGHH05aJS7a5MN36P9p%2BWuMAWm4%3D)
![](https://blog.kakaocdn.net/dna/bMqxm6/btsgC4ND8fE/AAAAAAAAAAAAAAAAAAAAAIM1n5krd4ZEaetRHzq2xfzcvmS7ppwYv8a9xjmE0vso/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Pw3pimq9K1i%2FWuUOq72EvSPHXSI%3D)

상태 가치 함수 분해

![](https://blog.kakaocdn.net/dna/quQTN/btsgGme9aZn/AAAAAAAAAAAAAAAAAAAAADmm4BlwC6xHPsO3GQVcm2qqlC9DOCDt1IV8ZIzm583W/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=LRNIJSb3DRf3VerPWVGFO%2FzbXv0%3D)

액션 가치 함수 분해

이전의 Student MDP에 state-value fucntion을 적용하면 다음과 같다.

![](https://blog.kakaocdn.net/dna/bSokIW/btsgGj3NmrK/AAAAAAAAAAAAAAAAAAAAACzBrQfvx4ULNDskzNE8_vceN_qhGokKccbYrf4v_pQl/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=BNxKXmx8WOxqcxzDXn4nSrUgiGY%3D)

Example: State Value Function for Student MDP

![](https://blog.kakaocdn.net/dna/cDXEG1/btsgMbdeqin/AAAAAAAAAAAAAAAAAAAAAH_CiSpOddL2XIr-42pDI0agKRoNAqKaQkJYfi81iJy9/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=IVgiJUTfS5dVIJpuAvzR%2F2w8ufM%3D)

Example: State Value Function for Student MDP

**Bellman Expectation Equation in Matrix Form**

벨만 기대 방정식(Bellman Expectation Equation)은 현재 상태에서 다음 상태로 전이될 때의 가치와 보상을 고려하여 현재 상태의 가치를 예측한다.   
  
벨만 기대 방정식을 행렬 형태로 표현하면 다음과 같다:

V = R + γPV

• V는 현재 상태의 상태-가치 함수를 나타내는 열 벡터  
• R은 현재 상태에서 즉시 받는 보상을 나타내는 열 벡터  
• γ는 할인율(discount factor)로, 0과 1 사이의 값입니다. 미래의 보상을 현재보다 덜 가치있게 여기는 역할  
• P는 상태 전이 확률 행렬로, 상태 전이 확률을 나타내는 행렬  
따라서 벨만 기대 방정식은 현재 상태의 가치 V를 현재 상태에서 받는 즉시 보상 R과 다음 상태의 가치 P\*V를 고려하여 예측한다. 할인율 γ는 미래 보상의 중요성을 조절하는 역할을 하며, 높은 값일수록 미래 보상에 더 중점을 둔라고 이전에 설명했다.

따라서 식을 다음과 같이 전개하면 다음의 matricx form을 볼 수 있다.

![](https://blog.kakaocdn.net/dna/cJUOi0/btsgCCjvR7i/AAAAAAAAAAAAAAAAAAAAAJa4gzDj2KAkFkgmSpz60f_j0FYPcypZnAvJvBXmPmwv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=EW0mr3D3Ws3P%2B1rgWPJHezVEP3k%3D)

• 계산 복잡도는 n 상태에 대해 O(n^3 )  
• 소규모 MDP에만 가능한 다이렉트 솔루션  
• 대규모 MRP에 대한 많은 반복 방법이 있다

이제 기본적인 policy, reward등에 대해서 계산하는 방법들을 배웠다.

어떻게 reward를 maximize하고 최적의 policy를 찾고, 더 좋은 policy를 결정하는 법에 대해서 들어가보자.

**Optimal Value Function**

최적 가치 함수(Optimal Value Function)는 주어진 환경에서 에이전트가 얻을 수 있는 최대 기대 보상을 나타낸다.

최적 가치 함수를 표현하기 위해 두 가지 종류의 가치 함수를 정의할 수 있다:

1. 최적 상태-가치 함수(Optimal State-Value Function): 최적 상태-가치 함수는 각 상태에 대해 에이전트가 최적의 정책을 따랐을 때 얻을 수 있는 최대 기대 보상을 나타냅니다. 최적 상태-가치 함수는 다음과 같이 표현됩니다: V\*(s) = max[Q\*(s, a)], 여기서 V\*(s)는 상태 s에서의 최적 가치를 나타내며, Q\*(s, a)는 최적 행동-가치 함수를 의미합니다. max 연산은 가능한 모든 행동에 대한 가치를 비교하여 최대 값을 선택하는 것을 의미합니다.
2. 최적 행동-가치 함수(Optimal Action-Value Function): 최적 행동-가치 함수는 각 상태와 행동 조합에 대해 에이전트가 최적의 정책을 따랐을 때 얻을 수 있는 최대 기대 보상을 나타냅니다. 최적 행동-가치 함수는 다음과 같이 표현됩니다: Q\*(s, a) = max[R + γV\*(s')], 여기서 Q\*(s, a)는 상태-행동 쌍 (s, a)에서의 최적 가치를 나타내며, R은 즉시 받는 보상을 의미하고, γ는 할인율(discount factor)입니다. V\*(s')는 다음 상태 s'에서의 최적 상태-가치 함수를 의미합니다.

최적 가치 함수를 찾기 위해서는 가치 함수를 초기화하고 벨만 최적 방정식(Bellman Optimality Equation)을 반복적으로 적용하여 가치 함수를 업데이트한다. 이를 통해 최적 가치 함수를 점진적으로 추정하고, 최적의 정책을 찾을 수 있다.

![](https://blog.kakaocdn.net/dna/bxkcR6/btsgKpo7V84/AAAAAAAAAAAAAAAAAAAAAJdP73jRMcZkAgWwX27-TR8rrUoq_rkXrz7Ea1s16bGS/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=88hL2cC8kq4Y6l1DFXEvMXs8%2B3s%3D)

**Optimal Policy**

최적 정책(Optimal Policy)은 최적 가치 함수(Optimal Value Function)를 기반으로 결정된다.  
강화학습에서 정책은 상태(state)에 따라 에이전트가 선택해야 할 행동(action)을 결정하는 매핑 함수로 나타낸다. 최적 정책은 에이전트가 상태에 따라 선택해야 할 최적의 행동을 결정하는 정책이다.  
최적 정책은 최적 가치 함수와 관련되어 있습니다. 최적 상태-가치 함수(V\*(s))를 사용하는 최적 정책은 각 상태에서 최적의 행동을 선택하는 방식으로 정의된다. 즉, 최적 정책에서는 각 상태에 대해 최대의 상태-가치를 갖는 행동이 선택됩니다. 따라서 최적 상태-가치 함수를 통해 최적 정책을 알 수 있다.  
또한, 최적 행동-가치 함수(Q\*(s, a))를 사용하는 최적 정책은 상태와 행동의 조합에 따라 최적의 행동을 선택하는 방식으로 정의된다. 최적 행동-가치 함수에서는 각 상태와 행동 조합에 대해 최대의 행동-가치를 갖는 행동이 선택된다. 따라서 최적 행동-가치 함수를 통해 최적 정책을 알 수 있다.  
최적 정책을 찾기 위해서는 최적 가치 함수를 추정하고, 추정된 최적 가치 함수를 기반으로 최적 정책을 결정한다. 주로 가치 반복(Value Iteration)이나 정책 반복(Policy Iteration)과 같은 알고리즘을 사용하여 최적 가치 함수와 최적 정책을 동시에 추정하고 개선한다.  
따라서 최적 정책은 강화학습에서 에이전트가 환경과 상호작용하며 최대의 보상을 얻을 수 있는 행동 선택을 도와준다.

![](https://blog.kakaocdn.net/dna/lDUnq/btsgNYdHai1/AAAAAAAAAAAAAAAAAAAAAMHhjXnaTQdWw9KPsziVgGqBYyj-2va-wzTarlotU5S_/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hVLHwUJSOc4P32fmmU56pCr51l8%3D)
![](https://blog.kakaocdn.net/dna/pG9p7/btsgEazMjsj/AAAAAAAAAAAAAAAAAAAAAMEKTAs7EZPPQJLOHtDAFH_lDedGNEc0Qaez-s5M-LTm/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=insA4dU6td4v0rHmB0fmu1CFu6k%3D)

**Bellman Optimality Equations**

벨만 최적 방정식(Bellman Optimality Equations)은 강화학습에서 최적 가치 함수와 최적 정책을 찾기 위해 사용되는 중요한 방정식이다. 벨만 최적 방정식은 현재 상태에서 가능한 모든 행동에 대해 최적 가치를 계산하기 위한 재귀적인 관계를 제공한다.

벨만 최적 방정식을 상태-가치 함수와 행동-가치 함수로 나누어 설명할 수 있다.

1. 상태-가치 함수의 벨만 최적 방정식: 최적 상태-가치 함수의 벨만 최적 방정식은 다음과 같이 표현됩니다: V\*(s) = max[Q\*(s, a)], 여기서 V\*(s)는 상태 s에서의 최적 상태-가치를 나타내며, Q\*(s, a)는 최적 행동-가치 함수를 의미합니다. 최적 상태-가치 함수는 현재 상태에서 가능한 모든 행동에 대한 최적 행동-가치를 고려하여 가장 높은 가치를 선택합니다.
2. 행동-가치 함수의 벨만 최적 방정식: 최적 행동-가치 함수의 벨만 최적 방정식은 다음과 같이 표현됩니다: Q\*(s, a) = max[R + γV\*(s')], 여기서 Q\*(s, a)는 상태-행동 쌍 (s, a)에서의 최적 가치를 나타내며, R은 즉시 받는 보상을 의미하고, γ는 할인율(discount factor)입니다. V\*(s')는 다음 상태 s'에서의 최적 상태-가치 함수를 의미합니다. 최적 행동-가치 함수는 현재 상태와 행동 조합에 대해 가능한 모든 다음 상태에서의 최적 상태-가치를 고려하여 가장 높은 가치를 선택합니다.

벨만 최적 방정식은 최적 가치 함수를 업데이트하기 위해 사용된다. 주어진 환경에서 최대의 보상을 얻기 위해 최적 가치 함수를 반복적으로 추정하고 개선하는 데 사용된다. 이를 통해 최적 가치 함수를 구하고, 이에 기반하여 최적 정책을 결정할 수 있다. 벨만 최적 방정식은 값 반복(Value Iteration)과 정책 반복(Policy Iteration)과 같은 알고리즘에서 주요한 개념으로 활용된다. 따라서 다음 그림이 유효하다.

![](https://blog.kakaocdn.net/dna/2j16g/btsgEdKqzty/AAAAAAAAAAAAAAAAAAAAACN_GLNx374shAEs7NACuTOChwPOjdKkMVERTbucQS28/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=fomuHrQ3cy8ZvZe3%2Fb8rKEmkL5M%3D)

상태-가치 함수의 벨만 최적 방정식

![](https://blog.kakaocdn.net/dna/dwt7oy/btsgQqOG6lf/AAAAAAAAAAAAAAAAAAAAAJv6C22-OoxgdE-IEjrf_Gl3WsgJHAjqd7VRKzzV5wrq/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=KdqEDtQmfdqv7J3slNhhmVTDnoo%3D)

행동-가치 함수의 벨만 최적 방정식

**Solving the Bellman Optimality Equation**

벨만 최적 방정식(Bellman Optimality Equation)을 해결하는 것은 강화학습에서 최적 가치 함수를 추정하고, 이를 통해 최적 정책을 결정하는 과정이다. 벨만 최적 방정식을 해결하기 위해서는 반복적인 방법을 사용한다.

벨만 최적 방정식을 해결하는 일반적인 접근 방법은 다음과 같다:

1. 초기화: 최적 가치 함수를 초기값으로 설정합니다. 초기값으로는 임의의 값을 사용하거나 모든 상태에 대해 초기값을 동일하게 설정할 수 있습니다.
2. 가치 함수 업데이트: 벨만 최적 방정식을 기반으로 현재의 가치 함수를 업데이트합니다. 최적 상태-가치 함수인 경우에는 다음과 같이 업데이트됩니다: V\_{k+1}(s) = max\_a [Q\_k(s, a)], 여기서 V\_{k+1}(s)는 k+1번째 반복에서의 상태 s의 가치를 나타내며, Q\_k(s, a)는 k번째 반복에서의 행동-가치 함수를 의미합니다. 모든 상태에 대해 최적 행동-가치를 계산하여 각 상태의 최적 상태-가치를 업데이트합니다.
3. 행동-가치 함수 업데이트: 최적 행동-가치 함수를 업데이트합니다. 최적 행동-가치 함수인 경우에는 다음과 같이 업데이트됩니다: Q\_{k+1}(s, a) = R(s, a) + γV\_k(s'), 여기서 Q\_{k+1}(s, a)는 k+1번째 반복에서의 상태-행동 쌍 (s, a)의 가치를 나타내며, R(s, a)는 즉시 받는 보상을 의미하고, γ는 할인율(discount factor)입니다. V\_k(s')는 k번째 반복에서의 다음 상태 s'의 가치를 의미합니다.
4. 수렴 검사: 가치 함수가 충분히 수렴할 때까지 2번과 3번의 단계를 반복합니다. 가치 함수가 수렴하는 조건은 알고리즘의 종료 조건으로 설정되는데, 일반적으로는 두 가치 함수 간의 차이가 작아지거나 일정한 값 미만이 될 때 수렴으로 간주합니다.
5. 정책 결정: 최적 가치 함수를 기반으로 최적 정책을 결정합니다. 최적 상태-가치 함수인 경우에는 각 상태에서 최대 가치를 가지는 행동을 선택하여 최적 정책을 구성합니다. 최적 행동-가치 함수인 경우에는 각 상태와 행동 조합에서 최대 가치를 가지는 행동을 선택하여 최적 정책을 구성합니다.

벨만 최적 방정식을 해결하는 것은 최적 가치 함수와 최적 정책을 찾는 과정이다. 이를 통해 강화학습 에이전트는 주어진 환경에서 최대의 보상을 얻을 수 있는 최적의 행동 선택을 학습하게 된다.

반복으로 푸는 방식은 다음과 같다.

• Policy Iteration   
• Value Iteration   
• SARSA   
• Q-Learning

![](https://blog.kakaocdn.net/dna/wdKLs/btsgECDcsQm/AAAAAAAAAAAAAAAAAAAAAI1QrqZnEhbhpGe7ataL2tuTOlvh_J-IeDhJ1LAjAYOq/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=t%2FFCgdtgwZMAFvibGCkNJmRk73s%3D)

Bellman Expectation vs. Optimality

지금까지 했던걸 예제로 적용하면 다음과 같다.

예제 환경은 다음과 같다.

![](https://blog.kakaocdn.net/dna/dXBiPb/btsgNW71znN/AAAAAAAAAAAAAAAAAAAAAPj4MqvJd7vpaOxssz3ai2j-lkz3r2c8QRr_xHTb0CV6/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=GXRvYZ39EOCKcFRHXOgMjRtMCY8%3D)
![](https://blog.kakaocdn.net/dna/InNP4/btsgDNZAT1O/AAAAAAAAAAAAAAAAAAAAADWEHlXeeED_3dNK_pzx1soRFLpq1BPr6SLULORc7gje/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hmWlCP1jkMzU9p7rn4XTWhv5hSI%3D)

State Value Function

![](https://blog.kakaocdn.net/dna/bBGjcw/btsgDMy84mO/AAAAAAAAAAAAAAAAAAAAALDccyOmMA_9y5zKyb0fpR9COBxkvK_pzopkZcXl9-Mu/img.gif?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=cKnvsIc7HkBUGWsNhl%2FiU%2FsqsNg%3D)

Bellman Equations

![](https://blog.kakaocdn.net/dna/sriak/btsgE6RzfW6/AAAAAAAAAAAAAAAAAAAAABI3-jfFUJpz4AhLidEiPgaSaZlyAHyXEbiNRiPT1eT_/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=z2GKeXbQghKNmrrSYrM%2BhyY5yu8%3D)
![](https://blog.kakaocdn.net/dna/bnh6sz/btsgC1pPnzC/AAAAAAAAAAAAAAAAAAAAAFndQyC7tzr5AIYOBsXtbhgL2OVH6U72I2FBjYU_F7Mb/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=e6ojC%2FPWp0hCi4PRBgYD0vp1WV0%3D)
![](https://blog.kakaocdn.net/dna/r7mvj/btsgEg8g89j/AAAAAAAAAAAAAAAAAAAAAH-Os9Tf55sCxA4Hx7Ry9rmSkS29B5HooJwsk63sMYt9/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ufpxbvzTp%2FPWwYG2y3Oy3p8uT5c%3D)
