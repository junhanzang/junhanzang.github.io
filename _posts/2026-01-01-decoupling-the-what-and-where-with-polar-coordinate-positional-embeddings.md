---
title: "Decoupling the 'What' and 'Where' With Polar Coordinate Positional Embeddings"
date: 2026-01-01 16:43:51
categories:
  - 인공지능
---

<https://arxiv.org/abs/2509.10534v2?utm_source=mail.bycloud.ai&utm_medium=newsletter&utm_campaign=rope-is-inherently-flawed&_bhlid=0527cfcf6b369e90c0e521fc6ecd4e9bf4a20c30>

[Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings](https://arxiv.org/abs/2509.10534v2?utm_source=mail.bycloud.ai&utm_medium=newsletter&utm_campaign=rope-is-inherently-flawed&_bhlid=0527cfcf6b369e90c0e521fc6ecd4e9bf4a20c30)

**초록 (Abstract)**

Transformer 아키텍처에서의 어텐션 메커니즘은 시퀀스 내에서의 **내용(the what)**과 **위치(the where)**라는 두 요소를 모두 기반으로 key와 query를 매칭한다. 본 논문에서는 널리 사용되는 **회전 위치 임베딩(Rotary Position Embedding, RoPE)**에서 이 두 요소가 서로 얽혀(entangled) 있음을 보이는 분석을 제시한다. 이러한 얽힘은 특히 내용과 위치에 대해 **독립적인 매칭**이 요구되는 결정 문제에서 성능 저하를 초래할 수 있다.

우리는 이 문제를 해결하기 위해 **극좌표 위치 임베딩(Polar Coordinate Position Embedding, PoPE)**이라 명명한 RoPE의 개선안을 제안하며, 이를 통해 what–where 간의 혼동(confound)을 제거한다. PoPE는 오직 **위치** 또는 **내용**만으로 인덱싱해야 하는 진단 과제에서 기존 방법 대비 월등한 성능을 보인다.

음악, 유전체, 자연어 도메인의 자기회귀 시퀀스 모델링에서, 위치 인코딩으로 PoPE를 사용하는 Transformer는 RoPE를 사용하는 기준 모델 대비 **평가 손실(퍼플렉시티)**과 **다운스트림 과제 성능** 모두에서 우수하다. 언어 모델링에서는 이러한 성능 향상이 **124M에서 774M 파라미터 규모**에 이르기까지 모델 크기 전반에 걸쳐 일관되게 유지된다. 특히 PoPE는 추가적인 파인튜닝과 주파수 보간이 필요한 **길이 외삽(length extrapolation)** 전용 방법인 YaRN과 비교하더라도, RoPE는 물론 YaRN보다도 강력한 **제로샷 길이 외삽 능력**을 보여준다.

## 1. 서론 (Introduction)

딥러닝의 초기 단계에서 중요한 도전 과제 중 하나는 **순차적 데이터** 또는 **위치가 부여된 데이터(position-coded data)**를 어떻게 표현할 것인가였다. 당시 관심의 대상이 된 과제들로는 문자 순서로부터 단어를 인식하는 문제(McClelland & Rumelhart, 1981), 파워 스펙트럼으로부터 음성을 인식하는 문제(Waibel et al., 1989), 예측 가능한 이벤트들 사이에 시간 지연이 존재하는 긴 시퀀스를 압축하는 문제(Schmidhuber et al., 1993), 스트로크 정보로부터 손글씨 기호를 분류하는 문제(Yaeger, 1996), 그리고 샘플로부터 시계열을 예측하는 문제(Mozer, 1993) 등이 있었다.

1980년대에는 두 가지 접근법이 주로 사용되었다. 하나는 **순환 신경망(Recurrent Neural Networks, RNNs)**이고, 다른 하나는 **슬롯 기반 인코딩(slot-based encodings)**이다. RNN은 입력의 순서를 통해 시퀀스상의 위치 정보를 **암묵적으로** 인코딩하는 방식이다. 반면, 슬롯 기반 인코딩은 입력 벡터를 시퀀스의 각 단계(step)에 대응되는 개별 구성 요소들로 분할한다. RNN의 중요한 장점 중 하나는 **근사적인 이동 등변성(translation equivariance)**을 가진다는 점인데, 이는 입력 내에서 어떤 부분 시퀀스의 절대 위치를 이동시키면 모델 상태 역시 이에 상응하는 방식으로 이동함을 의미한다. 슬롯 기반 표현은 이러한 성질을 공유하지 않는다. 즉, 하나의 슬롯에서 특정 내용에 반응하도록 학습되더라도, 동일한 내용이 다른 슬롯에 나타났을 때 그 지식이 일반화되지 않는다.

Transformer 아키텍처(Vaswani et al., 2017)는 슬롯 기반 표현의 판도를 바꾸었다. 자기 어텐션(self-attention) 메커니즘을 통해, 기본적인 Transformer는 (인과적 어텐션 경계를 고려할 경우) 이동 등변성뿐 아니라 **이동 불변성(translation invariance)**과 **순열 불변성(permutation invariance)**까지 갖는다. 따라서 Transformer에서의 핵심 과제는, 입력 요소들의 **상대적 위치**에 대한 민감성을 유지하면서도 이동 등변성을 어떻게 확보할 것인가가 되었다. 이를 해결하기 위해 등장한 방법들은 잠재 표현(latent representation)을 확장하여, 단순히 **내용(the what)**뿐만 아니라 시퀀스 위치 간의 관계, 즉 **위치(the where)**까지 함께 인코딩하도록 설계되었다(Vaswani et al., 2017; Dai et al., 2019; Su et al., 2024). 이러한 접근들은 언어와 같은 복잡한 시퀀스를 모델링하는 데 있어 내용과 위치가 모두 필수적이라는 점을 올바르게 전제하고 있다.

본 논문에서는 널리 채택되고 있는 방법인 **회전 위치 임베딩(Rotary Position Embedding, RoPE)**(Su et al., 2024)가 내용과 위치, 즉 what과 where를 서로 얽히게(entangle) 만들어, 특히 이 두 요소에 대해 **독립적인 매칭**이 필요한 의사결정 상황에서 모델 성능을 저하시킬 수 있음을 주장한다. 이에 대한 대안으로, 우리는 **PoPE**라 부르는 새로운 기법을 제안한다. PoPE는 다른 위치 인코딩 방식들에 비해 RoPE가 갖는 장점들을 그대로 유지하면서도, key–query 매칭을 **what 매칭과 where 매칭의 결합(conjunction)**으로 명확히 분리하여 규칙을 구현할 수 있도록 Transformer를 설계한다. PoPE는 RoPE에 대한 비교적 작은 수정에 불과하지만, 데이터 효율성, 점근적 정확도(asymptotic accuracy)를 향상시키고, 최신 방법들보다 우수한 **컨텍스트 길이 일반화(context-length generalization)** 성능을 제공하는 강력한 귀납적 편향(inductive bias)을 도입한다.

## 2. 배경 (Background)

RoPE(Su et al., 2024)는 **Llama 3**(Grattafiori et al., 2024), **DeepSeekv3**(Liu et al., 2024), **Gemma 3**(Team et al., 2025), **Qwen3**(Yang et al., 2025) 등을 포함한 다수의 최신 언어 모델에서 위치 정보를 통합하는 **지배적인 접근법**이다. RoPE는 각 query–key 쌍에 대해, 두 벡터가 얼마나 잘 매칭되는지뿐만 아니라 입력 시퀀스 내에서의 **상대적 위치**를 함께 반영한 어텐션 점수를 생성한다.

![](/assets/images/posts/614/img.png)

이며, θ는 **기본 파장(base wavelength)**이라 불린다. Query(또는 key) 성분의 구성과 2차원 회전 과정은 그림 1의 왼쪽에 도식화되어 있다.

![](/assets/images/posts/614/img_1.png)

RoPE에 대한 우리의 관점은 key와 query 성분을 **데카르트 좌표계(Cartesian coordinates)**에서 **극좌표계(polar coordinates)**로 재표현하는 것에서 출발한다. 즉,

![](/assets/images/posts/614/img_2.png)

이 표기를 사용하면 회전을 합성할 수 있고, 어텐션 점수는 다음과 같이 쓸 수 있다.

![](/assets/images/posts/614/img_3.png)

---

![](/assets/images/posts/614/img_4.png)

![](/assets/images/posts/614/img_5.png)

![](/assets/images/posts/614/img_6.png)

![](/assets/images/posts/614/img_7.png)

![](/assets/images/posts/614/img_8.png)

![](/assets/images/posts/614/img_9.png)

![](/assets/images/posts/614/img_10.png)

![](/assets/images/posts/614/img.jpg)

![](/assets/images/posts/614/img_1.jpg)

![](/assets/images/posts/614/img_2.jpg)

![](/assets/images/posts/614/img_3.jpg)

![](/assets/images/posts/614/img_4.jpg)

![](/assets/images/posts/614/img_5.jpg)

![](/assets/images/posts/614/img_6.jpg)

![](/assets/images/posts/614/img_7.jpg)

![](/assets/images/posts/614/img_8.jpg)

![](/assets/images/posts/614/img_9.jpg)

![](/assets/images/posts/614/img_10.jpg)

---
