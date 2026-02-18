---
title: "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities"
date: 2025-12-14 20:26:53
categories:
  - 인공지능
---

<https://arxiv.org/abs/2503.14858?utm_source=pytorchkr&ref=pytorchkr>

[1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities](https://arxiv.org/abs/2503.14858?utm_source=pytorchkr&ref=pytorchkr)

자기지도 학습(self-supervised learning)의 규모 확장은 언어와 비전 분야에서 큰 돌파구를 이끌어냈지만, 강화학습(RL)에서는 이에 상응하는 발전이 아직 뚜렷하게 나타나지 않았다. **본 논문에서는 자기지도 강화학습의 확장성을 크게 향상시키는 핵심 구성 요소를 분석하며, 특히 네트워크 깊이가 결정적인 요인임을 보여준다.** 최근 수년간 대부분의 RL 연구들은 2~5층 정도의 얕은(shallow) 아키텍처에 의존해 왔으나, 우리는 네트워크 깊이를 최대 1024층까지 늘릴 경우 성능이 상당히 향상될 수 있음을 입증한다.

실험은 시연(demonstrations)이나 보상(rewards)이 전혀 주어지지 않는 비지도 목표 조건(unsupervised goal-conditioned) 설정에서 진행되며, 이 환경에서 에이전트는 완전히 처음부터 탐색을 수행하고 주어진 목표에 도달할 확률을 최대화하는 방법을 학습해야 한다. 시뮬레이션 기반 보행 및 조작(manipulation) 태스크에서 평가한 결과, 제안 방식은 자기지도 대비학습 기반 RL(contrastive RL) 알고리즘의 성능을 2배에서 최대 50배까지 향상시키며, 다른 목표 조건 기반 기법들을 능가했다. 모델 깊이를 증가시키는 것은 단순히 성공률을 높이는 데 그치지 않고, 에이전트가 학습하는 행동 양상 자체를 질적으로 변화시키는 것으로 나타났다. 프로젝트 웹페이지와 코드는 다음 링크에서 확인할 수 있다: <https://wang-kevin3290.github.io/scaling-crl/>.

![](/assets/images/posts/609/img.png)

그림 1 설명  
네트워크 깊이를 확장하면 다양한 보행, 내비게이션, 조작 태스크 전반에서 성능 향상이 나타난다. Ant Big Maze에서는 깊이 8층, Humanoid U-Maze에서는 깊이 64층과 같이 특정 임계 깊이를 지날 때 성능이 완만하게 증가하는 것이 아니라 급격히 도약하는 양상을 보이는데, 이는 질적으로 구별되는 정책이 그 깊이에서 새롭게 나타나기 때문이다(섹션 4 참고). 성능 향상 폭은 2배에서 Humanoid 기반 태스크에서는 최대 50배까지 도달한다.

### 1 서론

모델 규모를 확장하는 것은 기계학습의 많은 분야에서 효과적인 전략으로 자리 잡았지만, 강화학습(RL)에서 그 역할과 영향은 아직 명확하지 않다. 상태 기반 RL 태스크에서 사용되는 전형적인 모델 규모는 2~5개의 레이어 수준이다(Raffin et al., 2021; Huang et al., 2022). 반면, 다른 분야에서는 수백 개의 레이어로 구성된 매우 깊은 네트워크를 사용하는 것이 드문 일이 아니다. 예를 들어, Llama 3(Dubey et al., 2024)와 Stable Diffusion 3(Esser et al., 2024)은 모두 수백 개의 레이어를 포함하고 있다. 비전(Radford et al., 2021; Zhai et al., 2021; Dehghani et al., 2023)이나 언어(Srivastava et al., 2023)와 같은 분야에서는 모델이 특정 임계 규모(critical scale)를 넘어설 때 비로소 새로운 능력이 발현되는 경우가 흔하다.

RL 분야에서도 유사한 잠재적 발현 현상(emergent phenomena)을 찾기 위한 연구가 활발히 진행되어 왔다(Srivastava et al., 2023). 그러나 기존 논문들은 대체로 매우 제한적인 성능 향상만 보고하며, 그것도 작은 모델이 어느 정도 성능을 내는 태스크에 국한되는 경우가 많다(Nauman et al., 2024b; Lee et al., 2024; Farebrother et al., 2024). 따라서 RL에서 제기되는 핵심 연구 질문은 다음과 같다. 과연 RL 모델의 네트워크 규모를 확장함으로써 다른 분야에서 관찰된 것처럼 성능이 도약하는 현상을 재현할 수 있는가?

언뜻 보기에는 매우 큰 RL 네트워크를 학습시키는 일이 어려운 것이 당연해 보인다. RL 문제는 관측 시퀀스가 길게 이어진 후에야 희소한 보상(sparse reward)과 같은 극히 제한된 피드백만 제공하므로, 파라미터 대비 피드백의 비율이 매우 낮기 때문이다. 이러한 관점은 기존의 통념(LeCun, 2016)과 여러 최신 모델들(Radford, 2018; Chen et al., 2020; Goyal et al., 2019)에서도 반영되어 있다. 즉, 대규모 AI 시스템은 주로 자기지도(self-supervised) 방식으로 학습되어야 하고, 강화학습은 이러한 모델을 미세조정(finetuning)하는 데만 사용해야 한다는 것이다.

실제로 최근 다른 분야에서 이루어진 주요 성과들은 대부분 자기지도 학습을 통해 달성되었다. 이는 컴퓨터 비전(Caron et al., 2021; Radford et al., 2021; Liu et al., 2024), 자연어 처리(NLP) 분야(Srivastava et al., 2023), 그리고 멀티모달 학습(Zong et al., 2024) 모두에서 동일하게 관찰된다. 따라서 강화학습 기법을 확장하려면 자기지도 학습 기제가 핵심 요소가 될 가능성이 높다.

본 논문에서는 강화학습을 확장하기 위한 구성 요소들을 체계적으로 탐구한다. 첫 번째 단계는 기존 통념을 다시 생각하는 것이다. 앞서 언급한 “강화학습”과 “자기지도 학습”은 서로 대립적인 학습 규칙이 아니라, 보상 함수나 시연(demonstrations)에 의존하지 않고도 탐색과 정책 학습을 수행할 수 있는 자기지도 강화학습(self-supervised RL) 시스템으로 결합될 수 있다(Eysenbach et al., 2021, 2022; Lee et al., 2022). 본 연구에서는 이러한 자기지도 RL 알고리즘 중 가장 단순한 형태 중 하나인 대비학습 기반 RL(contrastive RL, CRL)(Eysenbach et al., 2022)을 사용한다.

두 번째 단계는 이용 가능한 데이터의 양을 극적으로 확대하는 것이다. 이를 위해 최근 제안된 GPU 가속 기반 RL 프레임워크(Makoviychuk et al., 2021; Rutherford et al., 2023; Rudin et al., 2022; Bortkiewicz et al., 2024)를 기반으로 한다. 세 번째 단계는 네트워크 깊이를 크게 증가시키는 것으로, 기존 연구에서 일반적으로 사용되던 모델보다 최대 100배 더 깊은 네트워크를 활용한다. 이러한 깊은 네트워크를 안정적으로 학습시키기 위해서는 잔차 연결(residual connections)(He et al., 2015), 레이어 정규화(layer normalization)(Ba et al., 2016), Swish 활성화 함수(Ramachandran et al., 2018)와 같은 아키텍처 기법들을 적용해야 한다.

또한 본 연구의 실험에서는 배치 크기(batch size)와 네트워크 폭(network width)이 성능에 미치는 상대적 중요성 역시 분석한다.

본 연구의 주요 기여는 이러한 구성 요소들을 하나의 강화학습 기법으로 통합했을 때, 매우 강한 확장성이 실험적으로 드러난다는 점을 보이는 것이다.

• **경험적 확장성(Empirical Scalability):** 전체 환경의 절반에서 성능이 20배 이상 향상되며, 기존의 표준 목표 조건(goal-conditioned) 기반 알고리즘들을 크게 능가한다. 이러한 성능 향상은 단순한 수치 증가가 아니라, 모델 규모가 커질 때 새롭게 등장하는 질적으로 구별되는 정책의 출현과 밀접하게 관련된다.

• **네트워크 깊이 확장(Scaling Depth in Network Architecture):** 기존 RL 연구들은 주로 네트워크 폭(width)을 늘리는 데 집중했으며, 깊이(depth)를 확장했을 때는 성능 향상이 제한적이거나 오히려 성능이 저하되는 결과가 흔히 보고되었다(Lee et al., 2024; Nauman et al., 2024b). 반면 본 연구의 접근법은 네트워크 깊이 축(depth axis)에서의 확장을 실질적으로 가능하게 만들며, 단순히 폭을 확장할 때보다 더 큰 성능 향상을 이끌어낸다(섹션 4 참조).

• **경험적 분석(Empirical Analysis):** 제안된 확장 접근 방식의 핵심 구성 요소들을 면밀히 분석하여, 어떤 요인들이 중요한 역할을 하는지 규명하고 새로운 통찰을 제공한다.

우리는 향후 연구가 본 연구의 기반 위에서 추가적인 구성 요소(building blocks)를 발굴하고 확장할 수 있을 것으로 기대한다.

### 2 관련 연구

자연어 처리(NLP)와 컴퓨터 비전(CV)은 최근 유사한 아키텍처(예: Transformer)와 공통 학습 패러다임(예: 자기지도 학습)을 채택하며 서로 수렴하고 있다. 이러한 결합된 흐름은 대규모 모델의 능력을 크게 확장시키는 데 핵심적 역할을 했다(Vaswani et al., 2017; Srivastava et al., 2023; Zhai et al., 2021; Dehghani et al., 2023; Wei et al., 2022). 반면, 강화학습(RL)에서 이와 유사한 수준의 발전을 이루는 일은 여전히 도전적이다.

여러 연구는 대규모 RL 모델의 확장을 어렵게 만드는 다양한 요인을 분석해 왔다. 여기에는 파라미터 활용 부족(parameter underutilization)(Obando-Ceron et al., 2024), 가소성과 용량 상실(plasticity and capacity loss)(Lyle et al., 2022, 2024), 데이터 희소성(data sparsity)(Andrychowicz et al., 2017; LeCun, 2016), 학습 불안정성(training instabilities)(Ota et al., 2021; Henderson et al., 2018; Van Hasselt et al., 2018; Nauman et al., 2024a) 등이 포함된다.

이러한 제약으로 인해 RL 모델 확장 관련 연구는 모방 학습(imitation learning)(Tuyls et al., 2024), 다중 에이전트 게임(multi-agent games)(Neumann and Gros, 2022), 언어 기반 RL(language-guided RL)(Driess et al., 2023; Ahn et al., 2022), 이산 행동 공간(discrete action spaces)(Obando-Ceron et al., 2024; Schwarzer et al., 2023) 등 특정 문제 영역에 제한되는 경우가 많다.

최근 연구들은 여러 유망한 방향을 제시하고 있다. 여기에는 새로운 아키텍처 패러다임(Obando-Ceron et al., 2024), 분산 학습(distributed training) 기법(Ota et al., 2021; Espeholt et al., 2018), 분포 기반 강화학습(distributional RL)(Kumar et al., 2023), 그리고 지식 증류(distillation)(Team et al., 2023) 등이 포함된다. 이러한 접근과 비교했을 때, 본 연구의 방법은 기존 자기지도 강화학습 알고리즘에 단순한 확장을 적용한 것이다.

이와 유사한 최신 연구로는 Lee et al. (2024)과 Nauman et al. (2024b)가 있으며, 이들은 잔차 연결(residual connections)을 적용해 더 넓은 네트워크(wider networks)의 학습을 용이하게 하고자 한다. 하지만 이들 연구는 네트워크 폭(width)에 주로 집중하고 있으며, 깊이를 늘렸을 때 얻을 수 있는 성능 향상은 제한적이라고 보고한다. 실제로 두 연구 모두 네 개의 MLP 레이어만 사용하는 얕은 아키텍처를 유지한다.

반면 본 연구에서는 네트워크 폭을 확장하면 성능이 향상된다는 점을 확인함과 동시에(섹션 4.4 참조), 깊이(depth) 방향으로의 확장 역시 가능하다는 점을 보여주며, 이는 폭만을 확장하는 것보다 더 강력한 성능 향상을 제공한다.

더 깊은 네트워크를 학습하려는 시도로는 Farebrother et al. (2024)의 연구가 대표적이다. 이들은 TD 목표(TD objective)를 이산화(discretization)하여 범주형 교차 엔트로피(categorical cross-entropy) 손실로 변환함으로써, 가치 기반 RL(value-based RL)을 분류 문제로 재정의하였다. 이러한 접근은 분류 기반 접근법이 회귀 기반 방식보다 더 견고하고 안정적이며, 따라서 확장성 측면에서도 우수할 가능성이 있다는 가설에 기반한다(Torgo and Gama, 1996; Farebrother et al., 2024).

본 연구에서 사용하는 CRL(Contrastive RL) 알고리즘 역시 실질적으로 교차 엔트로피 기반 손실을 사용한다(Eysenbach et al., 2022). 구체적으로, CRL의 InfoNCE 목적 함수는 교차 엔트로피 손실의 일반화 형태이며, 현재 상태와 행동이 목표 상태로 이어지는 동일한 궤적(trajectory)에 속하는지 여부를 분류(classification)하는 방식으로 RL 문제를 해결한다.

이런 관점에서 본 연구는 NLP 분야에서 교차 엔트로피 기반 분류가 대규모 모델 확장의 핵심 역할을 했던 것처럼, RL에서도 분류 기반 접근법이 중요한 구성 요소가 될 수 있음을 시사하는 두 번째 실증적 근거를 제공한다.

### 3 사전 지식(Preliminaries)

본 섹션에서는 목표 조건(goal-conditioned) 강화학습과 대비학습 기반 강화학습(contrastive RL)에 사용되는 기호와 정의를 소개한다. 본 연구의 초점은 온라인 강화학습(online RL)에 있으며, 여기서는 리플레이 버퍼(replay buffer)가 최신 궤적(trajectory)을 저장하고, 크리틱(critic)은 자기지도(self-supervised) 방식으로 학습된다.

### 목표 조건(goal-conditioned) 강화학습

목표 조건 MDP는  
ℳ\_g = (?, ?, p₀, p, p\_g, r\_g, γ)  
와 같은 튜플로 정의된다. 여기서 에이전트는 임의의 목표에 도달하기 위해 환경과 상호작용한다(Kaelbling, 1993; Andrychowicz et al., 2017; Blier et al., 2021).

각 시간 단계 t에서 에이전트는 상태  
sₜ ∈ ?  
를 관측하고, 그에 대응하는 행동  
aₜ ∈ ?  
를 수행한다. 초기 상태는  
p₀(s₀)  
에서 샘플링되며, 상호작용의 동역학은 전이 확률  
p(sₜ₊₁ ∣ sₜ, aₜ)  
로 정의된다.

목표 g ∈ ? 는 목표 공간 ? 에 속하며, 상태 공간 ? 와는 사상  
f: ? → ?  
을 통해 연관된다. 예를 들어, ? 는 상태 벡터의 일부 차원을 나타낼 수도 있다. 목표에 대한 사전분포(prior)는  
p\_g(g)  
로 정의된다.

보상 함수는 다음 단계에서 목표에 도달할 확률 밀도로 정의된다.

r\_g(sₜ, aₜ) ≜ (1 − γ) p(sₜ₊₁ = g ∣ sₜ, aₜ),

여기서 γ는 할인 계수(discount factor)이다.

이 설정에서 목표 조건 정책  
π(a ∣ s, g)  
은 환경의 현재 관찰 s와 목표 g를 모두 입력으로 받는다.

할인된 상태 방문 분포(discounted state visitation distribution)는 다음과 같이 정의된다.

p^π\_γ(⋅ ∣ ⋅, g)(s) ≜ (1 − γ) ∑\_{t=0}^{∞} γᵗ p^π\_t(⋅ ∣ ⋅, g)(s),

여기서 p^π\_t(s)는 정책 π가 목표 g 조건 하에서 정확히 t스텝 후 상태 s를 방문할 확률이다.

이 표현은 정책 π(⋅ ∣ ⋅, g)의 Q-함수와 정확히 동일하며, 목표 보상 r\_g에 대해 다음과 같이 쓸 수 있다.

Q^π\_g(s, a) ≜ p^π\_γ(⋅ ∣ ⋅, g)(g ∣ s, a).

목표는 기대 보상을 최대화하는 것이다.

max\_π ?\_{p₀(s₀), p\_g(g), π(⋅ ∣ ⋅, g)} [ ∑\_{t=0}^{∞} γᵗ r\_g(sₜ, aₜ) ]  
(1)

### 대비학습 기반 강화학습(Contrastive Reinforcement Learning)

본 연구의 실험에서는 대비학습 기반 강화학습(contrastive RL) 알고리즘(Eysenbach et al., 2022)을 사용해 목표 조건 강화학습 문제를 해결한다. Contrastive RL은 액터 크리틱(actor-critic) 방식으로 구성되며,  
f\_{ϕ,ψ}(s, a, g)를 크리틱(critic),  
π\_θ(a ∣ s, g)를 정책(policy)으로 정의한다.

크리틱은 두 개의 신경망으로 구성되며, 하나는 상태-행동 쌍 임베딩 ϕ(s, a), 다른 하나는 목표 임베딩 ψ(g)를 출력하도록 한다. 크리틱의 출력은 두 임베딩 간의 L2 거리로 정의된다.

f\_{ϕ,ψ}(s, a, g) = ‖ϕ(s, a) − ψ(g)‖₂.

크리틱은 InfoNCE 목적 함수(Sohn, 2016)를 사용해 학습되며, 이는 이전 연구(Eysenbach et al., 2021, 2022; Zheng et al., 2023, 2024; Myers et al., 2024; Bortkiewicz et al., 2024)와 동일하다.

학습은 배치 ℬ에서 이루어지며, 여기서  
sᵢ, aᵢ, gᵢ는 동일한 궤적 내에서 샘플된 상태, 행동, 목표(미래 상태)를 의미한다.  
반면 gⱼ는 서로 다른 무작위 궤적으로부터 샘플된 목표이다.

목적 함수는 다음과 같이 정의된다.

min\_{ϕ,ψ} ?\_ℬ [  
 − ∑{i=1}^{|ℬ|} log (  
  e^{f{ϕ,ψ}(sᵢ, aᵢ, gᵢ)}  
  / ∑{j=1}^{K} e^{f{ϕ,ψ}(sᵢ, aᵢ, gⱼ)}  
 )  
].

정책 π\_θ(a ∣ s, g)는 크리틱 출력을 최대화하도록 학습된다.

max\_{π\_θ} ?\_{p₀(s₀), p(s\_{t+1} ∣ s\_t, a\_t), p\_g(g), π\_θ(a ∣ s, g)}  
[ f\_{ϕ,ψ}(s, a, g) ].

---

**Contrastive RL(CRL)의 핵심 차이점은 ‘학습 신호의 형태(보상)’와 ‘Q 함수의 역할(거리 기반 목적)’이 달라졌다는 것이지만, 근본적인 강화학습 프레임워크 자체는 변화가 거의 없다.**

즉, **이론적 구조는 RL 그대로인데, 신호를 만드는 방식과 해석이 바뀐 것**이다.

아래에서 너가 궁금해하는 두 가지 차이가 기존 RL 이론과 정확히 어떻게 다른지 구조적으로 설명해줄게.

---
