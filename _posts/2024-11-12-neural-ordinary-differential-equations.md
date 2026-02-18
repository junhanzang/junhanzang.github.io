---
title: "Neural Ordinary Differential Equations"
date: 2024-11-12 18:27:46
categories:
  - 인공지능
---

<https://arxiv.org/abs/1806.07366>

[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

초록  
우리는 새로운 계열의 심층 신경망 모델을 소개합니다. 은닉층의 이산적인 순서를 명시하는 대신, 우리는 은닉 상태의 미분을 신경망을 사용하여 매개변수화합니다. 네트워크의 출력은 블랙박스 미분방정식 솔버를 사용하여 계산됩니다. 이러한 연속-깊이 모델은 일정한 메모리 비용을 가지며, 각 입력에 따라 평가 전략을 조정할 수 있고, 수치적 정밀도를 속도와 명시적으로 교환할 수 있습니다. 우리는 이러한 속성을 연속-깊이 잔여 네트워크와 연속-시간 잠재 변수 모델에서 입증합니다. 또한, 데이터를 차원으로 나누거나 순서를 지정하지 않고 최대 우도 추정에 의해 학습할 수 있는 생성 모델인 연속 정규화 흐름을 구성합니다. 학습을 위해, 우리는 내부 연산에 접근하지 않고도 임의의 ODE(상미분 방정식) 솔버를 통해 확장 가능하게 역전파하는 방법을 보여줍니다. 이는 더 큰 모델 내에서 ODE의 종단간 학습을 가능하게 합니다.

Residual Network

![](/assets/images/posts/295/img.png)

ODE Network

![](/assets/images/posts/295/img_1.png)

그림 1:  
왼쪽: 잔차 네트워크(Residual Network)는 유한한 변환의 이산적인 순서를 정의합니다.  
오른쪽: ODE 네트워크는 상태를 연속적으로 변환하는 벡터 필드를 정의합니다.  
양쪽 모두: 원은 평가 위치를 나타냅니다.

**1. 서론**  
잔차 네트워크, 순환 신경망 디코더, 정규화 흐름과 같은 모델들은 은닉 상태에 일련의 변환을 구성함으로써 복잡한 변환을 만듭니다:

![](/assets/images/posts/295/img_2.png)

![](/assets/images/posts/295/img_3.png)

층을 더 많이 추가하고 작은 단계로 나아가면 어떻게 될까요? 극한에서는 은닉 유닛의 연속적인 동역학을 신경망으로 정의된 상미분 방정식(ODE)을 사용해 매개변수화할 수 있습니다:

![](/assets/images/posts/295/img_4.png)

입력 층 h(0)에서 시작하여, 우리는 출력 층 h(T)을 ODE 초기값 문제의 시간 T에서의 해로 정의할 수 있습니다. 이 값은 블랙박스 미분 방정식 솔버에 의해 계산될 수 있으며, 원하는 정확도로 솔루션을 결정하기 위해 필요한 곳에서 은닉 유닛의 동역학 f을 평가합니다. 그림 1은 이 두 접근 방식을 대조하고 있습니다.

ODE 솔버를 사용하여 모델을 정의하고 평가하는 것은 여러 가지 이점을 제공합니다:

**메모리 효율성**  
2절에서 우리는 어떤 ODE 솔버의 모든 입력에 대해 스칼라 값 손실의 기울기를 계산하는 방법을 보여줍니다. 여기서 솔버의 연산을 통해 역전파(backpropagation)하지 않으면서도 가능합니다. 순방향 패스의 중간 값을 저장하지 않음으로써, 깊이에 따른 메모리 비용을 일정하게 유지하면서 모델을 학습할 수 있습니다. 이는 심층 모델 학습의 주요 병목 문제를 해결하는 데 기여합니다.

**적응형 계산**  
오일러 방법은 ODE를 해결하는 가장 간단한 방법일 것입니다. 이후 120년 이상 동안 효율적이고 정확한 ODE 솔버들이 개발되어 왔습니다 (Runge, 1895; Kutta, 1901; Hairer et al., 1987). 현대 ODE 솔버는 근사 오차의 증가에 대한 보장을 제공하며, 오차 수준을 모니터링하고 평가 전략을 즉석에서 조정하여 요청된 정확도를 달성합니다. 이를 통해 모델 평가 비용이 문제의 복잡도에 맞게 확장될 수 있습니다. 학습 후에는 실시간 또는 저전력 애플리케이션을 위해 정확도를 낮출 수 있습니다.

**확장 가능하고 가역적인 정규화 흐름**  
연속적인 변환의 예상치 못한 부수적인 이점은 변수 변화 공식이 더 쉽게 계산된다는 점입니다. 4절에서 우리는 이 결과를 도출하고 이를 사용해 정규화 흐름의 단일 유닛 병목을 피하는 새로운 클래스의 가역적인 밀도 모델을 구성하며, 최대 우도 추정을 통해 직접 학습할 수 있습니다.

**연속 시간 시계열 모델**  
순환 신경망(RNN)과 달리, 관찰 및 방출 간격을 이산화할 필요 없이 연속적으로 정의된 동역학은 임의의 시간에 도착하는 데이터를 자연스럽게 통합할 수 있습니다. 5절에서 우리는 이러한 모델을 구성하고 시연합니다.

**2. ODE 해의 역방향 자동 미분**  
연속-깊이 네트워크를 학습하는 주요 기술적 어려움은 ODE 솔버를 통해 역방향 미분(역전파라고도 함)을 수행하는 것입니다. 순방향 연산을 통해 미분하는 것은 비교적 간단하지만, 높은 메모리 비용을 유발하고 추가적인 수치적 오류를 초래할 수 있습니다.

우리는 ODE 솔버를 블랙박스로 취급하고, 접합 감도 방법(adjont sensitivity method, Pontryagin et al., 1962)을 사용해 기울기를 계산합니다. 이 접근법은 두 번째, 증강된 ODE를 시간 역방향으로 풀어 기울기를 계산하며, 모든 ODE 솔버에 적용될 수 있습니다. 이 방법은 문제 크기에 대해 선형적으로 확장되며, 낮은 메모리 비용을 가지고, 수치적 오류를 명시적으로 제어합니다.

스칼라 값 손실 함수 L(⋅)을 최적화하는 것을 고려해 봅시다. 이 함수의 입력은 ODE 솔버의 결과입니다:

![](/assets/images/posts/295/img_5.png)

![](/assets/images/posts/295/img_6.png)

그림 2: ODE 해의 역방향 미분. 접합 감도 방법은 증강된 ODE를 시간 역방향으로 풉니다. 증강된 시스템은 원래의 상태와 손실이 상태에 대해 가지는 감도 모두를 포함합니다. 손실이 여러 관찰 시간에서 상태에 직접적으로 의존하는 경우, 접합 상태는 각 관찰에 대한 손실의 편미분 방향으로 업데이트되어야 합니다.

![](/assets/images/posts/295/img_7.png)

![](/assets/images/posts/295/img_8.png)

![](/assets/images/posts/295/img_9.png)

-----

<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({
startOnLoad: true,
theme: 'default'
});
</script>
<div class="mermaid">
flowchart LR
subgraph Forward["정방향 계산 t0 → t1"]
A["z(t0)"] -->|"dz/dt = f(z(t),t,θ)"| B["z(t1)"]
end
subgraph Backward["역방향 계산 t1 → t0"]
C["z(t1), a(t1), 0"] -->|"Augmented ODE Solver"| D["z(t0), ∂L/∂z(t0), ∂L/∂θ"]
subgraph Aug["증강된 상태"]
E["z(t): 상태"]
F["a(t): adjoint"]
G["∂L/∂θ: 파라미터<br/>그래디언트"]
end
end
B --> C
style Forward fill:#e1f5fe
style Backward fill:#fff3e0
style Aug fill:#f3e5f5
</div>
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
<!-- 배경 그리드 -->
<defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
<path d="M 40 0 L 0 0 0 40" fill="none" stroke="#eee" stroke-width="1"/>
</pattern>
</defs>
<rect width="800" height="400" fill="url(#grid)"/>
<!-- 축 -->
<line x1="50" y1="350" x2="750" y2="350" stroke="black" stroke-width="2"/>
<line x1="50" y1="50" x2="50" y2="350" stroke="black" stroke-width="2"/>
<!-- 축 레이블 -->
<text x="400" y="390" text-anchor="middle">Time</text>
<text x="30" y="200" transform="rotate(-90,30,200)" text-anchor="middle">State</text>
<text x="70" y="370">t0</text>
<text x="720" y="370">t1</text>
<!-- 정방향 궤적 -->
<path d="M 50 300 C 250 280, 450 200, 750 100"
fill="none" stroke="#2196F3" stroke-width="3"/>
<!-- 역방향 궤적 (adjoint) -->
<path d="M 750 100 C 450 150, 250 220, 50 250"
fill="none" stroke="#FF5722" stroke-width="3" stroke-dasharray="5,5"/>
<!-- 레전드 -->
<circle cx="600" cy="30" r="5" fill="#2196F3"/>
<text x="620" y="35">Forward trajectory</text>
<line x1="600" y1="60" x2="640" y2="60" stroke="#FF5722" stroke-width="3" stroke-dasharray="5,5"/>
<text x="660" y="65">Adjoint trajectory</text>
</svg>

-----

![](/assets/images/posts/295/img_10.png)

-----

github.com/rtqichen/torchdiffeq

-----

3. 잔차 네트워크를 ODE로 대체하여 지도 학습 수행하기

이 섹션에서는 지도 학습을 위해 신경 ODE를 학습하는 실험을 진행합니다.

**소프트웨어**  
ODE 초기값 문제를 수치적으로 해결하기 위해 우리는 LSODE와 VODE에 구현된 암시적 애덤스 방법(implicit Adams method)을 사용하며, 이는 scipy.integrate 패키지를 통해 인터페이스 됩니다. 암시적 방법이기 때문에, Runge-Kutta와 같은 명시적 방법보다 더 나은 보장을 제공하지만, 매 단계에서 비선형 최적화 문제를 해결해야 합니다. 이러한 설정은 통합기(integrator)를 통한 직접적인 역전파(backpropagation)를 어렵게 만듭니다. 우리는 Python의 autograd 프레임워크(Maclaurin et al., 2015)에 접합 감도 방법을 구현했습니다. 이 섹션의 실험에서 우리는 은닉 상태의 동역학과 그 편미분을 TensorFlow를 사용해 GPU에서 평가하였으며, 이는 Python의 autograd 코드에서 호출된 Fortran ODE 솔버로 전달되었습니다.

표 1: MNIST에 대한 성능. †는 LeCun et al. (1998)에서 가져옴.

![](/assets/images/posts/295/img_11.png)

![](/assets/images/posts/295/img_12.png)

ODE-Net에서의 오류 제어  
ODE 솔버는 출력이 실제 해와 주어진 오차 허용 범위 내에 있도록 보장할 수 있습니다. 이 허용 범위를 변경하면 네트워크의 동작이 변합니다. 먼저, 그림 3a에서 오류가 실제로 제어될 수 있음을 확인했습니다. 순방향 호출에 걸리는 시간은 함수 평가 수에 비례합니다(그림 3b). 따라서 허용 범위를 조정하면 정확도와 계산 비용 사이의 절충점을 얻을 수 있습니다. 학습할 때는 높은 정확도로 학습하고, 테스트 시에는 낮은 정확도로 전환할 수도 있습니다.

![](/assets/images/posts/295/img_13.png)

(a) NFE Foward

![](/assets/images/posts/295/img_14.png)

(b) NFE Foward

![](/assets/images/posts/295/img_15.png)

(c) NFE Foward

![](/assets/images/posts/295/img_16.png)

(d) Training Epoch

![](/assets/images/posts/295/img_17.png)

그림 3: 학습된 ODE-Net의 통계. (NFE = 함수 평가 수)

그림 3c는 놀라운 결과를 보여줍니다: 역전파 과정에서의 평가 수가 순방향 과정의 약 절반이라는 것입니다. 이는 접합 감도 방법이 메모리 효율적일 뿐만 아니라, 통합기를 통해 직접 역전파하는 것보다 계산적으로도 더 효율적임을 시사합니다. 후자의 접근법은 순방향 과정에서의 각 함수 평가에 대해 역전파를 필요로 하기 때문입니다.

**네트워크 깊이**  
ODE 해의 '깊이'를 어떻게 정의해야 할지는 명확하지 않습니다. 관련된 양은 은닉 상태의 동역학을 평가하는 데 필요한 횟수로, 이는 ODE 솔버에 위임된 세부 사항이며 초기 상태나 입력에 따라 달라집니다. 그림 3d는 모델의 복잡성이 증가함에 따라 함수 평가 횟수가 학습 내내 증가하는 것을 보여줍니다. 이는 모델의 복잡성에 적응한 결과로 보입니다.

**4. 연속 정규화 흐름**

![](/assets/images/posts/295/img_18.png)

![](/assets/images/posts/295/img_19.png)

놀랍게도, 이산적인 층 집합에서 연속 변환으로 이동하면 정규화 상수(normalizing constant)의 변화를 계산하는 과정이 단순해집니다.

**정리 1 (순간적인 변수 변환):**

![](/assets/images/posts/295/img_20.png)

-----

![](/assets/images/posts/295/img_21.png)

-----

**여러 은닉 유닛을 사용하여 선형 비용으로 평가하기**

![](/assets/images/posts/295/img_22.png)

**시간 의존적인 동역학**

![](/assets/images/posts/295/img_23.png)

4.1 연속 정규화 흐름 실험

![](/assets/images/posts/295/img_24.png)

그림 4: 정규화 흐름과 연속 정규화 흐름의 비교. 정규화 흐름의 모델 용량은 깊이(K)에 의해 결정되지만, 연속 정규화 흐름은 폭(M)을 늘림으로써 용량을 증가시킬 수 있어 학습이 더 쉽습니다.

우리는 먼저 연속 평면 흐름과 이산 평면 흐름을 비교하여 알려진 분포에서 샘플을 학습하는 능력을 평가합니다. M개의 은닉 유닛을 가진 평면 CNF가 K=M개의 층을 가진 평면 NF만큼이나 표현력이 있을 수 있으며, 때로는 훨씬 더 표현력이 있다는 것을 보였습니다.

**밀도 매칭**

![](/assets/images/posts/295/img_25.png)

**최대 우도 학습**

![](/assets/images/posts/295/img_26.png)

그림 6: 노이즈에서 데이터로의 변환 시각화. 연속-시간 정규화 흐름은 가역적이므로 밀도 추정 작업에서 학습한 후에도 학습된 밀도에서 효율적으로 샘플링할 수 있습니다.

![](/assets/images/posts/295/img_27.png)

이 작업을 위해 우리는 CNF에 대해 64개의 은닉 유닛을 사용하고, NF에 대해 64개의 하나의 은닉 유닛을 가지는 층을 쌓아올렸습니다. 그림 6은 학습된 동역학을 보여줍니다. 초기 가우시안 분포 대신, 초기 평면 흐름의 위치를 보여주는 작은 시간 후의 변환된 분포를 표시합니다. 흥미롭게도, 두 개의 원(Two Circles) 분포에 맞추기 위해 CNF는 평면 흐름을 회전시켜 입자들이 균일하게 원으로 퍼질 수 있도록 합니다. CNF 변환은 부드럽고 해석 가능하지만, NF 변환은 매우 직관적이지 않으며 그림 6(b)에 나타난 두 개의 달(Two Moons) 데이터셋을 맞추는 데 어려움을 겪는 것을 확인할 수 있습니다.

**5. 생성 잠재 함수 시계열 모델**  
의료 기록, 네트워크 트래픽, 또는 신경 스파이크 데이터와 같이 불규칙하게 샘플링된 데이터에 신경망을 적용하는 것은 어렵습니다. 일반적으로, 관찰된 데이터는 고정된 기간의 빈(bin)에 넣고 잠재 동역학(latent dynamics)도 같은 방식으로 이산화합니다. 이는 결측 데이터와 명확하지 않은 잠재 변수 문제를 초래합니다. 결측 데이터는 생성 시계열 모델(Álvarez and Lawrence, 2011; Futoma et al., 2017; Mei and Eisner, 2017; Soleimani et al., 2017a)이나 데이터 보간(Che et al., 2018)을 사용하여 해결할 수 있습니다. 또 다른 접근법으로는 시간 정보(time-stamp)를 RNN의 입력에 결합하는 방식도 있습니다(Choi et al., 2016; Lipton et al., 2016; Du et al., 2016; Li, 2017).

![](/assets/images/posts/295/img_28.png)

![](/assets/images/posts/295/img_29.png)

그림 7: 잠재 ODE 모델의 계산 그래프.

![](/assets/images/posts/295/img_30.png)

**훈련 및 예측**

![](/assets/images/posts/295/img_31.png)

![](/assets/images/posts/295/img_32.png)

그림 8: 푸아송 과정 가능도를 사용한 잠재 ODE 동역학 모델 피팅. 점들은 사건이 발생한 시간을 나타냅니다. 선은 학습된 푸아송 과정의 강도 λ(t)입니다.

**푸아송 과정 가능도**

![](/assets/images/posts/295/img_33.png)

![](/assets/images/posts/295/img_34.png)

![](/assets/images/posts/295/img_35.png)

![](/assets/images/posts/295/img_36.png)

그림 9:  
(9(a)): 순환 신경망(RNN)을 사용하여 불규칙한 시간 지점에서 나선형의 재구성 및 외삽.  
(9(b)): 잠재 신경 ODE에 의한 재구성 및 외삽. 파란색 곡선은 모델의 예측을 나타내고, 빨간색은 외삽을 나타냅니다.  
(9(c)): 추정된 4차원 잠재 ODE 경로를 처음 두 차원에 투영한 모습. 색상은 해당 경로의 방향을 나타냅니다. 모델은 두 방향을 구별하는 잠재 동역학을 학습했습니다.

우리는 λ(⋅)를 또 다른 신경망을 사용해 매개변수화할 수 있습니다. 편리하게도, 잠재 경로와 푸아송 과정 가능도를 모두 ODE 솔버의 단일 호출로 함께 평가할 수 있습니다. 그림 8은 이러한 모델이 장난감 데이터셋에서 학습한 사건 발생률을 보여줍니다.

관찰 시간에 대한 푸아송 과정 가능도는 데이터 가능도와 결합하여 모든 관찰과 그들이 발생한 시간을 함께 모델링할 수 있습니다.

**5.1 시계열 잠재 ODE 실험**

![](/assets/images/posts/295/img_37.png)

**양방향 나선 데이터셋**  
우리는 1000개의 2차원 나선 데이터셋을 생성했으며, 각 나선은 서로 다른 시작점에서 시작하고 100개의 동일 간격 시간 단계에서 샘플링되었습니다. 데이터셋은 두 가지 유형의 나선을 포함하며, 절반은 시계 방향, 나머지 절반은 반시계 방향입니다. 더 현실적인 문제를 만들기 위해, 우리는 관측에 가우시안 노이즈를 추가했습니다.

**불규칙한 시간 지점을 가진 시계열**  
불규칙한 타임스탬프를 생성하기 위해 각 경로에서 중복 없이 무작위로 지점을 샘플링했습니다 (n={30,50,100}). 우리는 학습에 사용된 시간 지점을 넘어서는 100개의 시간 지점에 대한 예측 루트 평균 제곱 오차(RMSE)를 보고합니다. 표 2는 잠재 ODE가 예측 RMSE에서 상당히 낮은 값을 보임을 보여줍니다.

그림 9는 30개의 하위 샘플링된 지점을 사용한 나선 재구성의 예시를 보여줍니다. 잠재 ODE에서의 재구성은 잠재 경로에 대한 사후 분포에서 샘플링하고 이를 데이터 공간으로 디코딩하여 얻었습니다. 다양한 시간 지점을 가진 예시는 부록 F에 나와 있습니다. 우리는 관측된 지점 수에 관계없이 재구성과 외삽이 실제 값과 일치하며, 노이즈에도 불구하고 일관되다는 것을 관찰했습니다.

표 2: 테스트 세트에서의 예측 RMSE

![](/assets/images/posts/295/img_38.png)

**잠재 공간 보간**

![](/assets/images/posts/295/img_39.png)

![](/assets/images/posts/295/img_40.png)

![](/assets/images/posts/295/img_41.png)

**6. 범위와 한계**

**미니배칭**

미니배치 사용은 표준 신경망에 비해 간단하지 않습니다. 각 배치 요소의 상태를 함께 연결하여 ODE 솔버를 통해 평가를 묶는 방법으로 여전히 배칭할 수 있습니다. 이로 인해 차원이 D×K인 결합된 ODE가 생성됩니다. 경우에 따라, 모든 배치 요소에 대한 오류를 함께 제어하려면 각 시스템을 개별적으로 해결할 때보다 결합된 시스템을 K배 더 자주 평가해야 할 수 있습니다. 하지만 실제로 미니배치를 사용할 때 평가 횟수가 크게 증가하지 않았습니다.

**유일성**  
연속적인 동역학이 언제 유일한 해를 가지는가? Picard의 존재 정리(Coddington과 Levinson, 1955)는 초기 값 문제의 해가 존재하고 유일하기 위해 미분 방정식이 z에 대해 균일한 립시츠 연속성을 가지며, t에 대해 연속적이어야 한다고 말합니다. 이 정리는 신경망이 유한한 가중치를 가지고 tanh나 relu와 같은 Lipshitz 비선형성을 사용할 경우 우리 모델에도 적용됩니다.

**허용 오차 설정**

![](/assets/images/posts/295/img_42.png)

**순방향 경로 재구성**  
동역학을 역방향으로 실행하여 상태 경로를 재구성하면 재구성된 경로가 원래 경로에서 벗어날 경우 추가적인 수치적 오류를 유발할 수 있습니다. 이 문제는 체크포인팅을 통해 해결할 수 있습니다: 순방향 패스에서 z의 중간 값을 저장하고, 해당 지점에서부터 다시 적분하여 정확한 순방향 경로를 재구성하는 것입니다. 우리는 이 문제를 실질적인 문제로 여기지 않았으며, 기본 허용치를 사용하여 연속 정규화 흐름의 여러 층을 역전파한 결과 초기 상태가 회복되는 것을 비공식적으로 확인했습니다.

**7.관련 연구**  
연속-시간 신경망 학습을 위한 접합 방법(adjont method)의 사용은 이전에도 제안된 바 있습니다(LeCun et al., 1988; Pearlmutter, 1995). 하지만 실제로는 시연되지 않았습니다. 잔차 네트워크(ResNet, He et al., 2016a)를 근사적인 ODE 솔버로 해석한 것은 ResNet에서 가역성(reversibility)과 근사 계산을 활용하는 연구를 촉발시켰습니다(Chang et al., 2017; Lu et al., 2017). 우리는 ODE 솔버를 직접 사용하여 이러한 특성을 더 일반화된 형태로 보여줍니다.

**적응형 계산**  
부차적인 신경망을 훈련시켜 순환 신경망(recurrent)이나 잔차 네트워크(residual)의 평가 횟수를 선택함으로써 계산 시간을 적응시킬 수 있습니다(Graves, 2016; Jernite et al., 2016; Figurnov et al., 2017; Chang et al., 2018). 그러나 이는 학습과 테스트 시간 모두에서 오버헤드를 유발하고, 추가적인 파라미터가 필요합니다. 반면, ODE 솔버는 잘 연구된, 계산적으로 저렴하고, 일반화 가능한 규칙을 제공하여 계산량을 적응시킵니다.

**가역성을 통한 상수 메모리 역전파**  
최근 연구들은 잔차 네트워크의 가역적인 버전을 개발했습니다(Gomez et al., 2017; Haber and Ruthotto, 2017; Chang et al., 2017). 이는 우리의 접근과 동일한 상수 메모리 이점을 제공합니다. 하지만 이러한 방법들은 은닉 유닛을 분할하는 제한된 아키텍처를 필요로 합니다. 우리의 접근은 이러한 제한이 없습니다.

**미분 방정식 학습**  
최근 많은 연구들이 데이터를 통해 미분 방정식을 학습하는 것을 제안했습니다. 피드포워드 또는 순환 신경망을 학습하여 미분 방정식을 근사하는 방식이 사용되며(Raissi and Karniadakis, 2018; Raissi et al., 2018a; Long et al., 2017), 유체 시뮬레이션(Wiewel et al., 2018)과 같은 응용에 활용됩니다. 또한, 가우시안 과정(Gaussian Processes, GPs)과 ODE 솔버를 연결하는 중요한 연구들도 있습니다(Schober et al., 2014). GPs는 미분 방정식에 맞춰 조정되었으며(Raissi et al., 2018b), 연속-시간 효과와 개입을 자연스럽게 모델링할 수 있습니다(Soleimani et al., 2017b; Schulam and Saria, 2017). Ryder et al. (2018)은 확률적 변이형 추론을 사용하여 주어진 확률적 미분 방정식의 해를 회복합니다.

**ODE 솔버를 통한 미분**  
dolfin 라이브러리(Farrell et al., 2013)는 일반적인 ODE 및 PDE 해에 대해 접합 계산을 구현하지만, 순방향 솔버의 개별 연산을 통해 역전파하는 방식만을 사용합니다. Stan 라이브러리(Carpenter et al., 2015)는 ODE 해를 통해 기울기를 추정하기 위해 순방향 감도 분석(forward sensitivity analysis)을 구현합니다. 하지만 순방향 감도 분석은 변수의 수에 대해 시간 복잡도가 제곱이지만, 접합 감도 분석은 선형 시간 복잡도를 가집니다(Carpenter et al., 2015; Zhang and Sandu, 2014). Melicher et al. (2017)은 맞춤형 잠재 동적 모델을 학습하기 위해 접합 방법을 사용했습니다.

반면, 우리는 일반적인 벡터-야코비안 곱을 제공함으로써 ODE 솔버가 다른 미분 가능한 모델 구성 요소와 함께 종단간 학습이 가능하게 합니다. 접합 방법을 해결하기 위해 벡터-야코비안 곱을 사용하는 것은 최적 제어에서 탐구된 바 있습니다(Andersson, 2013; Andersson et al., 2018). 우리는 블랙박스 ODE 솔버를 자동 미분에 통합하여 심층 학습과 생성 모델링에 활용할 수 있는 가능성을 강조합니다(Baydin et al., 2018).

**8. 결론**  
우리는 블랙박스 ODE 솔버를 모델 구성 요소로 사용하는 것을 조사하여, 시계열 모델링, 지도 학습, 밀도 추정을 위한 새로운 모델을 개발했습니다. 이러한 모델들은 적응형으로 평가되며, 계산 속도와 정확도 간의 절충을 명시적으로 제어할 수 있습니다. 마지막으로, 우리는 변수 변환 공식의 순간 버전을 유도하고, 대규모 층 크기에 확장 가능한 연속-시간 정규화 흐름을 개발했습니다.

**9. 감사의 말**  
증명에 도움을 주신 Wenyi Wang과 Geoff Roeder에게 감사드립니다. 또한 Daniel Duckworth, Ethan Fetaya, Hossein Soleimani, Eldad Haber, Ken Caluwaerts, Daniel Flam-Shepherd, 그리고 Harry Braviner의 피드백에도 감사드립니다. 유익한 논의를 제공해 주신 Chris Rackauckas, Dougal Maclaurin, Matthew James Johnson에게도 감사드립니다. 마지막으로, 파라미터 효율성에 대한 지원되지 않은 주장에 대해 지적해주신 Yuval Frommer에게도 감사드립니다.

[1806.07366v5.pdf

3.81MB](./file/1806.07366v5.pdf)

**부록 A: 순간적인 변수 변환 정리의 증명**

![](/assets/images/posts/295/img_43.png)

![](/assets/images/posts/295/img_44.png)

![](/assets/images/posts/295/img_45.png)

행렬식의 도함수는 야코비(Jacobi)의 공식을 사용하여 표현할 수 있습니다.

![](/assets/images/posts/295/img_46.png)

![](/assets/images/posts/295/img_47.png)

![](/assets/images/posts/295/img_48.png)

**A.1 특수 사례**

평면 CNF

![](/assets/images/posts/295/img_49.png)

해밀토니안 CNF

![](/assets/images/posts/295/img_50.png)

**A.2 포커-플랑크 방정식과 리우빌 방정식과의 연결**  
포커-플랑크 방정식(Fokker-Planck equation)은 시간에 따라 변화하는 확률 밀도 함수를 설명하는 잘 알려진 편미분 방정식(PDE)입니다. 우리는 순간적인 변수 변환을 확산이 없는 포커-플랑크의 특수 사례인 리우빌 방정식과 연관지어 설명합니다.

![](/assets/images/posts/295/img_51.png)

고정된 지점에서 p(⋅,t)를 평가하는 대신 입자 z(t)의 경로를 따라가면 다음과 같은 식을 얻습니다:

![](/assets/images/posts/295/img_52.png)

변수 변환의 순간적인 변화를 얻기 위해 로그를 취합니다:

![](/assets/images/posts/295/img_53.png)

이 식은 여전히 PDE이지만, z(t)와 결합하여 크기가 D+1인 ODE를 형성할 수 있습니다:

![](/assets/images/posts/295/img_54.png)

포커-플랑크 방정식 및 리우빌 방정식과 비교할 때, 변수 변환의 순간적인 변화는 더 실용적인 영향을 미치며, 수치적으로 훨씬 쉽게 해결할 수 있습니다. 이는 z(t)의 경로를 따르기 위해 D의 추가 상태만 필요합니다. 반면, 리우빌 방정식의 유한 차분 근사를 기반으로 한 접근법은 D에 대해 지수적인 크기의 그리드를 필요로 할 것입니다.

**부록 B: 접합 방법의 현대적 증명**  
우리는 접합 방법(Pontryagin et al., 1962)의 짧고 따라가기 쉬운 대체 증명을 제시합니다.

**B.1 연속 역전파**

![](/assets/images/posts/295/img_55.png)

![](/assets/images/posts/295/img_56.png)

![](/assets/images/posts/295/img_57.png)

우리는 접합 방법과 역전파(식 38)의 유사성을 지적했습니다. 역전파와 마찬가지로, 접합 상태에 대한 ODE는 시간 역방향으로 해결되어야 합니다. 우리는 마지막 시간 지점에서의 제약 조건을 명시하는데, 이는 단순히 마지막 시간 지점에 대한 손실의 기울기입니다. 이를 통해 초기값을 포함한 임의의 시간에서 은닉 상태에 대한 기울기를 얻을 수 있습니다.

![](/assets/images/posts/295/img_58.png)

![](/assets/images/posts/295/img_59.png)

B.2 θ와 t에 대한 기울기

![](/assets/images/posts/295/img_60.png)

이 상태에 해당하는 미분 방정식과 접합 상태는 다음과 같이 정의됩니다:

![](/assets/images/posts/295/img_61.png)

![](/assets/images/posts/295/img_62.png)

**부록 C: 전체 접합 감도 알고리즘**  
알고리즘 1의 이 보다 상세한 버전은 적분의 시작 및 종료 시간에 대한 기울기를 포함합니다.

![](/assets/images/posts/295/img_63.png)

**부록 D: Autograd 구현**

```
import scipy.integrate
import autograd.numpy as np
from autograd.extend import primitive, defvjp_argnums
from autograd import make_vjp
from autograd.misc import flatten
from autograd.builtins import tuple
odeint = primitive(scipy.integrate.odeint)
def grad_odeint_all(yt, func, y0, t, func_args, **kwargs):
    # Extended from "Scalable Inference of Ordinary Differential
    # Equation Models of Biochemical Processes", Sec. 2.4.2
    # Fabian Froehlich, Carolin Loos, Jan Hasenauer, 2017
    # https://arxiv.org/pdf/1711.08079.pdf
    T, D = np.shape(yt)
    flat_args, unflatten = flatten(func_args)
    def flat_func(y, t, flat_args):
        return func(y, t, *unflatten(flat_args))
    def unpack(x):
        #      y,      vjp_y,      vjp_t,    vjp_args
        return x[0:D], x[D:2 * D], x[2 * D], x[2 * D + 1:]
    def augmented_dynamics(augmented_state, t, flat_args):
        # Orginal system augmented with vjp_y, vjp_t and vjp_args.
        y, vjp_y, _, _ = unpack(augmented_state)
        vjp_all, dy_dt = make_vjp(flat_func, argnum=(0, 1, 2))(y, t, flat_args)
        vjp_y, vjp_t, vjp_args = vjp_all(-vjp_y)
        return np.hstack((dy_dt, vjp_y, vjp_t, vjp_args))
    def vjp_all(g,**kwargs):
        vjp_y = g[-1, :]
        vjp_t0 = 0
        time_vjp_list = []
        vjp_args = np.zeros(np.size(flat_args))
        for i in range(T - 1, 0, -1):
            # Compute effect of moving current time.
            vjp_cur_t = np.dot(func(yt[i, :], t[i], *func_args), g[i, :])
            time_vjp_list.append(vjp_cur_t)
            vjp_t0 = vjp_t0 - vjp_cur_t
            # Run augmented system backwards to the previous observation.
            aug_y0 = np.hstack((yt[i, :], vjp_y, vjp_t0, vjp_args))
            aug_ans = odeint(augmented_dynamics, aug_y0,
                             np.array([t[i], t[i - 1]]), tuple((flat_args,)), **kwargs)
            _, vjp_y, vjp_t0, vjp_args = unpack(aug_ans[1])
            # Add gradient from current output.
            vjp_y = vjp_y + g[i - 1, :]
        time_vjp_list.append(vjp_t0)
        vjp_times = np.hstack(time_vjp_list)[::-1]
        return None, vjp_y, vjp_times, unflatten(vjp_args)
    return vjp_all
def grad_argnums_wrapper(all_vjp_builder):
    # A generic autograd helper function.  Takes a function that
    # builds vjps for all arguments, and wraps it to return only required vjps.
    def build_selected_vjps(argnums, ans, combined_args, kwargs):
        vjp_func = all_vjp_builder(ans, *combined_args, **kwargs)
        def chosen_vjps(g):
            # Return whichever vjps were asked for.
            all_vjps = vjp_func(g)
            return [all_vjps[argnum] for argnum in argnums]
        return chosen_vjps
    return build_selected_vjps
defvjp_argnums(odeint, grad_argnums_wrapper(grad_odeint_all))
```

**부록 E: 잠재 ODE 모델 학습 알고리즘**

![](/assets/images/posts/295/img_64.png)

**부록 F: 추가 그림**

![](/assets/images/posts/295/img_65.png)

그림 11: 잠재 ODE를 사용하여 재구성된 나선형, 다양한 개수의 노이즈가 있는 관측값을 사용.
