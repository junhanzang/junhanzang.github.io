---
title: "유니티 ML-Agents를 이용한 AutoParkSimulation"
date: 2023-03-22 02:04:45
categories:
  - 프로젝트
---

Source: <https://github.com/hae-sung-oh/AutoParkSimulation>

일단 기본적으로 PPO를 사용하며 전반적으로 잘 갖춰져있다.

하지만 몇가지 부분에서 아쉬워서 다음 yaml코드로 변경하였다.

```
behaviors:
    default:
        trainer_type: ppo
        hyperparameters:
            batch_size: 256
            buffer_size: 10240
            learning_rate_schedule: linear
            learning_rate: 5.0e-4
        network_settings:
            hidden_units: 256
            normalize: true
            num_layers: 3
            vis_encode_type: simple
            memory:
                memory_size: 256
                sequence_length: 256
        max_steps: 10.0e5
        time_horizon: 64    
        summary_freq: 5000
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
    Autopark:
        trainer_type: ppo
        hyperparameters:
            batch_size: 128
            buffer_size: 5120
        network_settings:
            hidden_units: 256
            num_layers: 3
        max_steps: 20.0e6
        time_horizon: 128
        threaded: True
```

buffer\_size를 더 크게 조정하면 경험 리플레이 샘플의 다양성이 증가할 수 있습니다.   
시각 인코더 유형을 변경하거나 숨겨진 레이어 수를 늘리는 등의 실험을 통해 신경망 구조를 최적화할 수 있다.   
normalize 옵션이 false로 설정되어 있습니다. 이 경우, 입력 데이터를 정규화하지 않습니다. 데이터의 분포가 매우 다양한 경우에는 정규화가 필요합니다. 데이터의 범위가 크고 분포가 다른 경우 학습이 불안정해질 수 있습니다. 따라서 normalize 옵션을 true로 설정하고, 입력 데이터를 정규화하는 것이 좋습니다.   
Autopark 구성에서 threaded 옵션이 True로 설정되어 있습니다. 이 경우, 다중 스레드를 사용하여 학습 속도를 높입니다. 하지만 다중 스레드를 사용할 때는, 여러 스레드가 동시에 메모리에 접근할 수 있어, 메모리 충돌이 발생할 수 있습니다. 이 경우, batch\_size나 buffer\_size를 줄이는 것이 좋습니다.   
hyperparameters의 learning\_rate 값을 변경하여 학습 속도를 조정할 수 있습니다. 너무 작은 값으로 설정하면 학습이 느리게 진행되고, 너무 큰 값으로 설정하면 학습이 불안정해질 수 있습니다. 적절한 학습률을 찾기 위해 여러 가지 값을 실험해보는 것이 좋습니다.  
network\_settings의 hidden\_units나 num\_layers 값을 변경하여 신경망 구조를 최적화할 수 있습니다. 더 복잡한 모델을 사용하면 더 나은 성능을 얻을 수 있지만, 학습 시간이 더 오래 걸릴 수 있습니다. 따라서 적절한 모델 복잡도를 찾아야 합니다.
