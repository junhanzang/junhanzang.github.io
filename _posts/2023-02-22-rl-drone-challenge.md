---
title: "RL Drone Challenge"
date: 2023-02-22 00:10:40
categories:
  - 프로젝트
---

```
behaviors:
  My Behavior:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 0.0005
      beta: 0.01
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 5
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 128
      num_layers: 2
      vis_encode_type: nature_cnn
```

1. Batch size: The batch size has been decreased to 128 to improve stability during training.
2. Learning rate: The learning rate has been increased to 0.0005 to help the agent learn more quickly.
3. Number of epochs: The number of training epochs per update has been increased to 5 to give the agent more opportunities to learn from its experiences.
4. Network architecture: The network now uses the nature\_cnn visual encoder, which is a good choice for image-based observations. The number of hidden units has been reduced to 128 to improve training stability.
5. Normalization: Normalization has been enabled, which can help the agent learn more effectively by reducing the impact of different scales and offsets in the observations.
