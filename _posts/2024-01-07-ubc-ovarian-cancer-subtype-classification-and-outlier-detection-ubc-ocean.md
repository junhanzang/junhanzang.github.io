---
title: "UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)"
date: 2024-01-07 18:52:30
categories:
  - 개인용
---

NystromAttention은 이 문제를 해결하기 위해 'Nystrom 방법'을 적용합니다. Nystrom 방법은 행렬의 근사치를 계산하는 데 사용되는 수학적 기법으로, 원래 행렬의 작은 부분집합을 사용하여 전체 행렬을 근사하는 방식입니다. 이를 통해 NystromAttention은 전체 입력 데이터를 사용하지 않고도 어텐션 매트릭스의 근사치를 효율적으로 계산할 수 있습니다.

이 접근 방식은 계산 효율성을 크게 향상시키면서도 모델의 성능을 유지하거나 심지어 개선할 수 있는 잠재력을 가지고 있습니다. 따라서, NystromAttention은 자연어 처리, 이미지 분석 등 다양한 분야에서 트랜스포머 모델을 더 효율적으로 활용할 수 있는 방법으로 주목받고 있습니다.

NystromAttention 사용

Vision Transformer (ViT), ABMIL, DSMIL, 그리고 TransMIL은 모두 인공지능 분야, 특히 컴퓨터 비전과 관련된 중요한 개념들입니다. 각각에 대해 설명드리겠습니다.

1. **Vision Transformer (ViT)**:
   - Vision Transformer는 이미지 처리를 위해 트랜스포머 아키텍처를 적용한 모델입니다. 기존 트랜스포머는 주로 자연어 처리(NLP)에 사용되었지만, ViT는 이를 이미지 인식 분야로 확장했습니다.
   - ViT는 이미지를 여러 개의 작은 패치로 분할하고, 각 패치를 트랜스포머의 입력으로 사용합니다. 이를 통해 ViT는 이미지의 다양한 부분 간의 관계를 학습하며, 이는 전통적인 CNN(Convolutional Neural Networks) 기반 모델과는 다른 접근 방식입니다.
   - ViT는 이미지 분류, 객체 탐지, 세그멘테이션 등 다양한 컴퓨터 비전 작업에서 뛰어난 성능을 보여주었습니다.
2. **ABMIL (Attention-Based Multiple Instance Learning)**:
   - ABMIL은 Multiple Instance Learning (MIL)의 한 형태로, 주로 의료 영상 분석에 사용됩니다. MIL은 레이블이 개별 인스턴스가 아닌 데이터의 집합(백)에 할당되는 경우에 사용됩니다.
   - ABMIL은 어텐션 메커니즘을 사용하여 백 내의 중요한 인스턴스에 더 많은 가중치를 부여합니다. 이를 통해 모델은 백 전체를 분류하는 데 도움이 되는 핵심적인 특징을 학습할 수 있습니다.
   - 이 방법은 특히 암 진단과 같은 의료 이미지 분석에서 유용하게 사용됩니다.
3. **DSMIL (Deeply Supervised Multiple Instance Learning)**:
   - DSMIL은 깊은 감독 학습을 MIL에 적용한 방법입니다. 여기서 '깊은 감독'은 모델의 여러 레이어에 걸쳐서 정확한 레이블 정보를 제공하는 것을 의미합니다.
   - 이 방법은 모델이 보다 정확하고 세분화된 특징을 학습하는 데 도움을 줍니다. DSMIL은 특히 복잡한 구조를 가진 데이터셋에서 더 나은 성능을 보여줍니다.
4. **TransMIL (Transformer Multiple Instance Learning)**:
   - TransMIL은 트랜스포머 아키텍처를 MIL과 결합한 방법입니다. 이는 백 내의 인스턴스 간의 상호작용을 모델링하는 데 특히 효과적입니다.
   - TransMIL은 어텐션 메커니즘을 통해 백 내에서 중요한 인스턴스를 식별하고, 이를 바탕으로 전체 백의 레이블을 예측합니다. 이 방법은 의료 영상 분석 및 다른 분야에서 복잡한 데이터 구조를 다루는 데 유용합니다.

등등이 상위권에서 사용한 모델들이 존재하는데

데이터가 1000개밖에 안되서 이를 외부에서 추가적으로 가져오는게 중요한 것으로 보임

Birdclef와 동일한 문제로 인식됨

```
# import timm

# # 모든 모델 목록을 가져오기
# all_models = timm.list_models()
# print(all_models)

# # 사전 학습된 모델 목록만 가져오기
# pretrained_models = timm.list_models(pretrained=True)
# print(pretrained_models)
```
