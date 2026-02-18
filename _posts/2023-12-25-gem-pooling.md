---
title: "GEM Pooling"
date: 2023-12-25 23:36:29
categories:
  - 인공지능
---

```
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
```

여기서 설명된 GeM (Generalized Mean Pooling) 클래스는 딥러닝 모델에서 사용되는 pooling 방법 중 하나로, 이미지의 특징을 요약하는 데 사용됩니다. 특히, 이 클래스는 일반적인 평균 풀링 또는 최대 풀링과 다르게, generalized mean을 계산하여 풀링을 수행합니다. 이 방식의 효과를 살펴보겠습니다.

1. **유연성**: GeM은 일반적인 평균 풀링(avg\_pool2d)에 제곱(power) 연산을 적용합니다. 여기서 p는 제곱하는 정도를 결정하는 매개변수입니다. p=1일 때, GeM은 일반 평균 풀링과 같고, p가 높아질수록 최대 풀링에 가까워집니다. 이 매개변수를 통해 평균 풀링과 최대 풀링 사이에서 원하는 풀링 방식을 조절할 수 있습니다.
2. **로버스트성(안정성)**: eps는 계산 안정성을 위해 추가된 매우 작은 값입니다. 이는 x를 제곱하기 전에 x의 모든 값에 eps를 더하여, 0이나 매우 작은 값에 대한 제곱 연산에서 발생할 수 있는 수치적 불안정성을 방지합니다.
3. **특징의 강조**: p 매개변수를 사용함으로써, 풀링 과정에서 특징들이 다르게 강조됩니다. p 값이 크면 클수록 이미지의 더 두드러진 특징(강한 신호를 가진 부분)이 강조되고, 작은 특징들은 상대적으로 덜 중요하게 됩니다.
4. **적응성**: 이 클래스에서 p는 학습 가능한 매개변수(nn.Parameter)로 설정됩니다. 이는 학습 과정에서 데이터에 따라 최적의 p 값을 자동으로 찾을 수 있음을 의미합니다. 따라서 모델은 다양한 데이터셋과 태스크에 더 잘 적응할 수 있습니다.

결론적으로, GeM은 풀링 과정에서 이미지의 특징을 더 유연하고 로버스트하게 요약할 수 있게 해주며, 이를 통해 이미지 분류, 검색, 인식 등의 작업에서 성능을 향상시킬 수 있습니다.
