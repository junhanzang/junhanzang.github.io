---
title: "Chapter 8 Small Nets and EfficientNet"
date: 2023-05-20 15:04:13
tags:
  - EfficientNet
  - mobilenet
  - MobileNet-V1
  - MobileNet-V2
  - ShuffleNet
  - Small Nets
  - xception
---

우리는 지금까지 모델에 layer들이 추가되는 방식에 대해서 설명했다. 얼마나 더 큰 모델을 만들고 이것들이 잘작동하는지 말이다. 하지만 큰 모델들은 컴퓨터나 클라우드에서 사용가능하다. 이는 어디에서나 해당 모델을 사용할 수 없다는 말이다. 가령 인터넷이 끊기는 지역이라던가 컴퓨터가 없는 밖에서는 말이다. 그렇다면 핸드폰에서 인공지능을 사용하면 어떨까? 왠만한 곳에서 학습도 가능하고 실제 사용도 가능하지 않을까? 이렇게 해서 나온 것이 MobileNet이다.

MobileNet-V1

MobileNet-V1을 이해하려면 이전에 설명했던 것들을 기억해야한다.

먼저, Depthwise Separable Convolution이다.

![](https://blog.kakaocdn.net/dna/y7ltl/btsgEfnn78R/AAAAAAAAAAAAAAAAAAAAACI2bIe2quEC5GOIDsqBHObJnyRYMjFwniBdH0uikPra/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=MYjkR7%2B72idzEZIo%2Fw8UlJFviaA%3D)

우리는 이를 통해 총 연산을 줄이는 것을 알고 있다.

이를 활용하면 아래 그림의 왼쪽이 오른쪽처럼 구조가 바뀐다는 것을 알 수 있다.

![](https://blog.kakaocdn.net/dna/m93C9/btsgCi6du25/AAAAAAAAAAAAAAAAAAAAAHxwacyjeetbgl4ENsRAlyDn6I3LoAe3y-3OxavMk6Ad/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=lUGvd5M76WnneB%2BmEzgIufF8a7U%3D)

이를 통해 줄인 Model Structure를 보면 다음과 같다.

![](https://blog.kakaocdn.net/dna/bsRwWd/btsgG6iDNWC/AAAAAAAAAAAAAAAAAAAAADI1XVid97gYMgqPbKe7m1nczgNHxVXSXZncEokeKZtp/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ZT9t1mwu3hg03tqNAO1Z1%2BKKgyw%3D)

그래서 layer의 비율을 보면 1x1 Conv가 가장 많이 차지하는 것을 알 수 있다.

![](https://blog.kakaocdn.net/dna/HVzbc/btsgFvQvk6K/AAAAAAAAAAAAAAAAAAAAAJ7lQzPgU0kSFMr9EHywvTzb-9pvB_bFZgnpeJ5egO86/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=spE%2B2o8kBxpwvuOPzBQBvKpok9A%3D)

- 너비 승수 Thinner 모델 (Width Multiplier Thinner Models)  
주어진 레이어 및 폭 승수 α에 대해 입력 채널 수 M은 αM이 되고 출력 채널 수 N은 αN이 된다  
일반적인 설정이 1, 0.75, 0.6 및 0.25인 α  
- 분해능 승수 감소 표현 (Resolution Multiplier Reduced Representation)  
신경망의 계산 비용을 줄이기 위한 두 번째 하이퍼 매개변수는 해상도 승수 ρ  
0< ρ≤ 1, 일반적으로 네트워크의 입력 해상도가 224, 192, 160 또는 128( ρ = 1, 0.857, 0.714, 0.571)이 되도록 암시적으로 설정된다  
- 계산 비용:  
𝐷k×𝐷k×𝜶𝑀×𝝆𝐷F×𝝆𝐷F + 𝑀×𝑁×𝝆𝐷F×𝝆𝐷F

추가적으로 구조중에 Fully Connected Layer도 기억해야한다.

![](https://blog.kakaocdn.net/dna/rt7ay/btsgEK1st9A/AAAAAAAAAAAAAAAAAAAAAAIZCMYptSpHoUAN-IRBvIs_HjXhHNJ52jqLaFgNPgDH/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=CbBabxbadsJMljZyiA9vKSxbyhw%3D)

특징 맵의 모든 픽셀은 Fully Connected Layer에 연결된다.

추가적으로 Global Average Pooling을 마지막 부분에 사용한다. Global Average Pooling이란 네트워크의 출력을 하나의 고정된 크기의 벡터로 변환하는 역할을 수행한다. 이는 네트워크의 출력 특성 맵의 공간적인 정보를 간결하게 압축하고, 분류 작업과 같은 최종 출력을 위한 특징 벡터를 생성하는 데 사용된다.

![](https://blog.kakaocdn.net/dna/c14wlB/btsgFujNA83/AAAAAAAAAAAAAAAAAAAAADRWpjboH_S5ibVkQR3AmmK5ZKd1N3sAmOzU5euzFau5/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=qbz4AtesVKWpOCE4cqiHlRhWTYg%3D)

Global Average Pooling

그리고 종합하여 다음의 결과를 보여준다.

![](https://blog.kakaocdn.net/dna/ITGov/btsgC1XfpdR/AAAAAAAAAAAAAAAAAAAAAOVpC0ctsrmo3OASBO4DtoDfpjbaCU-2OKHGgPN55tSW/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=maH%2Bi3DfhN%2FfiWjftj%2B5LLAoRzM%3D)
![](https://blog.kakaocdn.net/dna/nEG8v/btsgEcw8Wy8/AAAAAAAAAAAAAAAAAAAAAEd0cPwQh4BKNY5-sNkZ-rYJXhEak0NAlLqFEAKpxUag/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=cylpPQQPeJyToCowluIH8%2FL8tAk%3D)

파라미터는 줄었지만 정확도는 비슷하다.

MobileNet-V2

MobileNet-V2는 Depthwise Separable Convolution block을 Bottleneck Residual block으로 업그레이드했다.

![](https://blog.kakaocdn.net/dna/p4bta/btsgG5c0zYF/AAAAAAAAAAAAAAAAAAAAACxDR1qWmPC0APHK7BOQth4MjkGuS1CZJ-ufLWYGwXY8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=PPX%2F1zZOvvPoFij5YaDj2L5cZ70%3D)

이 모듈은 먼저 high dim으로 확장되고 가벼운 깊이 방향 컨볼루션으로 필터링되는 low dim 압축 표현을 입력으로 사용한다. 즉, Bottleneck Residual 블록은 ResNet에서 사용되는 Bottleneck 구조를 적용한 것으로 Bottleneck Residual 블록은 입력 데이터를 더 낮은 차원으로 압축한 후, 중간 차원에서 컨볼루션 연산을 수행하고, 다시 원래 차원으로 확장하는 방식이다. 그림으로 표현되면 다음과 같다.

![](https://blog.kakaocdn.net/dna/AgOnq/btsgC5FkQqQ/AAAAAAAAAAAAAAAAAAAAANIhIhYqz0CJqPwRiA1b57Mi3jNoR9NW1IIX-v_T7ngi/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=cirHaVpYovfy%2BCFIC4gfp2%2FFosw%3D)

이를 2개의 Bottleneck Residual layer가 연결된 것으로 표현되면 다음과 같다.

![](https://blog.kakaocdn.net/dna/cacl01/btsgEcYcJkM/AAAAAAAAAAAAAAAAAAAAAHheESVt1VO8BB7x5ksKXveXaooNeJdkx6VlzWrX94Rg/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=PYi%2Bva3TPbArTXi3Y%2F4clilnm0E%3D)

즉, 2개를 연결하면 bottleneck이 있는 pointwise convolution과 거의 동일하게 보인다. 왜 이렇게 사용했을까?

- Manifold의 Dimension은 Input의 Dimension보다 훨씬 낮은 것으로 판단되었다  
- 중요한 정보를 효과적으로 추출하기 위해 사용되었다

어떻게 보면 encoder의 역활을 수행한다고 봐도 될 것 같다.

그렇다면 이렇게 사용하면 안될까?

![](https://blog.kakaocdn.net/dna/DNJAg/btsgDg1q8YQ/AAAAAAAAAAAAAAAAAAAAAF2gzQWTbDHVEjoFfsdZiFnDANK2qyAUfyZnd4D7zOOq/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=rhs5iQTr2j0X5zeqtAJsaIPuNTo%3D)

이는 목적에 부합하지 않기 때문에 안된다.

![](https://blog.kakaocdn.net/dna/coCblY/btsgCBqZ4zA/AAAAAAAAAAAAAAAAAAAAAOy5XswgXOZTYtu_PX-t9dZ1mR9WT3P-LukfeHhDezmv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=2%2F7xFdm9kFSuqJo4ryAzy3NpJPM%3D)

Small Intermediate Tensors가 다음의 목적에 더 부합하기 때문이다.   
- 모바일 장치의 작지만 매우 빠른 캐시 메모리에 맞출 수 있다.   
- 추론 중에 필요한 메모리 공간을 크게 줄일 수 있다.   
- 많은 임베디드 하드웨어 설계에서 메인 메모리 액세스의 필요성을 줄인다.

그렇다면 GoogleNet과 달리 왜 Mobilenet-V2는 linear하게 쓸까?

선형 변환(linear transformation)을 통해 차원을 조정하는 역할을 진행하며, 이를 통해 차원 축소를 진행하여 중요한 정보를 효율적으로 캡처해야 하기 때문이다. 그러나 장점만 있는 것은 아니다. ReLU는 몇 가지 중요한 기능을 잃을 수 있다. 평균적으로 특징 맵의 절반은 정보가 없는 ZERO다.

![](https://blog.kakaocdn.net/dna/pALEq/btsgEBKwuqJ/AAAAAAAAAAAAAAAAAAAAABv-L077YfP-TiDTf3DQFv6nYTSydgvMHYa_H5rUnGBy/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=stkbDiGTEzE7I%2B4cc3jiQeg%2BYAI%3D)

위의 그림처럼 더 많은 정보를 만들 수 있어야하지만 RELU의 중요기능을 잃어 정보가 부족할 수 있다.

다음은 Mobilenet-V2의 구조와 결과를 보고 Mobilenet-V2를 마친다.

![](https://blog.kakaocdn.net/dna/50rlE/btsgG4yq34W/AAAAAAAAAAAAAAAAAAAAAN9xCxyciyc51y4oIFKKVGALSSAhW20rY1QlIaI4_TAO/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=NgI4xLgvPgZHfV4dEgr%2Bc5EGyBk%3D)

구조

![](https://blog.kakaocdn.net/dna/G1nDy/btsgEgT7xnJ/AAAAAAAAAAAAAAAAAAAAAHcBwxJtnnIYoFA7rySAk2QDgco91RuZ8hSj8I8UwFT1/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=tQ6FeJHTF%2FZb7xH7%2FIvXSNYL8uM%3D)

Impact of non-linearities and residual link

![](https://blog.kakaocdn.net/dna/biYNlO/btsgDJWnMUT/AAAAAAAAAAAAAAAAAAAAAMWLEtSUsCHEWD8bm5RMDob6kvM12Z5tj-Ghi0k5DOU4/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=B5u7psZk1DUlVcoHGuWANTR0ZY0%3D)

ImageNet Classification 결과

![](https://blog.kakaocdn.net/dna/wZjS8/btsgGi4uK4a/AAAAAAAAAAAAAAAAAAAAAF-jjEo37LX9SEKvu2zvg4hGDmmXmxRYpzS764UbflGI/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ISsIZ%2BmjT8ztdd5ZjL%2By9fYbfHk%3D)
![](https://blog.kakaocdn.net/dna/Uen7l/btsgFeutsoL/AAAAAAAAAAAAAAAAAAAAAKcBvMK5Fm9buBkUTUQweAObhO_oY2tlQCJXYKNKe2Lt/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=CgQvB8yC2EywFpU0HSnJ88nXLxg%3D)

몇가지 추가적으로 보고 Small Nets를 마치려고 한다.

ShuffleNet

ShuffleNet은 Group Convolution을 사용한다.Group Convolution은 입력 채널을 그룹으로 나누어 각 그룹에 대해 독립적으로 컨볼루션 연산을 수행하는 방법이다. 기존의 컨볼루션은 입력 채널 전체에 대해 연산을 수행하는 반면, Group Convolution은 채널을 그룹으로 나누어 그룹 간에 독립적인 연산을 수행한다.  
Group Convolution의 주요 목적은 모델의 연산 비용을 줄이는 것이다. 기존의 컨볼루션 연산은 입력 채널의 크기에 비례하여 연산 비용이 증가하는 문제가 있다. 그러나 Group Convolution은 채널을 그룹으로 나누어 각 그룹에 대해 독립적으로 연산을 수행하므로, 입력 채널의 크기에 상관없이 일정한 연산 비용을 유지할 수 있다.

![](https://blog.kakaocdn.net/dna/w5EF3/btsgCi6fzmB/AAAAAAAAAAAAAAAAAAAAAE7Gy6VCtkb2DkedXNg9J9gLGujwaa128dezaQig1ODx/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=GoBrkwUsb3RfYfR%2BbbjaGDnHDPg%3D)

추가적으로 Channel Shuffling도 사용한다. ShuffleNet은 컨볼루션 연산 후에 채널을 섞는 Channel Shuffle 작업을 수행한다. 이를 통해 다양한 그룹 간의 정보 교환을 촉진하고, 효과적인 특성 학습을 도모한다. Channel Shuffle은 입력 채널을 섞어 다양한 정보를 조합하고, 모델의 표현력을 향상시킨다.

![](https://blog.kakaocdn.net/dna/bnrNBD/btsgCkiJJVL/AAAAAAAAAAAAAAAAAAAAAFB5S-5rEWFRmTJ8ueci2dq3a6kgY0PBtWQAb9DFIjR2/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=hH4%2BiN74HMCj0paQe5qKb10F6PI%3D)

이를 적용한 블럭을 보면 다음과 같다.

![](https://blog.kakaocdn.net/dna/b3qfPf/btsgEeINHz6/AAAAAAAAAAAAAAAAAAAAAI8C8AdtILcSR2gy0898YkTdyM2MzQxe6MzD5WNYo4eM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=BuPJusnHTmsfZlpXgEfwyNPjvXQ%3D)

SqueezeNet

SqueezeNet의 핵심 아이디어는 "squeeze"와 "expand" 단계이다. 이 단계들은 입력 특성 맵의 차원을 조절하고 필터의 파라미터 수를 줄이는 역할을 수행한다.   
"Squeeze" 단계에서는 입력 채널을 줄이기 위해 1x1 컨볼루션 연산을 수행한다.  
"Expand" 단계에서는 압축된 특성 맵을 다시 원래 차원으로 확장한다. 이를 위해 1x1 컨볼루션 연산을 수행하고, 이어서 3x3 컨볼루션 연산을 수행한다. 이를 통해 입력 채널의 차원을 확장하고, 더 풍부한 특성 표현을 얻을 수 있다.  
SqueezeNet은 또한 "fire module"이라고 불리는 구조를 사용한다. 이 모듈은 "squeeze" 단계와 "expand" 단계로 구성되어 있으며, 작은 모델 크기와 높은 효율성을 제공한다. SqueezeNet은 많은 파라미터를 공유하고, 효율적인 구조를 통해 고성능을 달성하는데 초점을 두고 있다.

![](https://blog.kakaocdn.net/dna/CUnv3/btsgE6XPbc1/AAAAAAAAAAAAAAAAAAAAANThebQoUItsiT_ljTAaWcPq_YjsF8Nzw7Ew1K9d0m1H/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=AHPkx%2FCCOsdb2kSi44RD4RUsnAA%3D)

Xception

Xception은 "Extreme Inception"의 줄임말로, Inception 모듈을 발전시킨 네트워크 구조다. Xception은 컨볼루션 연산에서 극단적인 방식을 사용하는데, 기존의 Inception 모듈의 방식과는 다르다. Inception 모듈은 다양한 커널 크기를 동시에 적용하여 특성을 추출하는데 비해, Xception은 깊이 방향의 컨볼루션을 통해 특성을 추출할 수 있다.  
기존의 컨볼루션은 입력 특성 맵의 공간 방향과 채널 방향을 동시에 학습하는 반면, Xception은 먼저 공간 방향의 컨볼루션을 수행한 후, 채널 방향의 컨볼루션을 수행합니다. 이를 통해 네트워크는 공간적인 정보와 채널 간의 관계를 독립적으로 학습할 수 있다. 이는 더 효율적인 파라미터 사용과 더 나은 표현력을 제공한다.  
Xception은 파라미터 공유를 최소화하는 장점도 가지고 있다. 각각의 깊이 방향 컨볼루션은 독립된 파라미터를 가지기 때문에 효율적인 파라미터 사용이 가능하며, 이를 통해 모델의 크기와 계산 비용을 줄일 수 있다.

![](https://blog.kakaocdn.net/dna/xeWkn/btsgGmeJj8x/AAAAAAAAAAAAAAAAAAAAAMhfYjbd6EMljr9RvnUWcipJcDF6nzUn1GtkVP_oW65M/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Y8%2FATVnvaKKLFx2Vj3owZDwCX5U%3D)

EfficientNet으로 넘어가자.

EfficientNet이 나올 2019년의 CNN의 trend는 Base Blocks을 반복사용하는 것이었다.

![](https://blog.kakaocdn.net/dna/1ZfvE/btsgDLzStyS/AAAAAAAAAAAAAAAAAAAAAFgop5hrv_32Jzotf-c7a3Gumr1fAv72NLJ_39FJByai/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=5iIHyXIP8RvlQTBoRpYGk%2FhkmHQ%3D)

이를 통해 모델의 Scale이 커지고 이는 성능향상으로 이어졌다. 이를 통해, 더 좋은 image들이 생성되는 시기었다. Scale을 크게하는 방식은 다음과 같다.

![](https://blog.kakaocdn.net/dna/cv7j8h/btsgJ1BqkAX/AAAAAAAAAAAAAAAAAAAAAKQm2tYcLRJZ8QApyYJ2WmBPVFTHt-ueDe4Atw4HOuet/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=x%2BYA75VKwvSbvEH4O%2B4COyjBjRU%3D)

하지만 이는 곧 Saturation(포화) 현상을 보였다. ResNet 1000은 훨씬 더 많은 레이어를 가지고 있지만 ResNet 101과 비슷한 정확도를 가지고 있다. 그리고 네트워크가 넓더라도 네트워크가 얕으면 좋은 기능을 캡처하기 어려움을 겪었다. 아래는 해당 그림이다. d는 deeper, r은 resolution이다.

![](https://blog.kakaocdn.net/dna/bbcXY5/btsgEehOhFy/AAAAAAAAAAAAAAAAAAAAANJBnRZznmFHunsqrsSe4TY4HEGlH6cIjxBQTjbMl42p/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=%2FtUbiCdm1yPHwp5cGGghOTX7Vw0%3D)

resolution

![](https://blog.kakaocdn.net/dna/syFrS/btsgEKtDNia/AAAAAAAAAAAAAAAAAAAAAGvFcVIn4mnY3jnXueiYUidmgg8GAcwCU-SzJSvUbiK3/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=Bw3w51xV38Bvucd%2BNx0xvl5m3Iw%3D)

이를 피하기 위해서 다음의 아이디어가 나왔다.

1. 좋은 기준 모델 찾기  
2. 스케일링을 위한 너비, 깊이 및 해상도의 황금 비율 찾기  
3. 너비, 깊이 및 해상도의 황금 비율을 유지하면서 기본 모델의 각 차원을 확장

CNN을 공식화하면 이전의 표를 바탕으로 다음과 같이 나온다.

![](https://blog.kakaocdn.net/dna/bczbIm/btsgC3grP1I/AAAAAAAAAAAAAAAAAAAAAOYf0LtXXMx95vEQX_xC1tx7FqJS-yqDX-h2DD5gUB15/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=sNFBMAsvPsS2LfS6y1a6Lz8ENOg%3D)

복합 스케일링을 적용하고 모든 단계와 레이어는 검색 공간을 줄이기 위해 배율 인수를 공유한다는 가정을 추가해보자.

![](https://blog.kakaocdn.net/dna/bPadL7/btsgFuYs9SH/AAAAAAAAAAAAAAAAAAAAANQy8dLYLHKK2fV-tix0jzL2d48jUoljL69vsfQbyodC/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=vy5OMxw2rPlAMOQWhHcfnHJuxrU%3D)

그렇게 되면 CNN의 FLOPS( "floating-point operations"의 약어로, 모델이 수행하는 부동소수점 연산의 총량을 나타냅니다.)는 d, w^2, r^2에 비례하게 된다. 즉, 다음과 같은 형태가 나온다.

![](https://blog.kakaocdn.net/dna/b0iJcZ/btsgCGsvSGL/AAAAAAAAAAAAAAAAAAAAAI1e3F_PeQb75IgWPmGmKVAFn7Daj5zIAR7-5k-Fj_bb/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=ZDTu%2Foi4Grb9f9IXAp26CYupu0w%3D)
![](https://blog.kakaocdn.net/dna/bqVbQT/btsgDhMQxXT/AAAAAAAAAAAAAAAAAAAAAGbRMColcSgRPKB9ySkWqpIqlSzJUB4lxQ-7ibYYqzO3/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=q5Y1LrKQHC8w81%2FUvJyXy%2FddbGM%3D)

위의 공식과 이전의 Baseline Model을 통해 Golden Ratio를 찾았더니, 𝝓 = 𝟏일때, 𝜶 =1.2, 𝜷=1.1, 𝜸=1.15의 비율을 알게되었다.

이렇게 7개의 𝝓에 대한 𝑤 = 𝛼^𝝓, 𝑑 = 𝛽^𝝓, 𝑟 = 𝛾^𝝓를 구하여 만든 모델이 EfficientNet이다.

EfficientNet의 Performance를 보자.

![](https://blog.kakaocdn.net/dna/ba0FDl/btsgKn5uGih/AAAAAAAAAAAAAAAAAAAAADGTusAXaAgOVpJoJtRak-IvP5cEPUvGJcAt0DXxApS3/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=1F1drAn9T2Gm67hZIxblBzOU4p4%3D)

이제는 모델이 커질수록 ACC도 높아지는 것을 알 수 있다.

다른 모델들과의 비교도 다음과 같이 진행되었고, 파라미터와 성능이 다른 모델에 비해 좋은것을 확인할 수 있다.

![](https://blog.kakaocdn.net/dna/badmpT/btsgDidVzpi/AAAAAAAAAAAAAAAAAAAAADqak6q27r5vKhVsd-fgUT2Nql_zMvLS3ZLO9xxdz0ga/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=%2BQnR%2B8z6wPcuhuTA4BYZaXf%2Fgn4%3D)

스케일링 비교는 다음과 같다.

![](https://blog.kakaocdn.net/dna/beARCa/btsgE7JbT3G/AAAAAAAAAAAAAAAAAAAAABSvwc5mMbjpapna8ppl43HrxKVKganHv1pedk_I4sBJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=jUKXWRjkFzt7pFu9poTDptujWjg%3D)

Compind scaling도 잘 적용되는 것을 볼 수 있다.

![](https://blog.kakaocdn.net/dna/bao9j2/btsgFdbhl3c/AAAAAAAAAAAAAAAAAAAAAHBFftBl7UUDjskZZXLVwJTB7slyKnuHitJucAJGoNwe/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=gHmwaxBPrUfq%2BJinBjxAHOBIRyI%3D)

MobileNet 및 ResNet을 확장하기 위해 동일한 접근 방식을 적용해보면 다음과 같다.

![](https://blog.kakaocdn.net/dna/Wk51P/btsgCF1vCUP/AAAAAAAAAAAAAAAAAAAAAJCV1BawNg_gWrrWPNK6q-9taUBh7EKJYijyJXa_E4df/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=XaPkbp6pPB0w4%2FWOPki68WdtagE%3D)
