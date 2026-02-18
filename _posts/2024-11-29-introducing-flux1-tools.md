---
title: "Introducing FLUX.1 Tools"
date: 2024-11-29 11:04:46
categories:
  - 인공지능
---

<https://blackforestlabs.ai/flux-1-tools/>

[Introducing FLUX.1 Tools](https://blackforestlabs.ai/flux-1-tools/)

![](/assets/images/posts/361/img.jpg)

FLUX.1 도구 소개  
2024년 11월 21일  
—

작성자: BlackForestLabs  
카테고리: 뉴스

오늘 우리는 FLUX.1 도구를 발표하게 되어 매우 기쁩니다. FLUX.1 도구는 우리의 기본 텍스트-이미지 모델인 FLUX.1에 제어 및 조정 기능을 추가하기 위해 설계된 모델 모음으로, 실제 이미지와 생성된 이미지를 수정하고 재생성할 수 있게 해줍니다. 이번 발표에서는 FLUX.1 도구가 네 가지의 뚜렷한 기능으로 구성되어 있으며, 이는 FLUX.1 [dev] 모델 시리즈의 일부로 오픈 액세스로 제공되고, FLUX.1 [pro]를 보완하는 BFL API에서도 사용할 수 있습니다.

- **FLUX.1 Fill**: 최첨단 인페인팅 및 아웃페인팅 모델로, 텍스트 설명과 이진 마스크를 사용하여 실제 및 생성된 이미지를 편집하고 확장할 수 있도록 지원합니다.
- **FLUX.1 Depth**: 입력 이미지에서 추출된 깊이 맵과 텍스트 프롬프트를 기반으로 구조적 가이드를 제공하도록 훈련된 모델입니다.
- **FLUX.1 Canny**: 입력 이미지에서 추출한 캐니 엣지와 텍스트 프롬프트를 기반으로 구조적 가이드를 제공하도록 훈련된 모델입니다.
- **FLUX.1 Redux**: 입력 이미지와 텍스트 프롬프트를 혼합하고 재생성할 수 있는 어댑터입니다.

이번 출시로 우리는 두 가지 약속을 강화합니다: 연구 커뮤니티를 위한 최첨단 오픈-가중치 모델을 제공하는 것과 동시에, API를 통해 최고의 기능을 제공하는 것입니다. 우리는 각 도구를 BFL API 내에서 FLUX.1 [pro] 변형으로 출시하며, 가이드-증류를 통해 오픈 액세스로 제공되는 FLUX.1 [dev] 변형으로 가중치와 추론 코드를 공개합니다. 또한, 이번에 발표된 모델들은 fal.ai, Replicate, Together.ai, Freepik, krea.ai와 같은 파트너들을 통해 이용할 수 있게 되어 매우 기쁩니다.

이후 섹션에서는 새로운 모델에 대한 세부 사항, 성능 분석 및 접근 방법을 다룰 예정입니다. FLUX 생태계가 이번 새로운 도구들로 얼마나 활발하게 보완될지 기대가 큽니다.

FLUX.1 Fill을 이용한 인페인팅 및 아웃페인팅  
FLUX.1 Fill은 Ideogram 2.0이나 AlimamaCreative의 FLUX-Controlnet-Inpainting과 같은 인기 있는 오픈 소스 변형을 능가하는 고급 인페인팅 기능을 도입합니다. 이 도구는 기존 이미지와 자연스럽게 통합되는 매끄러운 편집을 가능하게 합니다.

![](/assets/images/posts/361/img.png)

또한, FLUX.1 Fill은 오리지널 이미지의 경계를 넘어 이미지를 확장할 수 있는 아웃페인팅을 지원합니다.

![](/assets/images/posts/361/img_1.png)

우리는 이 모델의 성능을 벤치마크하여, 그 결과를 공개적으로 제공하고 있습니다. 결과에 따르면, FLUX.1 Fill [pro]는 다른 모든 경쟁 모델을 능가하며, 현 시점에서 가장 뛰어난 인페인팅 모델임을 입증했습니다. 그 뒤를 잇는 모델은 FLUX.1 Fill [dev]로, 독점 솔루션들을 능가하면서도 추론 효율성이 더 뛰어난 모습을 보였습니다.

![](/assets/images/posts/361/img_2.png)

FLUX.1 Fill [dev]는 Flux Dev License로 이용 가능하며:

- 전체 모델 가중치는 Hugging Face에서 이용 가능: [<https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev>]
- 추론 코드는 <https://github.com/black-forest-labs/flux>에서 이용 가능

FLUX.1 Fill [pro]는 [<https://docs.bfl.ml/>]를 통해 이용할 수 있습니다.

FLUX.1 Canny / Depth를 이용한 구조적 조건 설정  
구조적 조건 설정은 캐니 엣지 또는 깊이 감지를 사용하여 이미지 변환 시 정밀한 제어를 유지합니다. 엣지 또는 깊이 맵을 통해 원본 이미지의 구조를 보존함으로써 사용자는 텍스트 기반의 편집을 수행하면서도 핵심 구성을 유지할 수 있습니다. 이 접근법은 특히 이미지의 텍스처를 다시 지정하는 데 효과적입니다.

![](/assets/images/posts/361/img_3.png)

![](/assets/images/posts/361/img_4.png)

평가 결과에 따르면, 벤치마크 결과는 여기에서 확인할 수 있으며, FLUX.1 Depth는 Midjourney ReTexture와 같은 독점 모델을 능가하는 성능을 보였습니다. 특히, FLUX.1 Depth [pro]는 더 높은 출력 다양성을 제공하며, FLUX.1 Depth의 Dev 버전은 깊이 인지 작업에서 더 일관된 결과를 제공합니다. 캐니 엣지 모델의 경우, 벤치마크 결과는 여기에서 확인할 수 있으며, FLUX.1 Canny [pro]가 가장 우수한 성능을 보였고, FLUX.1 Canny [dev]가 그 뒤를 이었습니다.

![](/assets/images/posts/361/img_5.png)

FLUX.1 Canny / Depth는 두 가지 버전으로 제공됩니다: 최대 성능을 위한 전체 모델 버전과 개발 용이성을 위해 FLUX.1 [dev] 기반의 LoRA 버전입니다.

Flux Depth / Canny [dev]는 Flux Dev License로 이용 가능하며:

- 전체 모델 가중치는 Hugging Face에서 이용 가능: [<https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev>] [<https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev>]
- LoRA 가중치는 Hugging Face에서 이용 가능: [<https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora>] [<https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora>]
- 추론 코드는 <https://github.com/black-forest-labs/flux>에서 이용 가능

Flux.1 Depth / Canny [pro]는 <https://docs.bfl.ml/>에서 이용할 수 있습니다.

FLUX.1 Redux를 이용한 이미지 변형 및 리스타일링  
FLUX.1 Redux는 이미지 변형 생성을 위해 모든 FLUX.1 기본 모델을 위한 어댑터입니다. 입력 이미지를 주면, FLUX.1 Redux는 그 이미지를 약간 변형하여 재생성할 수 있으며, 이를 통해 주어진 이미지를 더욱 정교하게 다듬을 수 있습니다.

![](/assets/images/posts/361/img_6.png)

이 모델은 복잡한 워크플로우에 자연스럽게 통합되며, 프롬프트를 통해 이미지 리스타일링을 가능하게 합니다. 리스타일링 기능은 이미지와 프롬프트를 함께 제공하여 API를 통해 사용할 수 있습니다. 이 기능은 최신 모델인 FLUX1.1 [pro] Ultra에서 지원되며, 입력 이미지와 텍스트 프롬프트를 결합해 유연한 화면 비율의 고품질 4메가픽셀 출력을 생성할 수 있습니다.

![](/assets/images/posts/361/img_7.png)

우리의 벤치마크 결과에 따르면, FLUX.1 Redux는 이미지 변형 작업에서 최첨단 성능을 달성했습니다.

![](/assets/images/posts/361/img_8.png)

Flux.1 Redux [dev]는 Flux Dev License로 이용 가능하며:

- 모델 가중치는 Hugging Face에서 이용 가능: [<https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev>]
- 추론 코드는 <https://github.com/black-forest-labs/flux>에서 이용 가능

FLUX.1 Redux를 지원하는 FLUX1.1 [pro] Ultra는 <https://docs.bfl.ml/>에서 이용할 수 있습니다.

우리는 커뮤니티가 이번 새로운 도구 세트를 사용하여 어떤 것을 만들어낼지 기대하고 있습니다. 우리의 API를 [<https://api.bfl.ml/auth/login>]에서 시도해 보세요.
