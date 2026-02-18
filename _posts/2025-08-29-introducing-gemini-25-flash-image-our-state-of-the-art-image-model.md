---
title: "Introducing Gemini 2.5 Flash Image, our state-of-the-art image model"
date: 2025-08-29 13:25:57
categories:
  - 소식
tags:
  - nano-banana
---

<https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/?utm_source=pytorchkr&ref=pytorchkr>

[Introducing Gemini 2.5 Flash Image, our state-of-the-art image model- Google Developers Blog](https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/?utm_source=pytorchkr&ref=pytorchkr)

![](/assets/images/posts/596/img.jpg)

오늘 우리는 최신 이미지 생성 및 편집 모델인 **Gemini 2.5 Flash Image(일명 nano-banana)**를 소개하게 되어 기쁩니다. 이번 업데이트를 통해 여러 이미지를 하나로 결합하고, 스토리텔링을 위한 캐릭터 일관성을 유지하며, 자연어를 사용한 특정 변환을 수행하고, Gemini의 세계 지식을 활용해 이미지를 생성 및 편집할 수 있습니다.

올해 초 Gemini 2.0 Flash에서 기본 이미지 생성 기능을 처음 출시했을 때, 여러분은 짧은 지연 시간, 비용 효율성, 사용 편의성을 높이 평가해주셨습니다. 하지만 동시에 더 높은 품질의 이미지와 더 강력한 창의적 제어 기능이 필요하다는 피드백도 주셨습니다.

이 모델은 현재 개발자를 위한 **Gemini API 및 Google AI Studio**, 그리고 엔터프라이즈를 위한 **Vertex AI**에서 바로 사용 가능합니다. **Gemini 2.5 Flash Image**의 가격은 **100만 출력 토큰당 30달러**이며, 각 이미지는 **1290 출력 토큰(이미지당 약 0.039달러)**이 소요됩니다. 입력 및 출력의 다른 모든 모달리티는 Gemini 2.5 Flash 가격 정책을 따릅니다.

![](/assets/images/posts/596/img.png)

[(lmarena results come from](https://lmarena.ai/leaderboard) <https://lmarena.ai/leaderboard)>

**실행 중인 Gemini 2.5 Flash Image**  
Gemini 2.5 Flash Image를 활용한 빌드를 더욱 쉽게 할 수 있도록, Google AI Studio의 “빌드 모드(build mode)”에 대대적인 업데이트를 진행했습니다(앞으로도 추가 업데이트 예정). 아래 예시에서 보듯이, 사용자는 맞춤형 AI 기반 앱으로 모델의 기능을 빠르게 테스트할 수 있을 뿐만 아니라, 단 한 줄의 프롬프트만으로 앱을 리믹스하거나 아이디어를 현실화할 수 있습니다. 앱을 완성한 후에는 Google AI Studio에서 바로 배포하거나 GitHub에 코드를 저장할 수도 있습니다.

예를 들어, “사용자가 이미지를 업로드하고 다양한 필터를 적용할 수 있는 이미지 편집 앱을 만들어줘”라는 프롬프트를 입력하거나, 제공된 템플릿 중 하나를 선택해 리믹스할 수 있으며, 이 모든 기능은 무료로 이용할 수 있습니다.

**캐릭터 일관성 유지**  
이미지 생성에서 근본적인 도전 과제 중 하나는 여러 프롬프트와 편집 과정에서 캐릭터나 객체의 외형을 일관되게 유지하는 것입니다. 이제 동일한 캐릭터를 다른 환경에 배치하거나, 하나의 제품을 새로운 장면에서 여러 각도로 보여주거나, 브랜드 자산을 일관되게 생성하면서도 피사체를 그대로 보존할 수 있습니다.

이를 시연하기 위해 Google AI Studio에 템플릿 앱을 제작했으며, 사용자가 손쉽게 커스터마이즈하고 그 위에 코드를 추가할 수 있도록 설계했습니다.

<https://storage.googleapis.com/gweb-developer-goog-blog-assets/original_videos/PastForward-HighRes.mp4>

(시퀀스 생략)

캐릭터 일관성을 넘어, 이 모델은 시각적 템플릿을 따르는 데에도 뛰어난 성능을 보입니다. 이미 개발자들이 단일 디자인 템플릿만으로 **부동산 매물 카드, 직원용 배지, 전체 카탈로그에 대한 동적 제품 목업**과 같은 영역을 탐구하는 사례가 나타나고 있습니다.

![](/assets/images/posts/596/img_1.png)

**프롬프트 기반 이미지 편집**  
Gemini 2.5 Flash Image는 자연어를 통해 특정 영역 변환과 정밀한 로컬 편집을 지원합니다. 예를 들어, 이 모델은 이미지의 배경을 흐리게 하거나, 티셔츠의 얼룩을 제거하거나, 사진 속 인물을 통째로 지우거나, 피사체의 포즈를 바꾸거나, 흑백 사진에 색을 입히는 등 간단한 프롬프트만으로 다양한 편집을 수행할 수 있습니다.

이러한 기능을 실제로 보여주기 위해, 우리는 **UI와 프롬프트 기반 제어를 모두 갖춘 사진 편집 템플릿 앱**을 AI Studio에서 제작했습니다.

![](/assets/images/posts/596/img_2.png)

**내재된 세계 지식(Native world knowledge)**  
기존의 이미지 생성 모델들은 미적 이미지를 생성하는 데 뛰어났지만, 실제 세계에 대한 깊은 의미적 이해는 부족했습니다. 그러나 **Gemini 2.5 Flash Image**는 Gemini의 세계 지식을 활용하여 새로운 활용 사례들을 가능하게 합니다.

이를 시연하기 위해, 우리는 Google AI Studio에서 **단순한 캔버스를 대화형 교육 튜터로 변환하는 템플릿 앱**을 제작했습니다. 이 앱은 모델이 손으로 그린 다이어그램을 읽고 이해하며, 실제 세계와 관련된 질문을 돕고, 복잡한 편집 지시를 단 한 번에 따를 수 있는 능력을 보여줍니다.

<https://storage.googleapis.com/gweb-developer-goog-blog-assets/original_videos/gemini-2-5-flash-image-native-world-knowledge.mp4>

(예시 프롬프트 및 모델 결과)

**다중 이미지 융합(Multi-image fusion)**  
**Gemini 2.5 Flash Image**는 여러 입력 이미지를 이해하고 하나로 결합할 수 있습니다. 예를 들어, 특정 객체를 장면에 배치하거나, 방을 새로운 색상 조합이나 질감으로 재스타일링하거나, 단 한 줄의 프롬프트로 이미지를 융합할 수 있습니다.

이 기능을 시연하기 위해, 우리는 Google AI Studio에서 **제품을 새 장면으로 드래그해 사실적인 융합 이미지를 빠르게 생성할 수 있는 템플릿 앱**을 제작했습니다.

<https://storage.googleapis.com/gweb-developer-goog-blog-assets/original_videos/gemini-2-5-flash-image-multi-image-fusion_1.mp4>

(시퀀스 생략)

**빌드를 시작하세요**  
개발자 문서를 확인하고 **Gemini 2.5 Flash Image**로 빌드를 시작해보세요. 이 모델은 현재 Gemini API와 Google AI Studio를 통해 **프리뷰(preview)** 단계에서 제공되며, 몇 주 안에 안정(stable) 버전으로 전환될 예정입니다. 여기서 소개한 모든 데모 앱은 Google AI Studio에서 vibe 코드로 작성되었기 때문에, 단순히 프롬프트만으로도 리믹스하거나 원하는 대로 커스터마이즈할 수 있습니다.

**OpenRouter.ai**는 오늘부터 300만 명 이상의 개발자들이 Gemini 2.5 Flash Image를 사용할 수 있도록 협력하고 있습니다. 현재 OpenRouter에서 제공되는 480개 이상의 라이브 모델 중, 이미지를 생성할 수 있는 첫 번째 모델이 바로 Gemini 2.5 Flash Image입니다.

또한, 생성형 미디어를 위한 대표적인 개발자 플랫폼인 **fal.ai**와도 협력하여 Gemini 2.5 Flash Image를 더 넓은 개발자 커뮤니티에서 사용할 수 있도록 지원하게 되어 매우 기쁩니다.

Gemini 2.5 Flash Image로 생성되거나 편집된 모든 이미지에는 **보이지 않는 SynthID 디지털 워터마크**가 포함되어, AI 생성 혹은 편집된 이미지임을 식별할 수 있습니다.

```
from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client()

prompt = "Create a picture of my cat eating a nano-banana in a fancy restaurant under the gemini constellation"

image = Image.open('/path/to/image.png')

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt, image],
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO(part.inline_data.data))   
    image.save("generated_image.png")
```

Python 예제 코드

우리는 **장문 텍스트 렌더링**, **더 안정적인 캐릭터 일관성**, 그리고 이미지 내 세밀한 사실적 표현(factual representation)을 개선하기 위해 계속 노력하고 있습니다. 개발자 포럼이나 X(구 Twitter)를 통해 여러분의 피드백을 기다리고 있습니다.

**Gemini 2.5 Flash Image로 여러분이 무엇을 만들지 무척 기대됩니다!**
