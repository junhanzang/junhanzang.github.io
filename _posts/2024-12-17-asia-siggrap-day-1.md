---
title: "ASIA SIGGRAP DAY 1"
date: 2024-12-17 23:13:54
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://asia.siggraph.org/2024/>

[Home - SIGGRAPH Asia 2024](https://asia.siggraph.org/2024/)

## **시그라프 아시아 2024: 도쿄에서의 첫날 경험**

아시아 최고의 컴퓨터 그래픽 및 인터랙티브 기술 컨퍼런스, **시그라프 아시아 2024**가 도쿄 국제 포럼(Tokyo International Forum)에서 열렸습니다. 전 세계의 개발자와 연구자들이 최신 기술과 연구를 선보이는 이 행사에, 저는 많은 것을 배우고자 퇴직금의 상당 부분을 사용해 입장료만 200만 원을 투자했습니다.

행사는 화요일에 시작되었지만, 저는 하루 전인 월요일부터 도쿄에 도착해 일정을 준비했습니다. 첫날 오전에는 기조 연설과 각 발표자가 자신의 세션을 1분간 소개하는 시간이 있었습니다. 연설에서는 "자신의 분야 외에도 다양한 주제를 접하라"는 조언이 있었고, 저 역시 흥미로운 그래픽스 주제들을 접할 수 있었습니다. 하지만 솔직히 인공지능 외의 분야는 이해가 쉽지 않았습니다.

시그라프 아시아에서는 동시에 3개의 세션이 진행되었고, 저는 주로 \*\*Hall B7(1)\*\*과 \*\*Hall B5(2)\*\*에 머물렀습니다. 아쉬웠던 점은 발표 직후 질문 시간이 없고, 세 발표가 모두 끝난 후에야 밖에서 발표자와 소통할 수 있다는 점이었습니다. 떠오른 영감을 바로 나눌 수 없다는 점이 다소 불편했습니다.

이제 첫날 제가 참석한 세션들을 하나씩 정리해보겠습니다. 그중 첫 번째는 **‘Make it Yours - Customizing Image Generation’** 세션입니다.

# MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation

<https://dl.acm.org/doi/10.1145/3680528.3687662>

[MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation | SIGGRAPH Asia 2024 Conference P](https://dl.acm.org/doi/10.1145/3680528.3687662)

Attention을 MOE에서 영감을 받은 논문으로 나중에 나오겠지만, neurips 2024의 Multi head LORA와 비슷한 결이었다.

Multi-subject composition을 personalization finetuning을 위해 진행된 논문이고 실제로도 좋은 논문이었다.

학습을 위해 여러개의 loss를 사용하는데, 이 때 맨처음에 들었을때는 denoising, router loss, object loss등 다양하게 사용하는데 router loss, object loss에 대한 GT가 없었다. 이는 운좋게 물어봤을때, ai를 통해 해당 이미지에서 뽑아낸 것들을 GT를 사용했다. 또한 stable diffusion을 기반으로 했지만, 최근에 나온 FLUX에도 적용했다고 하면서 자기의 앱을 보여준 것도 재미있었다.

추천하는 논문이다.

# ReVersion: Diffusion-Based Relation Inversion from Images

<https://dl.acm.org/doi/10.1145/3680528.3687658>

[ReVersion: Diffusion-Based Relation Inversion from Images | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687658)

이 논문은 loss에 대해서 많이 말을 했는데, 생각보다 loss가 중요한게 아니라 결국 저자의 아이디어인 pesudo-word embedding에서 activated prepositions가 중요했다. 실제 저자와 말을 나누었을때, 처음 알게 되었다. 그림에도 학습하지 모델을 학습하지 않는다고 lock을 걸어두기도 했고 말이다.

# PALP: Prompt Aligned Personalization of Text-to-Image Models

<https://dl.acm.org/doi/10.1145/3680528.3687604>

[PALP: Prompt Aligned Personalization of Text-to-Image Models | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687604)

데이터 확보의 중요성을 말한것 같은데, 내 느낌에는 당시 발표에서는 뭔가 설명이 부족했다. 그래서 따로 찾아봤을때 ....

# Customizing Text-to-Image Models with a Single Image Pair

<https://dl.acm.org/doi/10.1145/3680528.3687642>

[Customizing Text-to-Image Models with a Single Image Pair | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687642)

standard 형식들은 image pair중에서 기초가 되는 부분에 overfitting되는 부분이 많았다. 따라서 이를 Lora를 사용해서 해결 했는데, Lora를 서로 다른 걸 2개를 사용해서 창의적인 논문이었다. Content, pre, style 순으로 noise를 사용했고, CFG 부분도 사용가능하고 현재 FLUX에서도 사용가능하다는 점에서 좋았다.

추천하는 논문이다.

# Customizing Text-to-Image Diffusion with Object Viewpoint Control

<https://dl.acm.org/doi/10.1145/3680528.3687564>

[Customizing Text-to-Image Diffusion with Object Viewpoint Control | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687564)

사진을 넣으면 360와 같은 각도를 잘 얻게해주는 것이다

360도에 대한 것을 최대한 많이 학습시켜서 진행한 것이고, text prompt를 위해서 T5 Encoder를 활용했다. nerflayer를 통해 3d를 활용해서 render하기 때문에 학습안했던 것도 만들 수 있다고 한다. 하지만 focal length, vertical translate에서 약점이 있고 training 데이터에 문제가 있다고 한다. 이 부분은 나중에 neurips 2024 논문이었나 구글에서 발표한 것이었나 이미지 한장으로 3d를 만드는 논문이 있는데, 그것을 사용하면 되겠다라고 생각했다.

# Identity-Preserving Face Swapping via Dual Surrogate Generative Models

<https://dl.acm.org/doi/10.1145/3676165>

[Identity-Preserving Face Swapping via Dual Surrogate Generative Models | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3676165)

loss 부분은 잘 설명을 이해를 못했고, ip adapter를 사용한 것만 이해했다.

이후 발표자들과 최대한 대화를 나누려고 노력하고 다음 세션으로 넘어갔다.

다음 세션은 Design it all: font, paint, and colors 였다.

# HFH-Font: Few-shot Chinese Font Synthesis with Higher Quality, Faster Speed, and Higher Resolution

<https://arxiv.org/abs/2410.06488>

[HFH-Font: Few-shot Chinese Font Synthesis with Higher Quality, Faster Speed, and Higher Resolution](https://arxiv.org/abs/2410.06488)

한자를 쓰는 중국화 문화권들에게 있을만한 고민이다. 근간을 이루는 한자들을 64x64로 바꾸고 이를 조합해서 업스케일 하는 방식으로 이미지로서 다양하게 표현이 가능하다는게 장점이다. font diffuser와 비슷하다고 느낌. Reference가 input으로 content cinder가 transformer의 q, style encpoder가 k,v로 crossattenion을 사용했는데, 굉장히 흥미로웠다. super resolution으로는 esrgan을 사용했고, SDS라는 생소한 방식으로 진행되서 재미있었음. 추가적으로 백터라이제이션과 finetune도 가능해서 엄청 추천하는 논문이다.

# SD-πXL: Generating Low-Resolution Quantized Imagery via Score Distillation

<https://dl.acm.org/doi/full/10.1145/3680528.3687570>

[SD-πXL: Generating Low-Resolution Quantized Imagery via Score Distillation | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/full/10.1145/3680528.3687570)

pixel 이미지를 만들기위한 것이었으면서, 디노이즈가 condition에 들어가는 것이 특징이다. 이게 뭘 의미했는지 기억이 않나지만 softmax -> argmax -> convexsum 진행이 특징이었고, argmax로 진행시 patch를 통해 일정 부분을 보는 것인지 아니면 다른 건지 잘이해가 안되는 것이 있었음. 그리고 많이 느리다는 단점도 있음. 1개 이미지 뽑는데 3시간...

# ProcessPainter: Learning to draw from sequence data

<https://dl.acm.org/doi/10.1145/3680528.3687596>

[ProcessPainter: Learning to draw from sequence data | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687596)

사람이 그리는 방식으로 진행되는 그림이었음. 최근에 X에서 나온 이미지가 가로로 천천히 그려나가는 것이 갑자기 생각나서 적어봄. 이 논문에서는 비교군이 좀 아쉬웠던것 같고, inverse painting으로 한개씩 그려서 사람처럼 그리는 것으로 rl 방식처럼 진행됨. 시간이 비디오의 시간이며 실제 화가가 그리는 방식으로 하나씩 진행되며, 뭘 어디에 그릴지 모두 text, mask 데이터를 바탕으로 진행된다.

# LVCD: Reference-based Lineart Video Colorization with Diffusion Models

<https://dl.acm.org/doi/10.1145/3687910>

[LVCD: Reference-based Lineart Video Colorization with Diffusion Models | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687910)

keyframe은 color filling이며 color는 디퓨젼 베이스였다. 실제는 빠르고, 애니메이션은 느리기 때문에 reference attention을 제안함. SVD는 14프레임, 애니메이션은 200프레임이라서 이어서 만들다보니 에러가 누적되어서 뭉게지는 현상이 발생. 이를 위해서 short-term consistency -> overlapped blending을 사용해서 진행함. 대충 프로세스만 봐서는 오래걸릴 것으로보아서 좀 속도를 개선해야되는 부분이 있어보임.

# Colorful Diffuse Intrinsic Image Decomposition in the Wild

<https://dl.acm.org/doi/10.1145/3687984>

[Colorful Diffuse Intrinsic Image Decomposition in the Wild | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687984)

기억이 잘안나... albedo x shading + shading, chroma, albeddo network -> 다이렉트로 가는 것, diffuse shading network 등 다양하게 사용했는데도 빠르다고 함. 이게 좋긴한데, 모델을 3개를 엮어서 했다고하니까 뭔가 좀 아쉬운?

다음 세션은 Color and Display 였다.

# COMFI: A Calibrated Observer Metameric Failure Index for Color Critical Tasks

<https://dl.acm.org/doi/10.1145/3680528.3687701>

[COMFI: A Calibrated Observer Metameric Failure Index for Color Critical Tasks | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687701)

518 simulated observe를 몬테카를로로 만듬 -> 만든 것이 어떤 효과를 주는지가 중요해보이는데 잘이해를 못했음... 내분야가 아니라서... 새로운 메트릭을 만들었는데, 신뢰도가 얼마나되는지 잘 모르겠음...

# Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation

<https://dl.acm.org/doi/10.1145/3680528.3687600>

[Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687600)

SLM

아이박스를 벗어나면 안보이는 부분이있어고, 잔상을 지우기 위해서 마스크를 사용해도 최적화가 안되서 이를 최적화하는 방향에 대한 논문임. 마스크를 만들기 위해서 푸리에 후, 복사한 것을 바탕으로 다시 만들어서 주는 것으로 stocathstic 딥러닝 방식과 같은 back drop을 진행함. 신선했음

# elaTCSF: A Temporal Contrast Sensitivity Function for Flicker Detection and Modeling Variable Refresh Rate Flicker

<https://dl.acm.org/doi/10.1145/3680528.3687586>

[elaTCSF: A Temporal Contrast Sensitivity Function for Flicker Detection and Modeling Variable Refresh Rate Flicker | SIGGRAPH As](https://dl.acm.org/doi/10.1145/3680528.3687586)

Flicer - 깜빡임은 사람에게 8, 16 hz 쯤에 민감하게 반응함. 내분야가 아니라 잘 모르겠지만, 초반에는 분석하고 이를 바탕으로 변수화해서, 새로운 function을 만들고 flicker detection 모델을 만들어서 진행하는 것이고 이를 vr에 적용하는 논문

# Perspective-Aligned AR Mirror with Under-Display Camera

<https://dl.acm.org/doi/10.1145/3687995>

[Perspective-Aligned AR Mirror with Under-Display Camera | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687995)

얘는 제품 소개에 더 가깝지 않을까 싶었음 사실... transparent display를 통해 카메라가 캡쳐 가능하게 만듬. 이를 바탕으로 이미지를 모아서 디노이징하는 모델을 통해 다시 깨끗한 상태로 만듬. 이 이미지를 복원 후, ar을 해주는 것이 이 제품. Full hd를 준다는 부분은 괜찮았고, 레이턴시 이슈도 별로 없어서 미팅으로 사용 가능하다고 함

# AR-DAVID: Augmented Reality Display Artifact Video Dataset

<https://dl.acm.org/doi/10.1145/3687969>

[AR-DAVID: Augmented Reality Display Artifact Video Dataset | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687969)

quality metric을 만드는 방법. Ar이 traditional과 얼마나 다를까에서부터 시작됨. 여러가지 오류들이 있고 432 컨디션으로 정의함. ASAP과 PWCMP로 result를 scaling했고 백드라운드는 뭐가 더 좋다 그런것이 없음.

결론은 ar 퀄리티는 결국 traditional하고 다름

# V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians

<https://dl.acm.org/doi/10.1145/3687935>

[V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians | ACM Transactions on Graphics](https://dl.acm.org/doi/10.1145/3687935)

비디오는 퀄리티가 제한되어 있음. 2d를 3d로 바꾸는건데, 2d에서 가우시안 스플리팅으로 바꾸고, uint 8을 사용하는 부분이 특이점. 이때 퀄리티가 내려오는 지점이 없는 것이 관건인데, 이때 발표에서는 못들었던것 같음. 일반적으로 사용하는 point colud를 바탕으로 만들고, 이를 바탕으로 temporal consistent regulaization -> residual에 loss를 사용한다는데...

긴 비도는 없어보였고, 생각보다 느려보였음. 인터넷 전송이라그런지...

이후 저녘에는 모든 참여자들이 가는 친목회? 환영파티? 그런거 같는데, 같이 온 사람들끼리만 이야기하는 분위기라서 아쉬웠음. 혼자와서 뭔가 말할 상대도 없어서 쉽지 않았다...
