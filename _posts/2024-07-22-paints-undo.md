---
title: "Paints Undo"
date: 2024-07-22 02:16:38
categories:
  - 개인용
---

<https://huggingface.co/spaces/MohamedRashad/PaintsUndo>

[PaintsUndo - a Hugging Face Space by MohamedRashad](https://huggingface.co/spaces/MohamedRashad/PaintsUndo)

<https://github.com/lllyasviel/Paints-UNDO>

[GitHub - lllyasviel/Paints-UNDO: Understand Human Behavior to Align True Needs](https://github.com/lllyasviel/Paints-UNDO)

<https://lllyasviel.github.io/pages/paints_undo/>

[PaintsUndo: A Base Model of Drawing Behaviors in Digital Paintings](https://lllyasviel.github.io/pages/paints_undo/)

![](/assets/images/posts/215/img.jpg)

출처 : <https://x.com/dding_ac/status/1365622485526319112>

위의 그림이 첨부 파일처럼 변경됨

[c7823ded-c8f3-4fe2-a30c-0760f2eaa54f.mp4

5.55MB](./file/c7823ded-c8f3-4fe2-a30c-0760f2eaa54f.mp4)

코드를 보니 다음과 같을 것으로 예상

원본 이미지에 clip 모델로 text를 추출하고, 이를 바탕으로 ddim 또는 ddpm 형식으로 만든다. 만들어지는 중간 단계의 noise + 원본 이미지를 통해 step들을 추출 후, 이를 바탕으로 gif를 만드는 방식.

또 다른 예상으로는 원본 이미지에 clip으로 딴거 negative noise설정해주고 완전히 지워질때까지 역산해서 만드는 방식으로 예상
