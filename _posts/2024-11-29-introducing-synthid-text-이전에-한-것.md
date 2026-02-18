---
title: "Introducing SynthID Text (이전에 한 것)"
date: 2024-11-29 15:56:58
categories:
  - 인공지능
---

<https://huggingface.co/blog/synthid-text>

[Introducing SynthID Text](https://huggingface.co/blog/synthid-text)

텍스트가 사람이 쓴 것인지 AI가 생성한 것인지 구분하기 어렵다고 느끼나요? AI 생성 콘텐츠를 식별하는 능력은 정보에 대한 신뢰를 증진시키고, 잘못된 출처 표기나 허위 정보와 같은 문제를 해결하는 데 필수적입니다. 오늘 Google DeepMind와 Hugging Face는 Transformers v4.46.0에서 SynthID Text를 출시하며 이에 대해 기쁘게 발표합니다. 이 기술은 생성 작업을 위해 logits 프로세서를 사용해 AI 생성 텍스트에 워터마크를 적용하고, 이를 분류기를 통해 감지할 수 있도록 해줍니다.

SynthID Text 알고리즘에 대한 전체 기술적 세부 사항은 Nature에 발표된 논문에서 확인할 수 있으며, SynthID Text를 제품에 적용하는 방법에 대해서는 Google의 책임 있는 GenAI 툴킷을 참고하세요.

**작동 방식**  
SynthID Text의 주요 목표는 AI로 생성된 텍스트에 워터마크를 인코딩하여 텍스트가 여러분의 LLM에서 생성된 것인지 여부를 파악할 수 있게 하는 것입니다. 이를 통해 LLM의 기본 동작을 변경하거나 생성 품질에 부정적인 영향을 미치지 않도록 합니다. Google DeepMind는 g-함수라고 불리는 의사 난수 함수(pseudo-random function)를 이용해 워터마크를 LLM의 생성 과정에 추가하는 기술을 개발했습니다. 이 워터마크는 사람에게는 인지되지 않지만, 훈련된 모델에는 감지될 수 있습니다. 이 기술은 모델의 generate() API를 사용하여 수정 없이 모든 LLM과 호환되는 생성 유틸리티로 구현되었으며, 워터마크가 적용된 텍스트를 인식하도록 탐지기를 훈련하는 종단 간 예제도 포함하고 있습니다. SynthID Text 알고리즘에 대한 더 자세한 내용은 연구 논문을 참고하세요.

**워터마크 구성하기**  
워터마크는 데이터클래스를 사용하여 구성되며, 여기에서 g-함수와 이를 토너먼트 샘플링 과정에 적용하는 방식을 매개변수화합니다. 사용하는 각 모델은 자체적인 워터마크 설정을 가지며, 이 설정은 안전하고 비공개적으로 저장되어야 합니다. 그렇지 않으면 다른 사람이 워터마크를 복제할 수 있습니다.

모든 워터마크 설정에서 두 가지 매개변수를 정의해야 합니다:

- **keys 매개변수**: 모델의 어휘 집합에서 g-함수 점수를 계산하는 데 사용되는 정수 목록입니다. 탐지 가능성과 생성 품질 간의 균형을 맞추기 위해서는 20~30개의 고유하고 무작위로 생성된 숫자를 사용하는 것이 좋습니다.
- **ngram\_len 매개변수**: 강인함(robustness)과 탐지 가능성 간의 균형을 맞추기 위해 사용됩니다. 값이 클수록 워터마크는 더 탐지하기 쉬워지지만, 변화에 취약해집니다. 기본값으로는 5가 좋지만, 최소 2 이상이어야 합니다.

성능 요구에 따라 워터마크를 추가로 구성할 수 있습니다. 자세한 내용은 SynthIDTextWatermarkingConfig 클래스를 참고하세요.

연구 논문에는 특정 설정 값이 워터마크 성능에 어떤 영향을 미치는지에 대한 추가 분석이 포함되어 있습니다.

**워터마크 적용하기**  
워터마크 적용은 기존 생성 호출에 간단한 변경을 가하는 것입니다. 구성 설정을 정의한 후, SynthIDTextWatermarkingConfig 객체를 watermarking\_config= 매개변수로 model.generate()에 전달하면 모든 생성된 텍스트에 워터마크가 추가됩니다. SynthID Text Space에서 워터마크 적용의 인터랙티브 예제를 확인하고, 여러분이 차이를 알아낼 수 있는지 확인해보세요.

```
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
)

# 표준 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained('repo/id')
model = AutoModelForCausalLM.from_pretrained('repo/id')

# SynthID Text 설정
watermarking_config = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, ...],
    ngram_len=5,
)

# 워터마크를 포함한 텍스트 생성
tokenized_prompts = tokenizer(["your prompts here"])
output_sequences = model.generate(
    **tokenized_prompts,
    watermarking_config=watermarking_config,
    do_sample=True,
)
watermarked_text = tokenizer.batch_decode(output_sequences)
```

**워터마크 감지하기**  
워터마크는 훈련된 분류기에 의해 감지될 수 있지만, 인간에게는 인지되지 않도록 설계되었습니다. 모델과 함께 사용하는 각 워터마크 설정은 그 마크를 인식하도록 훈련된 탐지기가 필요합니다.

기본적인 탐지기 훈련 과정은 다음과 같습니다:

1. 워터마크 설정을 결정합니다.
2. 워터마크가 있는지 없는지 여부와 훈련용 또는 테스트용으로 나뉜 탐지기 훈련 세트를 수집합니다. 최소 1만 개의 예제를 사용하는 것을 권장합니다.
3. 모델을 사용하여 워터마크가 없는 출력을 생성합니다.
4. 모델을 사용하여 워터마크가 있는 출력을 생성합니다.
5. 워터마크 탐지 분류기를 훈련시킵니다.
6. 워터마크 설정과 관련 탐지기를 사용해 모델을 실제 환경에서 사용할 수 있도록 준비합니다.

Transformers에는 특정 워터마크 설정을 사용해 워터마크가 있는 텍스트를 인식하는 탐지기를 훈련하는 종단 간 예제와 함께 베이즈 탐지기 클래스가 제공됩니다. 동일한 토크나이저를 사용하는 모델들은 워터마크 설정과 탐지기를 공유할 수 있으며, 이 경우 탐지기의 훈련 세트에는 워터마크를 공유하는 모든 모델의 예제가 포함되어야 합니다.

훈련된 탐지기는 조직 내에서 접근할 수 있도록 HF Hub에 비공개로 업로드할 수 있습니다. Google의 책임 있는 GenAI 툴킷에서는 SynthID Text를 제품에 적용하는 방법에 대해 더 자세히 설명하고 있습니다.

**한계점**  
SynthID Text 워터마크는 텍스트의 일부를 자르거나 몇몇 단어를 수정하거나 가벼운 패러프레이징과 같은 일부 변형에 대해 강인함을 가집니다. 그러나 이 방법에도 몇 가지 한계가 있습니다.

- 워터마크 적용은 사실적인 응답에서는 효과가 덜합니다. 생성 정확도를 저하시키지 않으면서 추가할 기회가 적기 때문입니다.
- AI로 생성된 텍스트가 철저히 재작성되거나 다른 언어로 번역될 경우 탐지기의 신뢰 점수는 크게 감소할 수 있습니다.
- SynthID Text는 직접적으로 악의적인 의도를 가진 공격을 막기 위해 설계된 것은 아닙니다. 그러나 AI 생성 콘텐츠를 악용하기 어렵게 만들 수 있으며, 다른 접근 방식과 결합하여 콘텐츠 유형 및 플랫폼 전반에 걸친 더 나은 적용 범위를 제공할 수 있습니다.

**감사의 말**  
저자들은 이 연구에 기여한 Robert Stanforth와 Tatiana Matejovicova에게 감사의 뜻을 전합니다.
