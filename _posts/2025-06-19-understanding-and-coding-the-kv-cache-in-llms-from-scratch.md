---
title: "Understanding and Coding the KV Cache in LLMs from Scratch"
date: 2025-06-19 13:21:04
categories:
  - Article
---

**Sebastian Raschka, PhD**  
**2025년 6월 17일**

KV 캐시(KV caches)는 대규모 언어 모델(LLM)을 실제 환경에서 효율적으로 추론하는 데 있어 가장 핵심적인 기술 중 하나입니다. KV 캐시는 계산 효율적인 LLM 추론을 가능하게 하는 중요한 구성 요소입니다. 이 글에서는 KV 캐시가 개념적으로 어떻게 작동하는지, 그리고 코드 수준에서 어떻게 구현되는지를 처음부터 사람이 읽기 쉬운 방식으로 설명합니다.

기본적인 LLM 개념을 설명하는 기술 튜토리얼을 공유한 지 꽤 오랜 시간이 흘렀습니다. 현재 부상에서 회복 중이며 LLM 연구에 초점을 맞춘 더 큰 규모의 글을 준비하고 있는 가운데, 독자분들 중 여러 분이 요청하셨던 주제(제 저서 Building a Large Language Model From Scratch에는 포함되지 않았던 주제)에 대해 튜토리얼 형식의 글을 공유해보고자 합니다.

즐겁게 읽어주세요!

### 개요 (Overview)

간단히 말해, **KV 캐시(KV cache)**는 추론 시에 중간 단계에서 계산된 key(K)와 value(V)를 저장해두고 재사용하는 메커니즘입니다. 이를 통해 텍스트를 생성할 때 상당한 속도 향상을 얻을 수 있습니다. KV 캐시의 단점은 코드가 복잡해지고, 메모리 요구량이 증가하며(이 때문에 제가 이 개념을 처음 책에 포함시키지 않았습니다), **훈련 과정에서는 사용할 수 없다는 점**입니다. 그럼에도 불구하고, **LLM을 실제 환경에서 사용할 때 추론 속도 향상은 코드 복잡도 및 메모리 사용 증가라는 단점을 충분히 상쇄할 만큼 큰 장점**이 됩니다.

### KV 캐시란? (What Is a KV Cache?)

LLM이 텍스트를 생성하고 있다고 가정해 봅시다. 예를 들어, "Time"이라는 프롬프트가 주어진 상황을 생각해보죠. LLM은 잘 알려진 바와 같이 **한 번에 하나의 단어(또는 토큰)를 생성**합니다. 아래 그림은 그 다음 두 단계를 시각적으로 보여줍니다:

![](/assets/images/posts/573/img.png)

LLM은 텍스트를 한 토큰씩 생성합니다. 프롬프트 "Time"에서 시작하여 다음 토큰인 "flies"를 생성하고, 이후 "Time flies" 전체 문장을 다시 처리하여 "fast"를 생성합니다.

이 과정에는 중복된 연산이 포함되어 있습니다. 다음 그림은 이러한 **중복된 컨텍스트("Time flies")가 매 단계마다 재처리**되어야 하는 문제를 보여줍니다.

![](/assets/images/posts/573/img_1.png)

LLM은 중간 key/value 상태를 캐시하지 않기 때문에, 매번 새로운 토큰(예: "fast")을 생성할 때마다 전체 문장을 다시 인코딩합니다.

일반적으로 텍스트 생성 함수를 구현할 때는 매 단계의 **마지막 토큰만 사용**합니다. 하지만 위 시각화를 보면, **개념적 수준에서 비효율성이 존재**함을 알 수 있습니다. 이러한 비효율은 **attention 메커니즘** 자체를 자세히 들여다보면 더 명확해집니다. (Attention 메커니즘이 궁금하다면 제 책 『Build a Large Language Model From Scratch』 3장을 참고하거나, 제 글 Understanding and Coding Self-Attention, Multi-Head Attention, Causal-Attention, and Cross-Attention in LLMs도 읽어보세요.)

다음 그림은 LLM의 핵심인 **attention 메커니즘 계산 일부**를 보여줍니다. 여기서는 입력 토큰("Time"과 "flies")이 3차원 벡터로 인코딩되어 있습니다 (실제로는 훨씬 더 고차원입니다). W는 학습된 가중치 행렬로, 입력을 **key, value, query 벡터로 변환**하는 역할을 합니다.

![](/assets/images/posts/573/img.jpg)

LLM은 attention 계산 과정에서 각 입력 토큰(예: "Time", "flies")을 W\_k, W\_v라는 가중치 행렬을 통해 각각 key(k), value(v) 벡터로 변환합니다.

앞서 언급했듯이, LLM은 한 번에 하나의 토큰만 생성합니다. 예를 들어, 이전에 "fast"라는 단어가 생성되었다면 다음 프롬프트는 "Time flies fast"가 됩니다.  
이때의 처리는 다음과 같이 시각화할 수 있습니다:

![](/assets/images/posts/573/img_1.jpg)

LLM은 이전에 보았던 토큰("Time"과 "flies")에 대한 key/value 벡터를 **다시 계산**합니다. 세 번째 토큰 "fast"를 생성할 때도 k(1)/v(1), k(2)/v(2)를 재계산하며, 이를 **재사용하지 않습니다**. 이처럼 **KV 캐시 없이 반복 계산하는 것은 매우 비효율적**입니다.

이전 두 그림을 비교해보면, 첫 두 토큰의 key/value 벡터는 **완전히 동일**하다는 것을 알 수 있습니다. 따라서 매번 이를 다시 계산하는 것은 낭비입니다. 여기서 **KV 캐시의 핵심 아이디어**는, 이미 생성된 key/value 벡터를 저장해두고 다음 단계에서 **재사용**함으로써 이러한 불필요한 계산을 피하는 것입니다.

### KV 캐시 없이/있이 LLM이 텍스트를 생성하는 방식

(How LLMs Generate Text Without and With a KV Cache)

앞서 KV 캐시의 기본 개념을 살펴보았으니, 이제 본격적으로 코드 구현을 보기 전에 조금 더 자세히 설명해보겠습니다.  
예를 들어, **"Time flies fast"**라는 문장을 KV 캐시 없이 생성하는 과정을 생각해보면 다음과 같습니다:

![](/assets/images/posts/573/img_2.png)

"Time"과 "flies"라는 토큰이 매번 새로운 토큰을 생성할 때마다 **반복해서 다시 계산**된다는 점에 주목하세요.  
KV 캐시는 이런 비효율을 해결합니다. **이미 계산된 key 및 value 벡터를 저장하고 재사용**하기 때문입니다:

1. 처음에는 입력 토큰들에 대해 key 및 value 벡터를 계산하고 이를 **캐시에 저장**합니다.
2. 이후 새로운 토큰이 생성될 때마다 **해당 토큰에 대해서만** key 및 value 벡터를 계산합니다.
3. 이전에 계산해둔 key/value 벡터는 **캐시에서 불러와서** 중복 계산을 방지합니다.

아래 표는 각 단계에서의 **계산 및 캐싱 동작**을 요약한 것입니다:

![](/assets/images/posts/573/img_3.png)

이 예시는 짧은 문장을 사용했지만, 텍스트 길이가 길어질수록 **재사용 가능한 key/value 벡터가 많아지고**, 그만큼 **생성 속도도 빨라진다는 점은 직관적으로 이해될 수 있습니다**.

다음 그림은 **세 번째 토큰을 생성할 때** KV 캐시를 사용한 경우와 사용하지 않은 경우를 **비교한 시각화**입니다:

![](/assets/images/posts/573/img_2.jpg)

위쪽 그림(캐시 없음)에서는 각 토큰에 대해 key/value 벡터가 매번 다시 계산되어 **중복 연산**이 발생합니다. 반면 아래쪽 그림(캐시 있음)에서는 **이미 계산된 벡터를 KV 캐시에서 불러와 사용**함으로써 불필요한 재계산을 방지하고 **생성 속도를 향상**시킵니다.

결론적으로, **KV 캐시를 코드로 구현**하고 싶다면 해야 할 일은 단순합니다:

- key 및 value 벡터를 **기존과 동일하게 계산**하되,
- 그 벡터들을 **저장해두었다가**,
- **다음 토큰 생성 시 재사용**할 수 있도록 하면 됩니다.

다음 섹션에서는 이 과정을 실제 코드 예제와 함께 살펴보겠습니다.

### KV 캐시 직접 구현하기

(Implementing a KV Cache from Scratch)

KV 캐시는 여러 가지 방식으로 구현할 수 있지만, **핵심 아이디어는 각 생성 단계에서 새롭게 생성된 토큰에 대해서만 key와 value 텐서를 계산**한다는 점입니다. 저는 **코드 가독성**을 우선시하는 간단한 구현 방식을 선택했습니다. 실제 구현 내용을 확인하는 가장 쉬운 방법은 코드 변경 사항을 **스크롤하면서 직접 보는 것**이라고 생각합니다.

GitHub에 공유한 두 개의 파일은 **KV 캐시를 적용한 버전과 적용하지 않은 버전**의 LLM을 **처음부터 구현한 독립 실행형 Python 스크립트**입니다:

- **gpt\_ch04.py**: 제 책 Build a Large Language Model From Scratch의 3장과 4장에서 가져온 자가 포함(self-contained) 코드로, LLM을 구현하고 간단한 텍스트 생성 함수를 실행합니다.
- **gpt\_with\_kv\_cache.py**: 위와 동일한 코드이지만, **KV 캐시를 구현하기 위한 필수 변경사항이 적용된 버전**입니다.

KV 캐시 관련된 코드 수정 내용을 읽는 방법은 다음 두 가지 중 하나를 선택하실 수 있습니다:

**a.** gpt\_with\_kv\_cache.py 파일을 열고, # NEW로 표시된 섹션을 따라가며 **추가된 코드 부분을 직접 확인**하세요.

![](/assets/images/posts/573/img_3.jpg)

**b.** 두 파일을 **파일 비교(diff) 도구**를 사용하여 비교하면, 어떤 부분이 변경되었는지 확인할 수 있습니다.

![](/assets/images/posts/573/img_4.jpg)

추가로, 구현 세부사항을 요약한 짧은 설명이 **다음 섹션들**에 포함되어 있습니다.

## 1. 캐시 버퍼 등록

MultiHeadAttention 클래스의 생성자 안에서, 각 스텝에서 생성된 key와 value를 누적 저장할 **비영속적(non-persistent) 버퍼** cache\_k와 cache\_v를 등록합니다:

```
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
```

(버퍼에 대해 더 배우고 싶다면 YouTube 영상 [Understanding PyTorch Buffers]를 참고하세요.)

## 2. use\_cache 플래그가 있는 Forward 패스

이제 MultiHeadAttention 클래스의 forward 메서드를 확장하여 use\_cache 인자를 받도록 합니다:

```
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
```

이 코드가 바로 **KV 캐시의 핵심 아이디어인 저장과 재사용(store & reuse)**를 구현한 부분입니다.

- **저장**: self.cache\_k is None 조건문을 통해 캐시를 초기화하고, 이후에는 torch.cat(...)으로 새로운 key/value를 누적 저장합니다.
- **조회**: keys, values = self.cache\_k, self.cache\_v로 저장된 값을 가져옵니다.

## 3. 캐시 초기화

문장을 생성할 때는 각 문장마다 이전 캐시를 **반드시 초기화**해야 합니다. 그렇지 않으면 이전 문장의 key/value들이 남아 **새로운 문장의 query가 잘못된 컨텍스트를 attend**하게 되어 **이상한 출력이 발생**합니다.  
이를 방지하기 위해 MultiHeadAttention 클래스에 reset\_cache 메서드를 추가합니다:

```
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
```

## 4. 모델 전체에서 use\_cache 전파하기

MultiHeadAttention에 변경이 적용되었으니, 이제 GPTModel에도 다음과 같은 변경이 필요합니다:

### a. 현재 위치 추적

```
self.current_pos = 0
```

→ 현재까지 생성한 토큰의 개수를 저장하는 **카운터**입니다. 다음 디코딩 호출 시 이전 위치 이후부터 이어붙이기 위해 필요합니다.

### b. 토큰 위치 계산 및 블록 호출

```
def forward(self, in_idx, use_cache=False):
    # ...

    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )

    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds

    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

→ use\_cache=True인 경우, self.current\_pos부터 시작하여 seq\_len만큼 포지션을 생성하고, 이후 포인터를 증가시켜 **다음 디코딩이 이어질 수 있게** 합니다.

→ TransformerBlock도 다음처럼 use\_cache 인자를 받도록 약간 수정해야 합니다:

```
def forward(self, x, use_cache=False):
    self.att(x, use_cache=use_cache)
```

### c. 모델 전체 캐시 초기화

전체 블록의 캐시를 초기화하는 함수도 GPTModel에 추가합니다:

```
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

## 5. 텍스트 생성 시 캐시 사용하기

이제 모든 구성요소가 준비되었으니, 다음은 KV 캐시를 활용하는 간단한 텍스트 생성 함수입니다:

```
def generate_text_simple_cached(model, idx, max_new_tokens, use_cache=True):
    model.eval()

    ctx_len = model.pos_emb.num_embeddings  # 예: 1024

    if use_cache:
        model.reset_kv_cache()
        with torch.no_grad():
            logits = model(idx[:, -ctx_len:], use_cache=True)

        for _ in range(max_new_tokens):
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)
            with torch.no_grad():
                logits = model(next_idx, use_cache=True)
    else:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(idx[:, -ctx_len:], use_cache=False)
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

### ? 핵심 차이

- use\_cache=True일 경우:
  - 처음에 전체 프롬프트를 넣어 캐시 초기화
  - 이후에는 **새로운 토큰만 입력**하여 효율적 생성
- use\_cache=False일 경우:
  - 매 스텝마다 전체 시퀀스를 다시 입력해야 함 (비효율적)

이 방식으로 **KV 캐시를 직접 구현 및 활용**할 수 있으며, 실제 추론 시 **속도와 VRAM 효율이 크게 향상**됩니다.

## 간단한 성능 비교

KV 캐시에 대한 개념적인 설명을 마친 뒤, **실제 성능이 얼마나 향상되는지**가 가장 궁금한 부분입니다. 이를 확인하기 위해, 앞서 소개한 두 개의 Python 스크립트를 실행해볼 수 있습니다. 각 스크립트는 파라미터 수 1억 2,400만(124M)짜리 작은 LLM을 실행하여, 4개의 토큰으로 된 프롬프트 "Hello, I am"을 입력으로 주고 200개의 새로운 토큰을 생성합니다:

```
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

Apple M4 칩이 탑재된 Mac Mini(CPU 기준)에서의 실행 결과는 다음과 같습니다:

> ? 결과: **KV 캐시를 사용한 경우 약 5배 빠른 속도 향상**을 보여줍니다.

작은 모델(124M 파라미터)에 짧은 200토큰 시퀀스일 뿐인데도 **이 정도 속도 향상**을 얻을 수 있습니다.  
(단, 본 구현은 **코드 가독성을 우선**으로 한 것으로, CUDA나 MPS에 최적화된 속도는 아닙니다. 실제 고속 실행을 원한다면, 텐서를 매번 새로 생성하거나 연결(concat)하는 대신 **미리 메모리를 할당**해야 합니다.)

### 참고: 출력 예시

모델이 아직 학습되지 않았기 때문에, 출력은 다음과 같은 **의미 없는 텍스트(gibberish)**입니다:

```
출력 텍스트: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...
```

이는 **모델이 학습되지 않았기 때문**이며, 다음 장에서는 이 모델을 학습시키고, **학습된 모델에 KV 캐시를 적용하여 실제로 유창한 텍스트를 생성**할 수 있습니다. (참고로, KV 캐시는 **오직 추론 시에만 사용되는 구조**입니다.)

무엇보다 중요한 점은:  
두 파일 gpt\_ch04.py와 gpt\_with\_kv\_cache.py가 **완전히 동일한 결과를 생성**한다는 것입니다.  
이것은 **KV 캐시가 정확하게 구현되었음을 보여주는 신호**입니다. (인덱싱에서 작은 실수만 해도 출력이 달라지기 쉬운 구조이기 때문이죠.)

읽어주셔서 감사합니다! **Ahead of AI** 뉴스레터를 구독하시면 새로운 글을 무료로 받아보실 수 있고, 저의 작업을 후원하실 수도 있습니다.

## KV 캐시의 장점과 단점

시퀀스 길이가 길어질수록, **KV 캐시의 장점과 단점이 더욱 뚜렷하게 드러납니다**:

### ✅ [장점] 계산 효율 향상

- **캐시를 사용하지 않을 경우**, t번째 스텝의 attention은 새로운 query를 이전 t개의 key들과 **모두 비교해야 하므로**, 누적 연산량이 **O(n²)**으로 증가합니다.
- 반면, **KV 캐시를 사용하면** 각 key와 value는 한 번만 계산되고 이후 재사용되므로, per-step 연산량이 **선형 O(n)**으로 줄어듭니다.

### ❌ [단점] 메모리 사용량 선형 증가

- **새로운 토큰이 생성될 때마다** 해당 토큰의 key와 value가 KV 캐시에 **덧붙여지기 때문에**,
- 시퀀스가 길어지고 모델 크기가 커질수록 **KV 캐시의 누적 크기도 커져서**,
- 결국 (특히 GPU의) **메모리를 상당히 많이 또는 과도하게 차지할 수 있습니다**.
- 이런 상황에서는 **KV 캐시를 잘라(truncate)** 일부만 유지하는 방식이 고려될 수 있지만,
  - 이 역시 **구현 복잡도를 증가**시킵니다.
  - 하지만 실제 LLM을 배포(deploy)할 때는 이 정도의 복잡도는 **충분히 감수할 가치가 있는 경우도 많습니다**.

## KV 캐시 구현 최적화

앞서 소개한 KV 캐시의 개념적 구현은 **코드의 가독성과 교육 목적**에 중점을 둔 것이지만,  
**실제 환경에서의 배포**(특히 더 큰 모델과 더 긴 시퀀스를 다룰 때)는 **보다 신중한 최적화**가 필요합니다.

### ✅ 대규모 확장 시 흔한 문제점

- **메모리 단편화 및 반복적인 할당 문제**:  
  앞서 보여준 torch.cat 방식은 **텐서를 반복해서 이어붙이기 때문에**, **메모리 재할당이 빈번히 발생하며 성능 병목을 유발**할 수 있습니다.
- **메모리 사용량의 선형 증가**:  
  적절한 처리를 하지 않으면, **KV 캐시의 크기가 시퀀스 길이에 비례해 폭발적으로 증가**하여 매우 긴 시퀀스에 대해 **비현실적인 메모리 요구량**을 초래합니다.

## ? 최적화 팁

### Tip 1: 메모리 사전 할당 (Pre-allocation)

- torch.cat을 반복적으로 호출하는 대신, **최대 시퀀스 길이에 기반하여 충분한 크기의 텐서를 미리 할당**할 수 있습니다.
- 이렇게 하면 **일관된 메모리 사용**이 가능하고, **불필요한 오버헤드를 줄일 수 있습니다.**

```
# 예시: key와 value를 위한 사전 할당
max_seq_len = 1024  # 예상 최대 시퀀스 길이
cache_k = torch.zeros(
    (batch_size, num_heads, max_seq_len, head_dim), device=device
)
cache_v = torch.zeros(
    (batch_size, num_heads, max_seq_len, head_dim), device=device
)
```

- 이후 추론 중에는 단순히 **지정된 위치(slice)**에 값을 기록하기만 하면 됩니다.

### Tip 2: 슬라이딩 윈도우를 통한 캐시 절단 (Truncate via Sliding Window)

- GPU 메모리 폭증을 방지하기 위해, **슬라이딩 윈도우(sliding window)** 방식의 **동적 캐시 절단**을 사용할 수 있습니다.
- 이는 KV 캐시에서 **최근 window\_size 개의 토큰만 유지**하는 방식입니다:

```
# 슬라이딩 윈도우 캐시 구현 예시
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

## ? 실전 최적화 적용

이러한 최적화는 gpt\_with\_kv\_cache\_optimized.py 파일에서 확인할 수 있습니다.

Apple M4 칩이 탑재된 Mac Mini (CPU 환경)에서, 200개의 토큰을 생성하고, **모델의 최대 컨텍스트 길이만큼 window size를 설정하여 결과의 정합성을 확보**한 후 실행 시간을 비교해본 결과는 다음과 같았습니다:

> ✅ 결과: 최적화된 캐시 덕분에 더 나은 속도를 확보할 수 있었습니다.

단, 이 모델이 너무 작기 때문에, **CUDA 환경에서는 오히려 전송 및 커뮤니케이션 오버헤드가 더 커져**  
**KV 캐시의 속도 이점이 사라지는 경우도** 있었습니다.

## ✅ 결론

KV 캐시는 **복잡도와 메모리 사용이라는 trade-off**를 동반하지만, **실제 배포 환경에서는 이점이 단점을 상쇄할 정도로 효과적**입니다. 이번 글에서는 **코드의 명확성과 가독성**을 우선시했지만, 실용적인 구현에서는 **메모리 사전 할당**, **슬라이딩 윈도우**, 등과 같은 전략적 최적화가 필요합니다. 이러한 측면에서 이 글이 도움이 되었기를 바랍니다.

자유롭게 실험해보시고, 즐거운 코딩 되시길 바랍니다! ?✨
