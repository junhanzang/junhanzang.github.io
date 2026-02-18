---
title: "Qwen3: Think Deeper, Act Faster"
date: 2025-04-30 17:36:07
categories:
  - Article
tags:
  - qwen3
---

<https://github.com/QwenLM/Qwen3>

[GitHub - QwenLM/Qwen3: Qwen3 is the large language model series developed by Qwen team, Alibaba Cloud.](https://github.com/QwenLM/Qwen3)

<https://huggingface.co/Qwen/Qwen3-235B-A22B>

[Qwen/Qwen3-235B-A22B · Hugging Face](https://huggingface.co/Qwen/Qwen3-235B-A22B)

<https://qwenlm.github.io/blog/qwen3/?_bhlid=09d1346eae3fed59bfb47ff2c847cff2d1f6714f>

[Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/?_bhlid=09d1346eae3fed59bfb47ff2c847cff2d1f6714f)

**소개**

오늘 우리는 Qwen 대형 언어 모델 계열의 최신 버전인 **Qwen3**의 출시를 기쁜 마음으로 발표합니다. 우리의 대표 모델 **Qwen3-235B-A22B**는 코딩, 수학, 일반적 능력 등 다양한 벤치마크 평가에서 DeepSeek-R1, o1, o3-mini, Grok-3, Gemini-2.5-Pro와 같은 최상위 모델들과 비교해도 경쟁력 있는 성능을 보여줍니다. 또한 소형 MoE(Mixture of Experts) 모델인 **Qwen3-30B-A3B**는 활성화된 파라미터 수가 10배 많은 **QwQ-32B**보다 우수한 성능을 보이며, **Qwen3-4B**와 같은 소형 모델조차도 **Qwen2.5-72B-Instruct**에 필적하는 성능을 자랑합니다.

![](/assets/images/posts/552/img.jpg)

![](/assets/images/posts/552/img_1.jpg)

우리는 두 개의 MoE(Mixture of Experts) 모델을 오픈 웨이트(open-weight)로 공개합니다. 하나는 **총 2,350억 개의 파라미터**와 **활성화된 220억 개의 파라미터**를 가진 대형 모델 **Qwen3-235B-A22B**, 다른 하나는 **총 300억 개의 파라미터**와 **활성화된 30억 개의 파라미터**를 가진 소형 MoE 모델 **Qwen3-30B-A3B**입니다.

추가로, **Qwen3-32B**, **Qwen3-14B**, **Qwen3-8B**, **Qwen3-4B**, **Qwen3-1.7B**, **Qwen3-0.6B** 등 총 6개의 덴스(dense) 모델도 **Apache 2.0 라이선스** 하에 오픈 웨이트로 공개됩니다.

![](/assets/images/posts/552/img.png)

![](/assets/images/posts/552/img_1.png)

**Qwen3-30B-A3B**와 같은 후속 학습(post-trained) 모델은, 그에 대응하는 사전 학습(pre-trained) 버전(예: **Qwen3-30B-A3B-Base**)과 함께 Hugging Face, ModelScope, Kaggle과 같은 플랫폼에서 지금 바로 이용할 수 있습니다. 배포 용도로는 **SGLang**이나 **vLLM**과 같은 프레임워크 사용을 권장합니다. 로컬 환경에서는 **Ollama**, **LMStudio**, **MLX**, **llama.cpp**, **KTransformers** 등의 도구가 매우 유용하게 활용될 수 있습니다. 이러한 다양한 옵션은 사용자들이 연구, 개발, 혹은 프로덕션 환경에서 Qwen3을 손쉽게 통합하고 활용할 수 있도록 도와줍니다.

우리는 **Qwen3의 공개 및 오픈소스화**가 대형 파운데이션 모델에 대한 연구와 개발을 크게 촉진할 것이라 믿습니다. 우리의 목표는 전 세계의 연구자, 개발자, 그리고 조직들이 이 최첨단 모델들을 기반으로 **혁신적인 솔루션**을 구축할 수 있도록 지원하는 것입니다.

Qwen3을 직접 사용해보고 싶다면, **Qwen Chat 웹사이트(chat.qwen.ai)** 또는 **모바일 앱**을 통해 자유롭게 체험해보세요!

**주요 특징**

### 하이브리드 사고 모드(Hybrid Thinking Modes)

Qwen3 모델은 문제 해결을 위한 **하이브리드 접근 방식**을 도입합니다. 다음 두 가지 모드를 지원합니다:

- **사고 모드(Thinking Mode)**: 이 모드에서는 모델이 최종 답변을 내리기 전에 단계별로 사고하며 시간을 들여 추론합니다. 복잡하고 심층적인 사고가 필요한 문제에 적합합니다.
- **비사고 모드(Non-Thinking Mode)**: 이 모드에서는 모델이 빠르고 거의 즉각적인 응답을 제공합니다. 속도가 중요하고 복잡하지 않은 간단한 질문에 적합합니다.

이러한 유연성 덕분에 사용자는 작업의 성격에 따라 모델이 “얼마나 사고할지”를 직접 제어할 수 있습니다. 예를 들어, 난이도 높은 문제는 충분한 사고 시간을 확보해 처리할 수 있고, 쉬운 문제는 지체 없이 바로 답변을 받을 수 있습니다.

특히, 이 두 모드를 통합함으로써 모델은 **안정적이고 효율적인 사고 예산(thinking budget)** 제어 능력이 크게 향상되었습니다. 앞서 보여준 바와 같이 Qwen3은 할당된 계산적 추론 예산에 비례해 **확장 가능하고 부드러운 성능 향상**을 보여줍니다. 이 설계는 사용자가 작업별로 예산을 손쉽게 설정할 수 있게 하여, **비용 효율성과 추론 품질 사이의 균형을 최적화**하는 데 기여합니다.

![](/assets/images/posts/552/img_2.png)

**다국어 지원 (Multilingual Support)**

Qwen3 모델은 **119개의 언어 및 방언**을 지원합니다. 이와 같은 광범위한 다국어 지원 능력은 국제적 응용 가능성을 크게 확장시키며, 전 세계 사용자들이 Qwen3의 강력한 성능을 활용할 수 있도록 합니다.

![](/assets/images/posts/552/img_3.png)

이처럼 Qwen3은 다양한 언어 환경에 적응할 수 있는 글로벌 모델로, **언어 장벽을 넘는 AI 활용**을 가능하게 합니다.

**향상된 에이전트 능력 (Improved Agentic Capabilities)**

우리는 Qwen3 모델의 **코딩 능력**과 **에이전트(agentic) 기능**을 최적화했으며, **MCP(Multi-turn Code Planning)**에 대한 지원도 더욱 강화했습니다. 아래에 Qwen3이 어떻게 사고하고 환경과 상호작용하는지를 보여주는 예시들을 제공합니다.

**사전 학습 (Pre-training)**

Qwen3의 사전 학습에서는 **Qwen2.5에 비해 데이터셋 규모가 크게 확장**되었습니다. Qwen2.5는 **18조 토큰**으로 사전 학습된 반면, Qwen3는 그 **두 배에 가까운 약 36조 토큰**을 사용하였으며, 이는 **119개의 언어와 방언**을 포함합니다. 이 대규모 데이터셋을 구축하기 위해, 웹 데이터뿐만 아니라 PDF 형식의 문서로부터도 데이터를 수집했습니다. 이러한 문서에서 텍스트를 추출하는 데에는 **Qwen2.5-VL**을 사용하고, 추출된 내용의 품질 향상을 위해 **Qwen2.5**를 활용했습니다. 수학 및 코딩 관련 데이터의 양을 늘리기 위해서는 **Qwen2.5-Math**와 **Qwen2.5-Coder**를 활용하여 **교과서, 질의응답 쌍, 코드 스니펫** 등 다양한 **합성 데이터(synthetic data)**를 생성하였습니다.

사전 학습 과정은 다음과 같은 **3단계(S1~S3)**로 구성되어 있습니다:

- **1단계 (S1)**: 모델은 **약 30조 토큰**으로 사전 학습되었으며, **문맥 길이(context length)**는 **4K 토큰**으로 설정되었습니다. 이 단계에서는 모델이 기본적인 언어 능력과 일반 지식을 습득하게 됩니다.
- **2단계 (S2)**: **STEM, 코딩, 추론**과 같은 지식 집약적인 데이터를 비중 있게 반영하여 데이터셋을 개선한 후, **추가로 5조 토큰**으로 사전 학습을 수행했습니다.
- **3단계 (S3)**: **고품질 장문 데이터**를 사용하여 문맥 길이를 **32K 토큰**으로 확장함으로써, 모델이 긴 입력도 효과적으로 처리할 수 있도록 했습니다.

![](/assets/images/posts/552/img_2.jpg)

모델 아키텍처의 진보, 학습 데이터의 확장, 더 효과적인 학습 기법의 적용 덕분에, **Qwen3 덴스(dense) 베이스 모델의 전체적인 성능은 Qwen2.5 베이스 모델보다 적은 파라미터 수로도 동등한 성능**을 보입니다. 예를 들어, **Qwen3-1.7B/4B/8B/14B/32B-Base** 모델은 각각 **Qwen2.5-3B/7B/14B/32B/72B-Base** 모델과 비슷한 성능을 발휘합니다. 특히 **STEM, 코딩, 추론 분야**에서는 **Qwen3 덴스 베이스 모델이 오히려 더 큰 Qwen2.5 모델을 능가**하는 결과를 보이기도 합니다. 또한 **Qwen3-MoE 베이스 모델**은 활성화된 파라미터 수가 **Qwen2.5 덴스 베이스 모델의 10%에 불과하면서도 유사한 성능**을 보이며, 이는 학습 비용과 추론 비용 모두에서 **상당한 절감 효과**를 제공합니다.

**후속 학습 (Post-training)**

단계별 추론과 빠른 응답을 모두 수행할 수 있는 **하이브리드 모델**을 개발하기 위해, 우리는 **4단계의 학습 파이프라인**을 구현했습니다. 이 파이프라인은 다음과 같은 과정으로 구성됩니다:

1. **긴 Chain-of-Thought(CoT) 콜드 스타트**,
2. **추론 기반 강화 학습(RL)**,
3. **사고 모드 융합**,
4. **일반 목적 강화 학습(RL)**.

- **1단계**에서는 수학, 코딩, 논리 추론, STEM 문제 등 다양한 작업과 도메인을 포괄하는 **장문 CoT 데이터**를 활용해 모델을 파인튜닝하였습니다. 이 단계의 목표는 모델에 **기본적인 추론 능력**을 부여하는 것이었습니다.
- **2단계**에서는 **계산 자원을 확장**하고, **규칙 기반 보상 함수**를 활용하여 **탐색(exploration)과 활용(exploitation)** 능력을 높이는 방식으로 **강화 학습**을 수행했습니다.
- **3단계**에서는 **장문 CoT 데이터**와 **일반적인 instruction tuning 데이터**를 혼합해 모델을 다시 파인튜닝함으로써, **사고(thinking) 능력을 가진 모델에 비사고(non-thinking) 능력**을 통합했습니다. 이 데이터는 **2단계에서 강화된 사고 모델**이 생성한 것이며, 이를 통해 **추론 능력과 빠른 응답 능력이 자연스럽게 결합**되도록 하였습니다.
- **마지막 4단계**에서는 **20개 이상의 일반 도메인 작업**을 대상으로 **강화 학습**을 적용하여 모델의 **일반적 능력**을 더욱 강화하고 **바람직하지 않은 행동을 수정**하였습니다. 이 작업에는 **지시 따르기(instruction following)**, **형식 준수(format following)**, **에이전트 능력(agent capabilities)** 등이 포함됩니다.

**Qwen3으로 개발하기**

아래는 **Qwen3**을 다양한 프레임워크에서 사용하는 방법에 대한 간단한 가이드입니다. 먼저, Hugging Face Transformers에서 **Qwen3-30B-A3B**를 사용하는 기본 예제를 소개합니다:

```
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 입력 프롬프트 준비
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 사고(thinking) 모드 활성화 여부. 기본값은 True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 텍스트 생성
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# 사고 내용(thinking content) 분리
try:
    # 토큰 151668 (</think>) 기준으로 분리
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

**사고(thinking) 모드를 비활성화**하려면 enable\_thinking 인자를 아래와 같이 설정하면 됩니다:

```
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 사고 모드 비활성화. 기본값은 True.
)
```

### **배포용 예시**

**OpenAI 호환 API 엔드포인트**를 만들고자 할 경우, 아래와 같이 sglang>=0.4.6.post1 또는 vllm>=0.8.4를 사용할 수 있습니다:

- **SGLang:**

```
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --reasoning-parser qwen3
```

- **vLLM:**

```
vllm serve Qwen/Qwen3-30B-A3B --enable-reasoning --reasoning-parser deepseek_r1
```

### **로컬 개발용 예시**

로컬에서 Qwen3을 사용하고 싶다면, 다음과 같은 방법이 있습니다:

- **Ollama**

```
ollama run qwen3:30b-a3b
```

- **LMStudio**, **llama.cpp**, **ktransformers** 등을 통해 직접 구축하여 사용할 수도 있습니다.

**고급 활용 (Advanced Usages)**

Qwen3은 enable\_thinking=True로 설정된 경우, **모델의 사고 모드(thinking mode)**를 동적으로 제어할 수 있는 **소프트 스위치 메커니즘(soft switch mechanism)**을 제공합니다. 구체적으로, **/think** 및 **/no\_think** 명령어를 사용자 프롬프트나 시스템 메시지에 추가하면, **다중 턴 대화(multi-turn conversation)**에서 각 턴마다 모델의 사고 모드를 전환할 수 있습니다. 모델은 대화 흐름 중 **가장 최근의 지시**를 따릅니다.

**멀티턴 대화 예시:**

```
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-30B-A3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 대화 히스토리 업데이트
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# 예시 실행
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # 첫 번째 입력 (/think 또는 /no_think 없이 기본적으로 사고 모드 활성화)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # 두 번째 입력: /no_think 추가
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    # 세 번째 입력: /think 추가
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
```

이 기능을 통해 **대화 흐름 중에 사고 수준을 정밀하게 조절**할 수 있으며, 질문의 난이도나 응답 속도에 따라 최적의 모드를 선택할 수 있습니다.

**에이전트 활용 (Agentic Usages)**

Qwen3는 **툴 호출(tool calling)** 기능에서 뛰어난 성능을 보입니다.  
이러한 **에이전트 능력(agentic ability)**을 최대한 활용하려면 **Qwen-Agent**의 사용을 권장합니다.  
Qwen-Agent는 **내부적으로 툴 호출 템플릿과 파서(parser)**를 내장하고 있어, 복잡한 코드를 작성할 필요 없이 손쉽게 도구 연동 기능을 구현할 수 있습니다.

사용 가능한 도구는 다음 중 하나의 방식으로 정의할 수 있습니다:

- **MCP 설정 파일(MCP configuration file)**을 사용
- Qwen-Agent가 제공하는 **내장 도구(integrated tool)** 사용
- 직접 **외부 도구를 연동**하여 사용

**예시 코드:**

```
from qwen_agent.agents import Assistant

# LLM 설정
llm_cfg = {
    'model': 'Qwen3-30B-A3B',

    # Alibaba Model Studio에서 제공하는 endpoint 사용 시:
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # OpenAI API와 호환되는 커스텀 endpoint 사용 시:
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # 기타 설정:
    # 'generate_cfg': {
    #     'thought_in_content': True,  # <think> 태그를 포함한 응답을 하나의 문자열로 받을지 여부
    # },
}

# 사용할 도구 정의
tools = [
    {
        'mcpServers': {  # MCP 설정 파일 지정
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            'fetch': {
                'command': 'uvx',
                'args': ['mcp-server-fetch']
            }
        }
    },
    'code_interpreter',  # 내장된 도구
]

# 에이전트 정의
bot = Assistant(llm=llm_cfg, function_list=tools)

# 스트리밍 방식으로 대화 수행
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
print(responses)
```

**Qwen의 친구들 (Friends of Qwen)**  
수많은 친구들의 응원과 지지에 깊이 감사드립니다.  
**Qwen은 친구들이 없다면 존재할 수 없습니다!**  
더 많은 사람들과 조직이 커뮤니티에 참여해 함께 성장해 가기를 진심으로 환영합니다!

![](/assets/images/posts/552/img_4.png)

**향후 계획 (Future Work)**

Qwen3는 **AGI(범용 인공지능)**와 **ASI(초지능)**을 향한 여정에서 **중요한 이정표**를 의미합니다. 우리는 사전 학습(pretraining)과 강화 학습(RL)을 대규모로 확장함으로써 더 높은 수준의 지능을 달성했으며, 사고(thinking) 모드와 비사고(non-thinking) 모드를 자연스럽게 통합해 사용자에게 **사고 예산(thinking budget)**을 유연하게 제어할 수 있는 기능을 제공했습니다. 또한 **119개 언어에 대한 폭넓은 지원**을 통해 글로벌 접근성도 크게 향상시켰습니다.

앞으로 우리는 여러 측면에서 모델을 더욱 발전시킬 계획입니다.  
그 핵심 방향은 다음과 같습니다:

- 모델 아키텍처와 학습 방법론의 정교화
- 데이터 규모 확장
- 모델 크기 증가
- 문맥 길이(context length) 확장
- 멀티모달 능력 강화
- 장기 추론(long-horizon reasoning)을 위한 환경 기반 피드백이 포함된 강화 학습 고도화

우리는 지금이 **모델을 훈련하는 시대에서, 에이전트를 훈련하는 시대로 전환되는 시점**이라고 믿습니다.  
다가올 다음 버전은 **모두의 작업과 일상에 실질적인 변화를 가져다줄 의미 있는 진보**를 약속합니다.
