---
title: "MONAI Integrates Advanced Agentic Architectures to Establish Multimodal Medical AI Ecosystem"
date: 2025-04-16 17:36:28
categories:
  - Article
tags:
  - monai
---

<https://developer.nvidia.com/blog/monai-integrates-advanced-agentic-architectures-to-establish-multimodal-medical-ai-ecosystem/?ncid=em-even-120973-vt12&mkt_tok=MTU2LU9GTi03NDIAAAGZ0afUn3t6hA6jFxLXibpOT6MQzu96oCnWq1cVxzHRCKU6dKwiA0MPIjvdcBLiSpIRMQksqarBkI3FfUPQZJsPipT40JPNUB8mGsIKu6oLECBok4W3lJ-E>

[MONAI Integrates Advanced Agentic Architectures to Establish Multimodal Medical AI Ecosystem | NVIDIA Technical Blog](https://developer.nvidia.com/blog/monai-integrates-advanced-agentic-architectures-to-establish-multimodal-medical-ai-ecosystem/?ncid=em-even-120973-vt12&mkt_tok=MTU2LU9GTi03NDIAAAGZ0afUn3t6hA6jFxLXibpOT6MQzu96oCnWq1cVxzHRCKU6dKwiA0MPIjvdcBLiSpIRMQksqarBkI3FfUPQZJsPipT40JPNUB8mGsIKu6oLECBok4W3lJ-E)

MONAI, 첨단 에이전트 아키텍처를 통합하여 다중모달 의료 AI 생태계 구축  
2025년 3월 19일  
저자: 마이클 제피르(Michael Zephyr), 몬티 자룩(Monty Zarrouk)

![](/assets/images/posts/539/img.png)

증가하는 의료 데이터의 양과 복잡성, 그리고 질병의 조기 진단과 의료 효율성 향상에 대한 절박한 필요성은 의료 인공지능(AI) 분야에서 전례 없는 발전을 이끌고 있다. 특히 이 분야에서 가장 혁신적인 발전 중 하나는 텍스트, 이미지, 비디오를 동시에 처리할 수 있는 **다중모달(multimodal)** AI 모델이다. 이 모델들은 기존의 단일 모달 시스템(single-modality systems)보다 환자 데이터에 대한 더욱 포괄적인 이해를 제공한다.

의료 영상 분야에서 가장 빠르게 성장하는 오픈소스 프레임워크인 **MONAI**는 현재 임상 워크플로우와 진단 정확도를 혁신적으로 향상시킬 강력한 다중모달 모델을 통합하며 발전하고 있다. 지난 5년간 MONAI는 의료 AI 플랫폼 분야에서 선도적인 위치를 차지하며 의료 영상 AI 연구의 사실상 표준 프레임워크로 자리 잡았다. 현재까지 450만 회 이상 다운로드되었으며, 3,000편 이상의 논문에서 언급될 정도로 광범위하게 활용되고 있다.

본 글에서는 MONAI가 첨단 **에이전트 AI(agentic AI)**, 즉 자율적이고 워크플로우 기반의 추론 방식을 활용하여 영상 데이터 중심의 기존 접근법에서 벗어나 다중모달 생태계로 확장하는 방안을 설명한다. 이 다중모달 생태계는 CT, MRI 등의 의료 영상 데이터뿐 아니라 전자의무기록(EHRs), 임상 문서와 같은 다양한 의료 데이터를 통합함으로써 영상의학, 외과, 병리학 분야의 연구 개발과 혁신을 촉진한다.

### MONAI Multimodal: 의료 데이터 사일로 간의 다리 역할 수행

의료 데이터가 점점 더 다양하고 복잡해짐에 따라, 이질적인 데이터 소스를 통합하는 포괄적 솔루션의 필요성이 그 어느 때보다 커지고 있다. **MONAI Multimodal**은 기존의 영상 분석 중심에서 한발 더 나아가 통합된 연구 생태계로의 전환을 목표로 하는 집중적인 노력의 결과물이다. 이 플랫폼은 CT, MRI, X-ray, 초음파 영상뿐만 아니라 전자의무기록(EHR), 임상 문서, DICOM 표준 데이터, 비디오 스트림, 병리 조직 슬라이드 영상(WSI, Whole Slide Imaging) 등을 포함한 다양한 의료 데이터를 통합하여, 연구자와 개발자들이 다중모달(multimodal) 분석을 수행할 수 있도록 지원한다.

주요 기능 개선 사항은 다음과 같다.

- **에이전트 AI 프레임워크(Agentic AI Framework)**
  - 이미지와 텍스트를 기반으로 자율적 에이전트가 여러 단계를 거쳐 추론 수행
- **의료에 특화된 LLM 및 VLM 모델**
  - 크로스 모달(cross-modal) 데이터 통합을 효과적으로 지원하도록 의료 분야에 맞춤화된 특수화된 언어모델(LLMs)과 시각-언어 모델(VLMs)을 제공
- **다양한 데이터 IO 컴포넌트 통합**
  - **DICOM**: CT, MRI 등의 의료 영상 데이터
  - **EHR**: 구조화 및 비구조화된 임상 데이터
  - **Video**: 수술 과정 영상 및 동적 의료 영상 데이터
  - **WSI**: 대용량·고해상도 병리 이미지
  - **텍스트(Text)**: 임상 기록 및 기타 의료 관련 텍스트 데이터
  - **이미지(Images)**: 병리 슬라이드 또는 정적인 의료 이미지를 위한 PNG, JPEG, BMP 포맷

MONAI Multimodal 플랫폼은 자율적 에이전트를 통한 이미지와 텍스트 기반의 다단계 추론을 가능하게 하는 고급 에이전트 AI를 탑재했으며, 의료 애플리케이션에 특화된 맞춤형 LLM 및 VLM 모델을 통해 서로 다른 모달 간의 데이터 통합을 쉽게 만든다. 이 협력 생태계는 NVIDIA를 비롯하여 주요 연구기관, 의료 조직 및 학술기관과의 협력을 통해 구축되었다. 이러한 통합적 접근은 의료 AI 혁신을 위한 일관되고 재현 가능한 프레임워크를 제공함으로써 연구 개발 속도를 높이고 임상 협력을 강화한다.

RadImageNet의 영상의학 전문의이자 매니징 파트너인 팀 데이어(Tim Deyer) 박사는 “첨단 다중모달 모델을 통해 다양한 데이터 스트림을 통합함으로써 단지 진단 정확성을 향상시키는 것뿐만 아니라 임상 의료진이 환자 데이터와 상호작용하는 방식 자체를 근본적으로 변화시키고 있다”며, “이러한 혁신은 보다 빠르고 신뢰할 수 있는 의료 의사 결정의 길을 열어주고 있다”고 밝혔다.

### 의료 AI 연구를 위한 통합 플랫폼 구축을 위한 MONAI Multimodal 구성요소

MONAI Multimodal 프레임워크는 광범위한 의료 AI 이니셔티브의 일환으로, 모달 간(cross-modal) 추론과 통합을 지원하기 위해 다음과 같은 핵심 구성요소들을 포함하고 있다.

## 에이전트 프레임워크 (Agentic Framework)

에이전트 프레임워크는 이미지와 텍스트 데이터를 인간과 유사한 논리로 통합하여 다단계(multistep) 추론을 수행할 수 있도록 설계된 멀티모달 AI 에이전트를 배포하고 관리하는 참조 아키텍처이다. 이는 에이전트 기반의 커스터마이징 가능한 프로세스를 통해 맞춤형 워크플로우를 지원하며, 시각(vision)과 언어(language) 구성요소 간의 복잡한 통합을 용이하게 해준다.

MONAI의 에이전트 아키텍처는 모듈식(modular) 디자인을 기반으로 의료 AI에서의 모달 간 추론을 가능하게 한다. 이 아키텍처는 방사선학(Radiology Agent Framework), 외과수술(Surgical Agent Framework)과 같은 전문화된 에이전트들을 조율하는 **중앙 오케스트레이션 엔진(Orchestration Engine)**, 일관된 배포를 위한 인터페이스, 그리고 표준화된 출력을 제공하는 추론 및 결정 계층(Reasoning and Decision Layer)을 포함한다. (그림 1 참고)

![](/assets/images/posts/539/img_1.png)

그림 1. MONAI 에이전트 아키텍처 개요

MONAI 에이전트 아키텍처 다이어그램은 의료 영상 AI를 위한 계층적 시스템을 보여준다. 최상단에는 방사선학 에이전트 프레임워크(좌측)와 외과 에이전트 프레임워크(우측)가 위치해 있다. 중앙의 오케스트레이션 엔진은 좌측의 영상 에이전트(Image Agent)와 우측의 수술 에이전트(Surgical Agent)를 조정하며, 두 에이전트 유형 모두 외부 모델의 지원을 받는다. 각 주요 에이전트는 보고서 생성 에이전트(Report Gen Agent), 분할(Segmentation) 등 도메인 특화 하위 에이전트로 연결된다. 작업 흐름은 추론 및 결정 계층을 통해 아래로 흐르며 최종적으로 표준화된 출력 인터페이스(Standardized Output Interface)에 도달한다.

## 기반 모델(Foundation Models) 및 커뮤니티 기여

MONAI Multimodal 플랫폼은 최첨단 AI 모델을 기반으로 구축되었으며, NVIDIA 주도의 프레임워크와 커뮤니티 파트너의 혁신적 기여가 통합되어 제공된다.

### NVIDIA 주도 프레임워크

NVIDIA가 이끄는 프레임워크는 다음과 같은 구성요소를 포함한다.

## 방사선학 에이전트 프레임워크(Radiology Agent Framework: Multimodal Radiology Agentic Framework)

방사선학 에이전트 프레임워크는 의료 영상과 텍스트 데이터를 결합하여 방사선 전문의의 진단 및 판독 업무를 지원하는 에이전트 기반 프레임워크이다.

**주요 특징:**

- **3D CT/MR 영상**과 환자 **전자 의무 기록(EHR)** 데이터 통합
- 포괄적인 분석을 위한 **대규모 언어 모델(LLMs)** 및 **시각-언어 모델(VLMs)** 활용
- 필요 시 특화된 전문가 모델(**VISTA-3D**, **MONAI BraTS**, **TorchXRayVision**) 접근 가능
- **Meta Llama 3** 기반으로 구축
- 상세한 결과를 위해 다양한 데이터 스트림 처리
- 복잡한 문제를 관리 가능한 단계로 나누어 다단계 추론 수행

![](/assets/images/posts/539/img_2.png)

그림 2. 방사선학 에이전트 프레임워크

**방사선학 AI 에이전트 프레임워크 개요 (그림 2):**

방사선학 AI 에이전트 프레임워크 다이어그램은 **VILA-M3 모델**을 중심으로 하는 다중모달 워크플로우를 설명한다. 사용자는 이미지 토큰과 텍스트 토큰을 입력하고 결과를 받을 수 있다. 이러한 입력 토큰은 중앙의 VILA-M3 에이전트(녹색 육각형)에 전달되며, 이 에이전트는 **VISTA-3D**, **TorchXRay**, **BRATS MRI**와 같은 전문 의료 영상 모델과 연결된다. 이를 통해 VILA-M3 에이전트는 ▲시각 질의응답(VQA), ▲보고서 생성(Report Generation), ▲영상 분할 및 분류(Segmentation and Classification), ▲추론(Reasoning) 등 네 가지 핵심 기능을 제공한다.

## 외과수술 에이전트 프레임워크(Surgical Agent Framework: Multimodal Surgical Agentic Framework)

외과수술 에이전트 프레임워크는 시각-언어 모델(VLM)과 검색 기반 생성(RAG, Retrieval-Augmented Generation)을 결합하여 외과 수술 환경에 특화된 다중 에이전트 시스템이다. 이 시스템은 수술의 전체 워크플로우를 처음부터 끝까지(end-to-end) 지원한다.

### 핵심 특징:

- **Whisper**를 이용한 **실시간 음성 전사 기능**
- 쿼리 라우팅, 질의응답(Q&A), 문서화(documentation), 주석(annotation), 보고서 작성(reporting)을 위한 **특화된 에이전트**
- 영상 분석을 위한 **컴퓨터 비전 통합**
- 선택적 **음성 응답 기능** 지원
- 환자 맞춤형 **사전 수술 데이터**, **임상 의사의 선호도**, **의료기기 정보** 통합
- 수술 중 데이터의 **실시간 처리**
- 수술의 전 과정(훈련, 계획, 가이드, 분석)을 지원하는 **디지털 어시스턴트**로서의 기능 제공

![](/assets/images/posts/539/img_3.png)

**그림 3. 외과수술 에이전트 프레임워크**

### 외과수술 AI 에이전트 프레임워크 개요 (그림 3):

외과수술 AI 에이전트 프레임워크 다이어그램은 수술 보조를 위한 워크플로우를 나타낸다. 사용자는 음성 쿼리(Voice Query)를 통해 입력할 수 있으며, 이는 **RIVA ASR/STT**를 통해 처리되거나, 텍스트 입력(Text Input)을 통해 입력할 수도 있다. 이러한 입력은 **선택 에이전트(Selector Agent)**에 전달되며, 선택 에이전트는 다시 **수술 어시스턴트**로 데이터를 전송한다.

수술 어시스턴트는 이미지 토큰과 텍스트 토큰을 처리하고, 수술 중 보조 채팅(Intra-Op Surgical Assistant Chat), 수술 기록(Surgical Note Taker), 수술 후 요약(Post-Op Summarization Agent), LMM 채팅 에이전트(LMM Chat Agent)와 같은 특수한 하위 기능들과 연계되어 있다. 최종 결과물은 **RIVA TTS NIM**을 통해 사용자에게 텍스트 출력(Text Output) 형태로 전달된다.

## 커뮤니티 주도 파트너 모델(Community-led partner models)

커뮤니티 주도의 파트너 모델은 다음과 같다.

### 1. **RadViLLA**

**RadViLLA**는 **Rad Image Net**, 마운트 시나이 아이칸 의과대학의 생명의학공학 및 영상 연구소(**BioMedical Engineering and Imaging Institute**), 그리고 **NVIDIA**가 공동 개발한 방사선학 특화 3D 시각-언어 모델(VLM)이다. RadViLLA는 흉부, 복부, 골반 등 자주 촬영되는 인체 부위에 대한 임상 질의응답에 특화되어 있으며, 75,000건 이상의 3D CT 영상과 100만 개 이상의 시각적 질의응답 데이터 세트를 통해 훈련되었다.

RadViLLA는 3D CT 스캔 데이터를 텍스트 데이터와 통합하는 새로운 **2단계 훈련 전략**을 사용하여, 임상 질의에 자율적으로 응답하며 다양한 데이터셋에서 **F1 점수**와 **균형 정확도(balanced accuracy)** 면에서 뛰어난 성능을 입증했다.

### 2. **CT-CHAT**

**CT-CHAT**은 **취리히 대학교(University of Zurich)**에서 개발한 첨단 시각-언어 챗 모델로, 3D 흉부 CT 영상의 해석 및 진단 능력 향상에 특화되어 있다. CT-CHAT은 **CT-CLIP** 프레임워크와 **CT-RATE**를 기반으로 한 시각 질의응답(VQA) 데이터 세트를 활용하여 개발되었다.

CT-CHAT은 CT-RATE로부터 얻은 **270만 개 이상의 질의응답 데이터 쌍**을 통해 훈련되었으며, 3D 공간 정보를 효과적으로 활용하여 2D 기반 모델보다 뛰어난 성능을 발휘한다. 이 모델은 CT-CLIP의 시각 인코더와 사전 훈련된 대규모 언어 모델을 결합하여 영상 판독 시간을 줄이고 정확한 진단 정보를 제공하여, 의료 영상 진단 분야에서 강력한 성능을 보여준다.

## Hugging Face 통합(Hugging Face integration)

MONAI Multimodal 플랫폼은 Hugging Face 연구 인프라와의 표준화된 파이프라인을 통해 다음과 같은 기능을 제공한다.

- 연구 목적을 위한 **모델 공유(model sharing)**
- 신규 모델의 신속한 **통합 및 활용(integration of new models)**
- 광범위한 연구 생태계 내에서의 **커뮤니티 참여 확대(broader participation)**를 촉진

## 커뮤니티 통합(Community integration)

모델 공유, 검증 및 협력적 개발을 위한 인프라를 제공한다.

- **표준화된 모델 카드(model cards) 및 에이전트 워크플로우(agent workflows)** 제공
- 커뮤니티 내의 **지식 교류 및 우수 사례(best practices) 공유**
- 협력적 연구를 위한 기반 구축

![](/assets/images/posts/539/img_4.png)

**그림 4. MONAI Multimodal 커뮤니티 아키텍처 및 통합 생태계**

### MONAI Multimodal 커뮤니티 아키텍처 개요(그림 4)

MONAI Multimodal 커뮤니티 아키텍처 다이어그램은 3단계로 구성된 생태계를 제시한다.

- **최상위 계층(Integration Infrastructure)**
  - MONAI 모델
  - Hugging Face 파이프라인 API
- **중간 계층(Community Contributions)**
  - 참조 모델 카드(Reference Model Cards)
  - 에이전트 코드(Agent Code)
  - 활용 사례 및 예시(Examples)
- **하위 계층(Institutional and Community Contributions)**
  - NVIDIA, 취리히 대학교, RadImageNet 등 기관 및 커뮤니티 참여 모델
  - **모델 및 에이전트 생태계(Model and Agent Ecosystem)** 포함:
    - 커뮤니티 모델(Community Models)
    - 비전 모델(Vision Models)
    - 에이전트 워크플로우(Agent Workflows)

## MONAI Multimodal과 함께하는 의료 AI의 미래 구축(Build the future of medical AI with MONAI Multimodal)

MONAI Multimodal은 의료 영상 AI 분야의 대표적인 오픈소스 플랫폼인 MONAI의 다음 단계 진화를 보여준다. 기존 영상 분석을 넘어서서 방사선학, 병리학, 임상 기록 및 전자의무기록(EHR) 등 다양한 의료 데이터를 포괄적으로 통합한다.

NVIDIA 주도의 첨단 프레임워크와 파트너들의 협력적 생태계를 통해, MONAI Multimodal은 특화된 에이전트 아키텍처를 바탕으로 향상된 추론 역량을 제공한다. 데이터 사일로(silo)를 허물고 원활한 크로스모달 분석을 가능하게 함으로써, 의료 전반에 걸쳐 주요 과제 해결을 돕고, 연구 혁신과 임상 적용 속도를 높인다.

다양한 데이터 소스를 통합하고 최첨단 모델을 활용하여, MONAI Multimodal은 의료 분야를 혁신하며, 임상의, 연구자 및 개발자들이 의료 영상과 진단 정확성에서 획기적인 성과를 달성하도록 돕는다.

우리는 단순히 소프트웨어를 개발하는 것이 아니라, 전 세계 연구자와 의료진, 환자 모두에게 혜택을 줄 수 있는 의료 AI 혁신 생태계를 함께 만들어 나가고 있다. 지금 바로 MONAI를 통해 의료 AI 혁신 여정에 동참하라.

NVIDIA GTC 2025에서 다음과 같은 관련 세션에 참여해 더 많은 내용을 확인할 수 있다.

- **디지털 및 물리적 AI가 의학의 새로운 장을 열다** [S71353]  
  (Digital and Physical AI Helps Write a New Chapter in Medicine)
- **의료의 미래를 가속하다: 영상 진단, 디지털 헬스 및 그 이상을 위한 AI 혁신** [S72493]  
  (Accelerate the Future of Healthcare: AI-Powered Innovations for Imaging, Digital Health, and Beyond)

---

기존 MONAI와 MONAI Multimodal의 차이점은 다음과 같은 부분에서 명확하게 구분할 수 있습니다.

### 1. **데이터 처리 범위**

- **기존 MONAI**  
  주로 의료 영상(CT, MRI 등)의 처리와 분석을 중심으로 하며, 단일 모달(주로 이미지)에 초점을 맞춤.
- **MONAI Multimodal**  
  의료 영상뿐 아니라 임상 기록(EHR), 텍스트 데이터, 병리 영상, 비디오 스트림 등 여러 형태의 의료 데이터를 통합하고 분석하여 다중모달(multimodal) 환경을 지원.

### 2. **추론 방식과 아키텍처**

- **기존 MONAI**  
  주로 단일 모달의 영상 분석과 간단한 파이프라인 기반의 작업 흐름을 통해 분석을 수행.
- **MONAI Multimodal**  
  에이전트 기반(agentic) 아키텍처를 활용해 복잡한 데이터 간의 상호작용과 다단계 추론(multistep reasoning)을 가능하게 하며, 인간과 유사한 논리로 데이터를 분석하는 자율적 에이전트 프레임워크를 제공.

### 3. **활용하는 AI 모델의 종류**

- **기존 MONAI**  
  주로 CNN 기반의 영상 처리 모델 및 단일 모달 모델 중심으로 구성됨.
- **MONAI Multimodal**  
  대규모 언어 모델(LLMs), 시각-언어 모델(VLMs) 등 다양한 AI 모델을 활용하여 이미지와 텍스트를 통합 분석하고, 특화된 모델(RadViLLA, CT-CHAT 등)을 포함해 보다 포괄적이고 강력한 분석 역량을 제공.

### 4. **에이전트 기반 협력적 생태계 구축**

- **기존 MONAI**  
  이미지 분석 중심의 프레임워크로서 의료 영상 분석 및 연구에 주로 활용되며, 협력 모델이 상대적으로 제한적.
- **MONAI Multimodal**  
  NVIDIA와 연구기관 및 커뮤니티 파트너가 적극 참여하는 개방형 협력 생태계를 구축해 보다 폭넓은 데이터와 AI 모델을 공유하고 협력 개발 가능.

### 5. **커뮤니티 통합과 공유 기능**

- **기존 MONAI**  
  이미지 분석 연구 커뮤니티 중심으로 공유되며, 데이터 형태나 분석 모델이 영상 위주로 한정.
- **MONAI Multimodal**  
  Hugging Face 등의 플랫폼과 통합되어 표준화된 모델 카드, 에이전트 워크플로우, 코드 공유 등 더 넓고 체계적인 공동 연구 및 개발 환경을 지원.
