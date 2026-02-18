---
title: "[Your Wish is my Command: Generate, Edit, Rearrange] LLM-enhanced Scene Graph Learning for Household Rearrangement"
date: 2024-12-19 00:52:54
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687607>

[LLM-enhanced Scene Graph Learning for Household Rearrangement | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687607)

### **LLM-Enhanced Scene Graph Learning for Household Rearrangement**

**"LLM-Enhanced Scene Graph Learning"** 논문은 가정에서의 물건 정리 작업을 보다 효율적으로 수행하기 위해 **Scene Graph**와 \*\*Large Language Model (LLM)\*\*을 활용하는 방법을 제시했습니다.

![](/assets/images/posts/420/img.jpg)

---

### **핵심 내용 요약**

1. **Scene Graph와 Affordance Enhanced Graph (AEG)**
   - 기본 **Scene Graph**는 객체 간의 관계와 위치를 나타냅니다.
   - 이 논문에서는 LLM을 활용해 **Affordance Enhanced Graph (AEG)**를 생성하여 객체의 기능적 관계를 강화합니다.
   - AEG는 새로운 **Semantic Edge**와 **Affordance 정보**를 추가하여 객체의 적절한 배치 및 사용 가능성을 분석합니다.
2. **Misplacement Detection**
   - **LLM 기반 Scorer**를 활용해 현재 배치된 물체가 잘못된 위치에 있는지 판단합니다.
   - 객체와 Receptacle(수납 위치)의 적합성을 점수화(0~100)하여 **Misplacement**를 감지합니다.
3. **Object Rearrangement Planning**
   - 잘못 배치된 물체를 올바른 위치로 이동시키기 위해 LLM을 사용해 최적의 Receptacle 후보를 선택합니다.
   - Retrieval-Augmented Generation (RAG) 기법을 활용해 관련 없는 정보를 필터링하고, 적합한 Receptacle만 LLM에 전달해 정확한 배치 계획을 생성합니다.
4. **벤치마크와 성능**
   - Habitat 3.0 Simulator를 사용해 새로운 컨텍스트 기반 데이터셋에서 테스트한 결과, **Misplacement Detection** 및 **Rearrangement Planning**에서 최첨단 성능을 기록했습니다.

---
