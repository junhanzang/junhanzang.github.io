---
title: "OpenAI Swarm"
date: 2024-11-26 17:02:50
categories:
  - 인공지능
---

<https://github.com/openai/swarm>

[GitHub - openai/swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Managed by OpenAI Solut](https://github.com/openai/swarm)

<https://www.magicaiprompts.com/blog/openai-swarm-ai-agent-collaboration>

[OpenAI Swarm 이란? AI 에이전트 간 협력을 위한 혁신적인 프레임워크 | 프롬프트해커 대니](https://www.magicaiprompts.com/blog/openai-swarm-ai-agent-collaboration)

**개요**  
Swarm은 에이전트의 협업과 실행을 가볍고, 매우 통제 가능하며, 쉽게 테스트할 수 있도록 하는 데 중점을 둡니다.

Swarm은 두 가지 원시적 추상화, 즉 '에이전트(Agents)'와 '핸드오프(handoffs)'를 통해 이를 실현합니다. '에이전트'는 지시사항과 도구들을 포함하며, 어느 시점에서든지 다른 에이전트로 대화를 넘길 수 있습니다.

이러한 기본 요소들은 도구 및 에이전트 네트워크 간의 복잡한 동적 관계를 표현할 수 있을 만큼 강력하여, 사용자가 가파른 학습 곡선을 피하면서 확장 가능한 현실적인 솔루션을 구축할 수 있도록 합니다.

참고  
Swarm 에이전트는 Assistants API의 '어시스턴트'와는 관련이 없습니다. 두 이름이 비슷한 것은 편의성을 위해서이며, 그 외에는 완전히 관련이 없습니다. Swarm은 전적으로 'Chat Completions API'에 의해 구동되며, 호출 간 상태를 유지하지 않는(stateless) 시스템입니다.

Swarm의 필요성  
Swarm은 경량화되고, 확장 가능하며, 설계상 매우 커스터마이징 가능한 패턴을 탐구합니다. Swarm과 유사한 접근법은 많은 독립적인 기능과 지시사항을 단일 프롬프트에 인코딩하기 어려운 상황에 가장 적합합니다.

Assistants API는 완전 호스팅된 스레드와 내장된 메모리 관리 및 검색 기능을 원하는 개발자에게 좋은 선택입니다. 그러나 Swarm은 멀티 에이전트 오케스트레이션에 대해 배우고자 하는 개발자를 위한 교육 자원입니다. Swarm은 거의 모든 기능을 클라이언트에서 실행하며, Chat Completions API와 마찬가지로 호출 간 상태를 저장하지 않습니다.

예시  
영감을 얻기 위해 /examples를 확인해 보세요! 각 예시의 README에서 자세히 알아볼 수 있습니다.

- basic: 설정, 함수 호출, 핸드오프, 컨텍스트 변수와 같은 기본 사항에 대한 간단한 예제
- triage\_agent: 올바른 에이전트로 넘기기 위해 기본 분류 단계를 설정하는 간단한 예제
- weather\_agent: 함수 호출의 간단한 예제
- airline: 항공사 고객 서비스 요청을 처리하기 위한 다중 에이전트 설정
- support\_bot: 사용자 인터페이스 에이전트와 여러 도구를 가진 도움 센터 에이전트를 포함한 고객 서비스 봇
- personal\_shopper: 판매 및 환불 주문을 돕는 개인 쇼핑 에이전트
