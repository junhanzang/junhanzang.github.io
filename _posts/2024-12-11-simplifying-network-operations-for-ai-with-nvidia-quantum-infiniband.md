---
title: "Simplifying Network Operations for AI with NVIDIA Quantum InfiniBand"
date: 2024-12-11 21:54:02
categories:
  - 소식
---

<https://resources.nvidia.com/en-us-accelerated-networking-resource-library/simplifying-network-operations-for-ai-with-nvidia-quantum-infiniband?ncid=em-nurt-707292&mkt_tok=MTU2LU9GTi03NDIAAAGXS2dr_OZeYjhR4ro72w3LgEc0qx6AP72WF02v1cD7CyZxIqFEIkSpqQx0bhV2EJwy3cG3j-zUNGDsgua_-kzwpavKaLZtsNZUrnLalvOM9UnSIA36Lk6a>

[Simplifying Network Operations for AI with NVIDIA Quantum InfiniBand](https://resources.nvidia.com/en-us-accelerated-networking-resource-library/simplifying-network-operations-for-ai-with-nvidia-quantum-infiniband?ncid=em-nurt-707292&mkt_tok=MTU2LU9GTi03NDIAAAGXS2dr_OZeYjhR4ro72w3LgEc0qx6AP72WF02v1cD7CyZxIqFEIkSpqQx0bhV2EJwy3cG3j-zUNGDsgua_-kzwpavKaLZtsNZUrnLalvOM9UnSIA36Lk6a)

기술적인 오해 중 하나는 성능과 복잡성이 직접적으로 연결되어 있다는 것입니다. 즉, 최고 성능의 구현이 가장 구현과 관리가 어려운 것이라는 인식입니다. 그러나 데이터 센터 네트워킹을 고려할 때 이는 사실이 아닙니다.

InfiniBand는 이더넷에 비해 어렵고 생소하게 들릴 수 있지만, 최고 성능을 위해 처음부터 설계되었기 때문에 실제로는 배포와 유지 관리가 더 간단합니다. AI 인프라의 연결성을 고려할 때, **InfiniBand 클러스터 운영 및 유지 관리 가이드**는 완전한 스택의 InfiniBand 네트워크를 최대한 간단하게 설정하고 운영할 수 있도록 돕습니다.

이 포괄적인 가이드는 Day 0(초기 설정), Day 1(구축), Day 2(운영) 네트워크 작업을 간소화하는 데 필요한 주요 단계를 다룹니다. 특히 NVIDIA Unified Fabric Manager(UFM)를 활용하여 초기 프로비저닝과 지속적인 유지 관리 계획을 지원하는 방법을 자세히 설명합니다.

UFM은 광범위한 텔레메트리 및 분석 기능을 갖춘 강력한 도구 세트입니다. 그러나 클러스터 모니터링 및 관리를 위한 기본적인 작업을 시작하는 데는 고급 전제 조건이나 전문 지식이 필요하지 않습니다.

![](/assets/images/posts/373/img.png)

**그림 1. UFM 패브릭 대시보드**

**클러스터 구축 및 운영**  
이 가이드는 다음과 같은 초기 설정 과정을 단계별로 안내합니다:

- UFM의 운영 상태 확인
- 패브릭 상태 보고서 생성 및 토폴로지 검증
- 클러스터 성능 확인

또한, UFM Telemetry를 활용한 혼잡 분석의 기초도 소개합니다.  
UFM의 텔레메트리 및 모니터링 기능은 강력하며, Grafana, Fluentd, Slurm, Zabbix와 같은 도구의 서드파티 플러그인을 사용하여 주요 네트워킹 지표를 캡처하고 선호하는 플랫폼에서 활용할 수 있습니다.

관리자가 클러스터가 초기 건강 상태에 있음을 확인하면, 가이드는 주기적인 유지 관리를 위한 점검 목록과 함께 클러스터 유지 관리 체계를 제안합니다.

### **세부 유지 관리 일정**

**분 단위/지속적인 유지 관리:**

- 문제 해결 목록에 있는 시나리오를 확인하고 해결 지침을 따릅니다.

**주간 유지 관리:**

- 링크 모니터링 주요 지표(UFM 사용자 인터페이스에서 확인 가능)의 트렌드를 모니터링합니다.
- 클러스터 토폴로지 검증 및 패브릭 상태 검증 테스트를 실행합니다.
- HPC-X 소프트웨어 패키지에 포함된 ClusterKit을 사용하여 성능 KPI를 확인합니다.
- UFM에서 캡처한 온도 차이를 검토하여 냉각 시스템이 제대로 작동하는지 확인합니다.

**분기/연간 유지 관리:**

- 최신 펌웨어 및 소프트웨어 릴리스 노트와 검증된 설정을 검토하고 가능하면 업그레이드합니다.
- NVIDIA 네트워킹 지원팀 또는 지정된 NVIDIA 연락 창구를 통해 연간 NVIDIA 네트워크 상태 검토를 실시합니다.

이러한 점검의 상당수는 자동화가 가능하며 API를 통해 구성할 수 있습니다. 가이드는 관련 UFM API 문서에 대한 링크를 제공하여 설정 과정을 간단하고 매끄럽게 만듭니다.

**문제 해결**  
물론, 어떤 시스템도 완벽하지는 않습니다. InfiniBand 클러스터와 같은 잘 운영되는 시스템조차도 예상치 못한 문제를 가끔 겪을 수 있습니다.

하지만 관리자 입장에서, 클러스터 유지 관리 가이드는 **모든 것을 해결할 수 있는 포괄적인 안내서**입니다. 이 가이드에는 자주 발생하는 시나리오와 해결 방법을 설명하는 장이 포함되어 있습니다.

이 섹션은 각 시나리오와 이를 감지하는 방법(UFM 경고 이벤트 ID와 함께) 및 문제 해결을 위한 단계별 지침을 제공합니다. 단순하고 일반적인 오류(예: 불량 포트, 불안정한 링크, 케이블 연결 문제)부터 성능 저하나 낮은 대역폭과 같은 더 복잡한 문제까지 다룹니다.

![](/assets/images/posts/373/img_1.png)

**그림 2. UFM 이벤트 및 알람 대시보드**

### **요약**

네트워크를 구축할 때 성능은 중요한 고려 사항이지만, 성능과 사용 편의성을 상호 배타적으로 생각할 필요는 없습니다.

InfiniBand는 AI를 위해 도입, 배포, 운영하기 쉽습니다. UFM의 강력한 기능을 활용하여, 클러스터 운영 및 유지 관리 가이드는 네트워크 관리자가 알아야 할 모든 정보를 담고 있습니다. 이 가이드는 네트워킹 자격증 교재를 펼치는 것보다 훨씬 간단하며, 40페이지가 채 되지 않습니다.

AI 인프라를 위해 NVIDIA Quantum InfiniBand의 단순함을 선택해보세요.

### **관련 자료**

- **GTC 세션:** 새로운 AI 네트워킹 혁신의 프론티어 진입
- **GTC 세션:** AI를 위한 네트워킹 모범 사례: 클라우드 서비스 제공자의 관점
- **GTC 세션:** AI 진화를 촉진하다: VMware 기반 NVIDIA AI Enterprise를 위한 QCT의 혁신적인 인프라 (QCT 발표)
- **SDK:** NVIDIA Fleet Command
- **SDK:** NVIDIA UFM
- **웨비나:** 혁신을 이끌고 과학 워크로드를 가속화하고 싶으신가요? 네트워크에서 시작하세요.
