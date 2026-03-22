# 논문 번역 스킬

arxiv 논문을 한국어로 완전 번역하여 블로그 포스트로 작성한다.

## 사용법
`/translate-paper <arxiv_html_url>`

## 입력
$ARGUMENTS

## 절차

반드시 아래 4단계를 순서대로 수행한다. 단계를 건너뛰지 않는다.

### 1단계: 목차(TOC) 추출

WebFetch로 논문 URL을 가져와서 **전체 섹션/서브섹션 목록**을 추출한다.

```
프롬프트: "List EVERY section and subsection heading in this paper in order.
Include section numbers, appendices, and supplementary material.
I need the complete table of contents."
```

추출된 목차를 기록하고, 이것이 완성도 체크리스트가 된다.

### 2단계: 섹션별 개별 Fetch

목차의 각 섹션을 **개별적으로** WebFetch한다. 한 번에 2~3개 섹션씩 묶어서 가져온다.
절대로 한 번의 fetch로 전체 논문을 요약하려 하지 않는다.

각 fetch에서 다음 프롬프트 패턴을 사용한다:
```
"Extract the COMPLETE text of Section X (제목).
Include every paragraph, every formula, every detail.
Do NOT summarize - return the raw text word-for-word."
```

**필수 fetch 그룹:**
- Abstract + Introduction
- Methodology (각 서브섹션 별도)
- Experiments + Results + Discussion
- Ablation Study
- Related Work + Conclusion + Limitations
- Appendices (있는 경우 - 프롬프트 템플릿, 하이퍼파라미터, 구현 세부사항)

### 3단계: 번역 + 다이어그램 작성

_posts/ 디렉토리에 Jekyll 포스트를 작성한다.

**포스트 포맷:**
```yaml
---
title: "한국어 제목"
date: YYYY-MM-DD HH:MM:SS
categories:
  - 인공지능
tags:
  - 관련 태그들
  - 논문 리뷰
---
```

**번역 규칙:**
- 모든 섹션을 빠짐없이 번역 (목차 기준)
- 수식은 MathJax (`$$...$$`, `$...$`)로 렌더링
- 모든 수식 아래에 `> **직관적 이해**: ...` 추가
- 인라인 수식의 언더스코어는 `\_`로 이스케이프
- 수식 블록 전후 빈 줄 필수

**다이어그램 규칙 (핵심):**
- 아키텍처/파이프라인 → Mermaid flowchart (`<div class="mermaid">`)
- 데이터 흐름/차원 변화 → Mermaid diagram
- 수치 비교/추세 → ASCII 차트
- 비교/대조 → Mermaid subgraph 나란히 배치
- Mermaid 테마: dark

**반드시 포함할 내용:**
- 저자, 소속, 논문 링크
- 모든 테이블 데이터 (실험 결과, 하이퍼파라미터 등)
- 부록의 프롬프트 템플릿, 훈련 설정, 구현 세부사항

### 4단계: 완성도 검증

1단계에서 추출한 목차와 작성된 포스트를 대조한다.

**체크리스트:**
- [ ] 모든 섹션이 번역되었는가?
- [ ] 모든 수식이 포함되었는가?
- [ ] 모든 테이블이 포함되었는가?
- [ ] 부록이 포함되었는가? (있는 경우)
- [ ] 각 주요 수식에 직관적 이해가 있는가?
- [ ] 아키텍처 다이어그램이 있는가?
- [ ] 수식 흐름 다이어그램이 있는가?

누락된 항목이 있으면 해당 섹션을 다시 fetch하여 보충한다.
모든 항목이 완료되면 사용자에게 결과를 보고한다.
