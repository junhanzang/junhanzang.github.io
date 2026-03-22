# CLAUDE.md — JunHan's AI Factory 블로그 작업 가이드

## 프로젝트 개요
- Jekyll 기반 블로그 (minimal-mistakes 테마, 다크 스킨)
- 주제: AI/딥러닝 논문 리뷰 (한국어)
- 배포: GitHub Pages (junhanzang.github.io)

## 논문 번역 규칙

### 요청 방식
사용자가 arxiv URL을 주면 전체 논문을 한국어로 번역하여 블로그 포스트로 작성한다.

### 번역 범위
- **전체 상세 번역**: Abstract, Introduction, Methodology, Experiments, Ablation, Conclusion 모든 섹션
- 저자, 소속, 논문 링크를 상단에 표기

### 수식 처리 (핵심)
1. **MathJax로 수식 렌더링**: `$$...$$` (블록), `$...$` (인라인)
2. **모든 수식 아래에 "직관적 이해" 추가**: `> **직관적 이해**: ...` 형태로 수식의 의미를 쉽게 설명
3. **수식의 데이터 흐름/차원 변화를 Mermaid 다이어그램으로 시각화**
4. **복잡한 수식은 ASCII 차트로 보충** (곡선, 비교 등)

### 시각화 규칙
- **아키텍처/파이프라인**: Mermaid flowchart (`graph TB/LR`)
- **데이터 흐름**: Mermaid로 입력→처리→출력 흐름도
- **비교/대조**: Mermaid subgraph로 나란히 배치
- **수치 변화**: ASCII 차트
- Mermaid 테마는 `dark` 사용
- 다이어그램 문법: `<div class="mermaid">...</div>`

### 포스트 포맷
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

### 파일명 규칙
`YYYY-MM-DD-논문제목-영어-소문자-하이픈.md`
- `_posts/` 디렉토리에 저장

### kramdown 주의사항
- 인라인 수식에서 언더스코어(`_`)는 `\_`로 이스케이프
- 수식 블록 전후에 빈 줄 필수

## 기술 스택
- Markdown: kramdown
- 수식: MathJax 2.7.9 (인라인 `$...$`, 블록 `$$...$$`)
- 다이어그램: Mermaid 11 (CDN, 다크 테마)
- 댓글: Giscus

## 빌드/배포
- `git push origin main` → GitHub Pages 자동 배포
