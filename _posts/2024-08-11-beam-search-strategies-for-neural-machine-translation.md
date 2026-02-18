---
title: "Beam Search Strategies for Neural Machine Translation"
date: 2024-08-11 21:05:28
categories:
  - 인공지능
---

<https://arxiv.org/abs/1702.01806>

[Beam Search Strategies for Neural Machine Translation](https://arxiv.org/abs/1702.01806)

**초록**

신경 기계 번역(NMT)의 기본 개념은 주어진 병렬 말뭉치에서 번역 성능을 극대화하는 대규모 신경망을 훈련시키는 것입니다. NMT는 훈련된 조건부 확률을 최대화하는 새로운 번역을 생성하기 위해 단순한 좌->우 빔 검색(beam-search) 디코더를 사용합니다. 현재의 빔 검색 전략은 각 시간 단계에서 고정된 수의 활성 후보들을 유지하면서 좌->우로 목표 문장을 단어 단위로 생성합니다. 첫째, 이 단순한 검색은 현재 최고 점수보다 훨씬 낮은 후보들도 확장하기 때문에 적응성이 떨어집니다. 둘째, 점수가 최고와 가까운 경우에도 최고 점수의 후보들에 포함되지 않는다면 가설을 확장하지 않습니다. 후자의 문제는 빔 크기를 늘려서 성능 개선이 관찰되지 않을 때까지 피할 수 있습니다. 더 나은 성능을 얻을 수 있지만, 이 경우 디코딩 속도가 느려지는 단점이 있습니다. 이 논문에서는 후보 점수에 따라 각 시간 단계에서 후보 크기가 달라질 수 있는 보다 유연한 빔 검색 전략을 적용하여 디코더의 속도를 높이는 것에 중점을 둡니다. 독일어-영어 및 중국어-영어 두 언어 쌍에 대해 원래의 디코더 속도를 최대 43%까지 높이면서도 번역 품질을 잃지 않았습니다.

**1. 서론**

신경 기계 번역(NMT)은 기존의 통계적 기계 번역(SMT) 모델에 비해 유사하거나 더 나은 성능을 보여주면서 (Jean et al., 2015; Luong et al., 2015), 최근 몇 년 동안 매우 인기를 끌고 있습니다 (Kalchbrenner and Blunsom, 2013; Sutskever et al., 2014; Bahdanau et al., 2014). NMT의 최근 성공과 함께, 이를 더 실용적으로 만드는 것에 대한 관심이 집중되고 있습니다. 그 중 하나의 과제는 주어진 원문에 대해 최적의 번역을 추출하는 검색 전략입니다. NMT에서는 훈련된 NMT 모델의 조건부 확률을 최대화하는 번역을 찾기 위해 간단한 빔 검색 디코더를 사용하여 새로운 문장을 번역합니다. 빔 검색 전략은 각 시간 단계에서 고정된 수의 활성 후보(빔)를 유지하면서 좌->우로 단어 단위로 번역을 생성합니다. 빔 크기를 늘리면 번역 성능이 향상될 수 있지만, 디코더 속도가 크게 감소하는 대가를 치러야 합니다. 일반적으로, 빔 크기를 계속 증가시켜도 더 이상 번역 품질이 향상되지 않는 포화점에 도달합니다. 이 연구의 동기는 두 가지입니다. 첫째, 검색 그래프를 가지치기하여 번역 품질을 잃지 않고 디코딩 속도를 높입니다. 둘째, 최고 점수를 받은 후보들이 종종 동일한 이력을 공유하며 동일한 부분 가설에서 나온다는 점을 관찰했습니다. 우리는 동일한 부분 가설에서 나오는 후보들의 수를 제한하여, 더 높은 빔 크기만을 사용하는 것이 아니라 디코딩 속도를 줄이지 않으면서도 다양성을 도입하려고 합니다.

**2. 관련 연구**

시퀀스-투-시퀀스 모델에 대한 원래의 빔 검색은 (Graves, 2012; Boulanger-Lewandowski et al., 2013)에 의해 도입되었으며, 신경 기계 번역에서는 (Sutskever et al., 2014)에 의해 설명되었습니다. (Hu et al., 2015; Mi et al., 2016)는 계산 복잡성을 줄이기 위해 제한된 번역 후보 단어 집합만을 고려하는 제약 소프트맥스 함수로 빔 검색을 개선했습니다. 이 방법의 장점은 소수의 후보들만 정규화하여 디코딩 속도를 향상시킨다는 것입니다. (Wu et al., 2016)는 검색 중에 최적의 토큰보다 빔 크기만큼 이하인 지역 점수를 가진 토큰만을 고려했습니다. 또한, 저자들은 모든 부분 가설 중에서 최종 가설(이미 생성된 경우)보다 점수가 빔 크기만큼 낮은 것들을 가지치기했습니다. 이 연구에서는 통계적 기계 번역에서 구문 테이블 가지치기(Zens et al., 2012) 등에서 성공적으로 적용된 다양한 절대 및 상대 가지치기 방식을 조사합니다.

**3. 원래의 빔 검색**

원래의 빔 검색 전략은 특정 모델이 제공하는 조건부 확률을 최대화하는 번역을 찾습니다. 이 전략은 좌->우로 번역을 구축하며, 각 시간 단계에서 가장 높은 로그 확률을 가진 고정된 수(빔)의 번역 후보를 유지합니다. 가장 높은 점수를 가진 후보 중에서 시퀀스 끝 기호가 선택될 때마다 빔 크기는 하나씩 줄어들고, 번역은 최종 후보 리스트에 저장됩니다. 빔 크기가 0이 되면 검색을 중지하고, 최종 후보 리스트에서 (목표 단어 수로 정규화된) 가장 높은 로그 확률을 가진 번역을 선택합니다.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
<defs>
<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
<polygon points="0 0, 10 3.5, 0 7" />
</marker>
</defs>
<!-- 배경 -->
<rect width="100%" height="100%" fill="#f0f0f0"/>
<!-- 노드 -->
<g font-family="Arial, sans-serif" font-size="14">
<!-- 시작 노드 -->
<circle cx="50" cy="200" r="30" fill="#4CAF50"/>
<text x="50" y="205" text-anchor="middle" fill="white">시작</text>
<!-- 첫 번째 단계 -->
<rect x="150" y="100" width="100" height="60" rx="10" fill="#2196F3"/>
<text x="200" y="135" text-anchor="middle" fill="white">나는 (-1.0)</text>
<rect x="150" y="170" width="100" height="60" rx="10" fill="#2196F3"/>
<text x="200" y="205" text-anchor="middle" fill="white">저는 (-1.2)</text>
<rect x="150" y="240" width="100" height="60" rx="10" fill="#2196F3"/>
<text x="200" y="275" text-anchor="middle" fill="white">난 (-1.5)</text>
<!-- 두 번째 단계 -->
<rect x="350" y="70" width="150" height="60" rx="10" fill="#FFC107"/>
<text x="425" y="105" text-anchor="middle" fill="white">나는 사랑합니다 (-2.0)</text>
<rect x="350" y="140" width="150" height="60" rx="10" fill="#FFC107"/>
<text x="425" y="175" text-anchor="middle" fill="white">나는 좋아합니다 (-2.2)</text>
<rect x="350" y="210" width="150" height="60" rx="10" fill="#FFC107"/>
<text x="425" y="245" text-anchor="middle" fill="white">저는 사랑합니다 (-2.3)</text>
<!-- 마지막 단계 -->
<rect x="600" y="40" width="180" height="60" rx="10" fill="#FF5722"/>
<text x="690" y="75" text-anchor="middle" fill="white">나는 사랑합니다 당신을 (-3.0)</text>
<rect x="600" y="110" width="180" height="60" rx="10" fill="#FF5722"/>
<text x="690" y="145" text-anchor="middle" fill="white">나는 사랑합니다 너를 (-3.2)</text>
<rect x="600" y="180" width="180" height="60" rx="10" fill="#FF5722"/>
<text x="690" y="215" text-anchor="middle" fill="white">저는 사랑합니다 당신을 (-3.3)</text>
</g>
<!-- 화살표 -->
<g stroke="black" stroke-width="2" marker-end="url(#arrowhead)">
<line x1="80" y1="200" x2="140" y2="130"/>
<line x1="80" y1="200" x2="140" y2="200"/>
<line x1="80" y1="200" x2="140" y2="270"/>
<line x1="250" y1="130" x2="340" y2="100"/>
<line x1="250" y1="130" x2="340" y2="170"/>
<line x1="250" y1="200" x2="340" y2="240"/>
<line x1="500" y1="100" x2="590" y2="70"/>
<line x1="500" y1="100" x2="590" y2="140"/>
<line x1="500" y1="170" x2="590" y2="210"/>
</g>
<!-- 레이블 -->
<g font-family="Arial, sans-serif" font-size="16" font-weight="bold">
<text x="200" y="50" text-anchor="middle">첫 번째 단계</text>
<text x="425" y="50" text-anchor="middle">두 번째 단계</text>
<text x="690" y="20" text-anchor="middle">마지막 단계</text>
</g>
</svg>

**4. 검색 전략**

이 섹션에서는 우리가 실험한 다양한 전략에 대해 설명합니다. 모든 확장에서 우리는 먼저 후보 리스트를 현재 빔 크기로 줄이고, 그 위에 하나 이상의 가지치기 방식을 적용합니다.

**상대적 임계값 가지치기 (Relative Threshold Pruning).** 상대적 임계값 가지치기 방법은 최상의 활성 후보보다 훨씬 성능이 떨어지는 후보를 삭제합니다. 가지치기 임계값 rp와 활성 후보 리스트 C가 주어졌을 때, 후보 cand가 C에 속하고 다음 조건을 만족하면 삭제됩니다:

![](/assets/images/posts/247/img.png)

**절대 임계값 가지치기 (Absolute Threshold Pruning).** 점수의 상대적 차이를 고려하는 대신, 우리는 최상의 활성 후보보다 특정 임계값만큼 낮은 후보를 삭제합니다. 가지치기 임계값 ap와 활성 후보 리스트 C가 주어졌을 때, 후보 cand가 C에 속하고 다음 조건을 만족하면 삭제됩니다:

![](/assets/images/posts/247/img_1.png)

**상대적 지역 임계값 가지치기 (Relative Local Threshold Pruning).** 이 가지치기 접근법에서는 전체 점수가 아닌 마지막으로 생성된 단어의 점수 scorew만을 고려합니다. 가지치기 임계값 rpl과 활성 후보 리스트 C가 주어졌을 때, 후보 cand가 C에 속하고 다음 조건을 만족하면 삭제됩니다:

![](/assets/images/posts/247/img_2.png)

**노드당 최대 후보 수 (Maximum Candidates per Node).** 디코딩 과정에서 대부분의 부분 가설들이 동일한 선행 단어들을 공유한다는 점을 관찰했습니다. 더 많은 다양성을 도입하기 위해, 각 시간 단계에서 동일한 이력을 가진 후보 수를 고정된 수로 제한합니다. 최대 후보 임계값 mc와 활성 후보 리스트 C가 주어졌을 때, 후보 cand가 C에 속하고 동일한 이력을 가진 더 높은 점수의 부분 가설이 이미 mc개 존재하면 삭제됩니다.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
<!-- 배경 -->
<rect width="100%" height="100%" fill="#f0f0f0"/>
<!-- 제목 -->
<text x="400" y="30" font-family="Arial, sans-serif" font-size="24" font-weight="bold" text-anchor="middle">빔 검색 가지치기 방법</text>
<!-- 1. 상대적 임계값 가지치기 -->
<g transform="translate(0, 50)">
<rect x="10" y="10" width="380" height="220" fill="#e1f5fe" stroke="#01579b" stroke-width="2"/>
<text x="200" y="40" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">상대적 임계값 가지치기</text>
<line x1="30" y1="70" x2="370" y2="70" stroke="#01579b" stroke-width="2"/>
<circle cx="70" cy="120" r="30" fill="#4caf50"/>
<text x="70" y="125" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">최상의 후보</text>
<circle cx="200" cy="160" r="30" fill="#ff9800"/>
<text x="200" y="165" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">유지</text>
<circle cx="330" cy="200" r="30" fill="#f44336"/>
<text x="330" y="205" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">제거</text>
<path d="M100 120 L170 160" stroke="#000" stroke-width="2" fill="none"/>
<path d="M230 160 L300 200" stroke="#000" stroke-width="2" fill="none"/>
<text x="135" y="130" font-family="Arial, sans-serif" font-size="12">rp \* score</text>
<text x="280" y="170" font-family="Arial, sans-serif" font-size="12">&lt; rp \* score</text>
</g>
<!-- 2. 절대 임계값 가지치기 -->
<g transform="translate(410, 50)">
<rect x="10" y="10" width="380" height="220" fill="#fff3e0" stroke="#e65100" stroke-width="2"/>
<text x="200" y="40" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">절대 임계값 가지치기</text>
<line x1="30" y1="70" x2="370" y2="70" stroke="#e65100" stroke-width="2"/>
<circle cx="70" cy="120" r="30" fill="#4caf50"/>
<text x="70" y="125" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">최상의 후보</text>
<circle cx="200" cy="160" r="30" fill="#ff9800"/>
<text x="200" y="165" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">유지</text>
<circle cx="330" cy="200" r="30" fill="#f44336"/>
<text x="330" y="205" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">제거</text>
<path d="M100 120 L170 160" stroke="#000" stroke-width="2" fill="none"/>
<path d="M230 160 L300 200" stroke="#000" stroke-width="2" fill="none"/>
<text x="135" y="130" font-family="Arial, sans-serif" font-size="12">score - ap</text>
<text x="280" y="170" font-family="Arial, sans-serif" font-size="12">&lt; score - ap</text>
</g>
<!-- 3. 상대적 지역 임계값 가지치기 -->
<g transform="translate(0, 300)">
<rect x="10" y="10" width="380" height="220" fill="#e8f5e9" stroke="#1b5e20" stroke-width="2"/>
<text x="200" y="40" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">상대적 지역 임계값 가지치기</text>
<line x1="30" y1="70" x2="370" y2="70" stroke="#1b5e20" stroke-width="2"/>
<circle cx="70" cy="120" r="30" fill="#4caf50"/>
<text x="70" y="125" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">최상의 단어</text>
<circle cx="200" cy="160" r="30" fill="#ff9800"/>
<text x="200" y="165" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">유지</text>
<circle cx="330" cy="200" r="30" fill="#f44336"/>
<text x="330" y="205" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">제거</text>
<path d="M100 120 L170 160" stroke="#000" stroke-width="2" fill="none"/>
<path d="M230 160 L300 200" stroke="#000" stroke-width="2" fill="none"/>
<text x="135" y="130" font-family="Arial, sans-serif" font-size="12">rpl \* scorew</text>
<text x="280" y="170" font-family="Arial, sans-serif" font-size="12">&lt; rpl \* scorew</text>
</g>
<!-- 4. 노드당 최대 후보 수 -->
<g transform="translate(410, 300)">
<rect x="10" y="10" width="380" height="220" fill="#fce4ec" stroke="#880e4f" stroke-width="2"/>
<text x="200" y="40" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">노드당 최대 후보 수</text>
<line x1="30" y1="70" x2="370" y2="70" stroke="#880e4f" stroke-width="2"/>
<rect x="50" y="90" width="300" height="120" fill="#f8bbd0" stroke="#880e4f" stroke-width="2"/>
<text x="200" y="115" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">동일한 이력을 가진 노드</text>
<circle cx="100" cy="160" r="20" fill="#4caf50"/>
<circle cx="160" cy="160" r="20" fill="#4caf50"/>
<circle cx="220" cy="160" r="20" fill="#ff9800"/>
<text x="220" y="165" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">mc</text>
<circle cx="280" cy="160" r="20" fill="#f44336"/>
<text x="280" y="165" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="white">제거</text>
</g>
</svg>

**5. 실험**

독일어-영어 번역 작업을 위해, 우리는 WMT 2016 훈련 데이터(Bojar et al., 2016)에 기반한 NMT 시스템을 훈련시켰습니다 (390만 개의 병렬 문장). 중국어-영어 실험에서는 BOLT 프로젝트의 1100만 개 문장에서 훈련된 NMT 시스템을 사용했습니다.

모든 실험에서 우리는 (Bahdanau et al., 2014)과 유사한 주의(attention) 기반의 NMT 구현을 사용했습니다. 독일어-영어 번역에서는, 단어 대신 바이트 페어 인코딩(Sennrich et al., 2015)을 통해 추출된 서브워드 단위를 사용하여 어휘를 축소하고, 소스와 타겟 모두에 대해 40k 서브워드 심볼로 어휘를 축소했습니다. 중국어-영어 번역에서는 소스와 타겟 언어 모두에서 가장 빈번하게 사용되는 상위 300,000개의 단어로 어휘를 제한했습니다. 이 어휘에 포함되지 않은 단어들은 미지의 토큰으로 변환됩니다. 번역 중에, 우리는 정렬(주의 메커니즘에서 얻은)을 사용하여 미지의 토큰을 잠재적 목표어(병렬 데이터에서 훈련된 IBM Model-1에서 얻은) 또는 소스 단어 자체(목표어가 발견되지 않은 경우)로 대체합니다 (Mi et al., 2016). 임베딩 차원은 620으로 설정하고, RNN GRU 레이어는 각각 1000개의 셀로 고정했습니다. 훈련 절차에서는 미니배치 크기 64로 SGD(Bishop, 1995)를 사용하여 모델 파라미터를 업데이트했습니다. 훈련 데이터는 각 에포크 후에 섞습니다.

![](/assets/images/posts/247/img_3.png)

**Figure 1:** 독일어-영어: newstest2014에서 다양한 빔 크기로 원래의 빔 검색 전략을 사용한 결과.

![](/assets/images/posts/247/img_4.png)

**Figure 2:** 독일어-영어: newstest2014에서 측정된 상대적 가지치기의 다양한 값들.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
<!-- 배경 -->
<rect width="100%" height="100%" fill="#f0f0f0"/>
<!-- 제목 -->
<text x="400" y="30" font-family="Arial, sans-serif" font-size="24" font-weight="bold" text-anchor="middle">신경망 모델 구조 및 훈련 과정</text>
<!-- 모델 구조 -->
<g transform="translate(50, 80)">
<rect x="0" y="0" width="300" height="300" fill="#e1f5fe" stroke="#01579b" stroke-width="2"/>
<text x="150" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">모델 구조</text>
<!-- 임베딩 레이어 -->
<rect x="50" y="60" width="200" height="40" fill="#bbdefb" stroke="#1565c0" stroke-width="2"/>
<text x="150" y="85" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">임베딩 레이어 (620 차원)</text>
<!-- GRU 레이어 -->
<rect x="50" y="120" width="200" height="40" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2"/>
<text x="150" y="145" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">GRU 레이어 1 (1000 셀)</text>
<rect x="50" y="180" width="200" height="40" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2"/>
<text x="150" y="205" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">GRU 레이어 2 (1000 셀)</text>
<!-- 출력 레이어 -->
<rect x="50" y="240" width="200" height="40" fill="#ffcdd2" stroke="#b71c1c" stroke-width="2"/>
<text x="150" y="265" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">출력 레이어</text>
</g>
<!-- 훈련 과정 -->
<g transform="translate(400, 80)">
<rect x="0" y="0" width="350" height="300" fill="#fff3e0" stroke="#e65100" stroke-width="2"/>
<text x="175" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">훈련 과정</text>
<!-- 미니배치 -->
<rect x="50" y="60" width="100" height="60" fill="#ffe0b2" stroke="#ef6c00" stroke-width="2"/>
<text x="100" y="90" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">미니배치</text>
<text x="100" y="110" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">크기: 64</text>
<!-- SGD -->
<rect x="200" y="60" width="100" height="60" fill="#ffccbc" stroke="#d84315" stroke-width="2"/>
<text x="250" y="90" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">SGD</text>
<text x="250" y="110" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">(Bishop, 1995)</text>
<!-- 에포크 -->
<rect x="50" y="160" width="250" height="60" fill="#fff9c4" stroke="#f9a825" stroke-width="2"/>
<text x="175" y="190" font-family="Arial, sans-serif" font-size="14" text-anchor="middle">에포크 완료</text>
<text x="175" y="210" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">데이터 섞기</text>
<!-- 화살표 -->
<path d="M150 90 L200 90" stroke="#000" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
<path d="M250 120 L250 160" stroke="#000" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
<path d="M175 220 L175 250 L50 250 L50 90" stroke="#000" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
</g>
<!-- 화살표 마커 정의 -->
<defs>
<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
<polygon points="0 0, 10 3.5, 0 7" />
</marker>
</defs>
</svg>

디코딩 속도는 두 가지 숫자로 측정합니다. 첫째, 가지치기를 하지 않은 동일한 설정과 비교하여 실제 속도를 비교합니다. 둘째, 시간 단계당 평균 팬 아웃(fan out)을 측정합니다. 각 시간 단계에서 팬 아웃은 우리가 확장하는 후보들의 수로 정의됩니다. 팬 아웃의 상한선은 빔의 크기이지만, 조기 중지(문장 끝 기호를 예측할 때마다 빔을 줄임) 또는 제안된 가지치기 방식으로 인해 줄어들 수 있습니다. 각 가지치기 기법에 대해, 우리는 다른 가지치기 임계값으로 실험을 수행하고, 선택된 세트를 기준으로 번역 성능이 저하되지 않은 가장 큰 임계값을 선택했습니다.

**그림 1**에서는 독일어-영어 번역 성능과 다양한 빔 크기에 따른 문장당 평균 팬 아웃을 볼 수 있습니다. 이 실험을 기반으로 우리는 빔 크기 5와 14로 가지치기 실험을 진행하기로 결정했습니다. 독일어-영어 결과는 **표 1**에서 확인할 수 있습니다. 모든 가지치기 기법을 조합하여 사용함으로써, 성능 저하 없이 빔 크기 5에서 디코딩 속도를 13%, 빔 크기 14에서 43%까지 향상시킬 수 있었습니다. 상대적 가지치기 기법은 빔 크기 5에서 가장 효과적이었으며, 절대적 가지치기 기법은 빔 크기 14에서 가장 잘 작동했습니다. **그림 2**에서는 빔 크기 5에서 다양한 상대적 가지치기 임계값을 사용한 디코딩 속도를 보여줍니다. 임계값을 0.6 이상으로 설정하면 번역 성능이 저하됩니다. 가지치기를 적용하면 고정된 빔 크기 없이도 디코딩이 가능해지는 긍정적인 부수 효과가 나타났습니다. 그럼에도 불구하고, 번역 성능이 변하지 않는 반면 디코딩 속도는 떨어졌습니다. 또한, 가지치기 기법에 의해 발생한 검색 오류(최고 점수를 받은 가설이 가지치기된 횟수)의 수를 조사했습니다. 빔 크기 5에서 5%의 문장이 검색 오류로 인해 변경되었고, 빔 크기 14에서는 9%의 문장이 변경되었습니다.

중국어-영어 번역 결과는 **표 2**에서 확인할 수 있습니다. 빔 크기 5에서 디코딩 속도를 10%, 빔 크기 14에서 24%까지 번역 품질의 손실 없이 향상시킬 수 있었습니다. 추가로, 검색 가지치기에 의해 발생한 검색 오류의 수를 측정했습니다. 빔 크기 5에서는 4%의 문장이 변경된 반면, 빔 크기 14에서는 22%의 문장이 변경되었습니다.

![](/assets/images/posts/247/img_5.png)

**표 1:** 독일어-영어 결과: 상대적 가지치기(rp), 절대적 가지치기(ap), 상대적 지역 가지치기(rpl), 노드당 최대 후보 수(mc). 평균 팬 아웃은 디코딩 동안 각 시간 단계에서 유지되는 후보들의 평균 수를 나타냅니다.

![](/assets/images/posts/247/img_6.png)

**표 2:** 중국어-영어 결과: 상대적 가지치기(rp), 절대적 가지치기(ap), 상대적 지역 가지치기(rpl), 노드당 최대 후보 수(mc).

**6. 결론**

신경 기계 번역에서 사용되는 원래의 빔 검색 디코더는 매우 단순합니다. 이 디코더는 고정된 수의 후보(빔)만을 고려하며, 좌->우로 번역을 생성합니다. 빔 크기를 충분히 크게 설정하면 최적의 번역 성능을 보장할 수 있지만, 최상의 후보와 점수가 크게 차이 나는 많은 후보들도 탐색하게 되는 단점이 있습니다. 이 논문에서는 최상의 후보와 점수가 크게 차이 나는 후보들을 가지치기하는 여러 가지 가지치기 기법을 도입했습니다. 절대적 및 상대적 가지치기 방식을 결합하여 적용함으로써 번역 품질의 손실 없이 디코더 속도를 최대 43%까지 높였습니다. 디코더에 더 많은 다양성을 추가하는 것은 번역 품질을 향상시키지 못했습니다.

[1702.01806v2.pdf

0.08MB](./file/1702.01806v2.pdf)
