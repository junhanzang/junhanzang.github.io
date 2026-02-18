---
title: "[Look at it Differently: Novel View Synthesis] Neural Light Spheres for Implicit Image Stitching and View Synthesis"
date: 2024-12-19 01:26:14
categories:
  - 컨퍼런스/ASIA SIGGRAPH 2024
---

<https://dl.acm.org/doi/10.1145/3680528.3687660>

[Neural Light Spheres for Implicit Image Stitching and View Synthesis | SIGGRAPH Asia 2024 Conference Papers](https://dl.acm.org/doi/10.1145/3680528.3687660)

### **Neural Light Spheres: 파노라마 이미지를 고품질로 변환하기**

**"Neural Light Spheres"** 논문은 파노라마 이미지의 한계를 극복하고, 고품질의 새로운 시점을 합성할 수 있는 방법을 제시합니다.  
파노라마 촬영은 제한된 뷰포인트와 스파스 뷰(sparse view) 문제로 인해 합성의 품질이 저하되는 경우가 많습니다. 이를 해결하기 위해, 본 연구는 **Neural Light Sphere 모델**을 통해 파노라마 데이터를 효율적으로 활용하여 새로운 접근법을 제안했습니다.

![](/assets/images/posts/424/img.jpg)

---

### **핵심 내용 요약**

1. **스파스 뷰 문제 해결**
   - 기존의 파노라마 이미지는 종종 서로 다른 뷰포인트에서 촬영된 이미지 간의 불일치로 인해 합성 품질이 떨어졌습니다.
   - Neural Light Spheres는 **스피어(sphere) 형식**으로 이미지를 압축하여 촬영 경로를 최적화하고, 각 이미지의 뎁스(depth), 색상(color), 그리고 뷰 디펜던트 효과를 반영합니다.
2. **Test-Time Optimization (TTO)**
   - 모델은 **테스트 시점 최적화(TTO)**를 통해 입력된 파노라마 비디오를 학습합니다.
   - 특정 장면의 특징을 학습하면서, **광각 뷰**와 **고품질 이미지 합성**이 가능하도록 최적화합니다.
3. **실시간 렌더링**
   - Neural Light Spheres는 80MB의 소형 모델 크기와 **1080p 해상도에서 50FPS의 실시간 렌더링**을 지원하며, 파노라마를 대화형 환경으로 변환합니다.
4. **멀티레이어 접근 방식**
   - **레이 오프셋 모델(ray offset model)**과 **색상 모델(color model)**을 분리하여 파라랙스(parallax), 조명 효과, 그리고 장면 움직임을 정교하게 재현합니다.

---
