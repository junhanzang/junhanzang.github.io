---
title: "2071. Maximum Number of Tasks You Can Assign"
date: 2025-05-01 14:12:30
categories:
  - 알고리즘 공부
---

```
from typing import List
import collections
import bisect

class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        # Sort tasks and workers in ascending order
        tasks.sort()
        workers.sort()
        
        n, m = len(tasks), len(workers)
        
        # Check if we can assign k tasks
        def can_assign(k):
            # Consider only the k easiest tasks
            selected_tasks = tasks[:k]
            # Consider the k strongest workers
            selected_workers = workers[m-k:]
            
            # Use multiset (simulated by sorted list + counts) for efficient operations
            worker_multiset = []
            worker_counts = {}
            
            for w in selected_workers:
                if w in worker_counts:
                    worker_counts[w] += 1
                else:
                    worker_counts[w] = 1
                    bisect.insort(worker_multiset, w)
            
            remaining_pills = pills
            
            # Process tasks from hardest to easiest
            for i in range(k-1, -1, -1):
                task = selected_tasks[i]
                
                # If the strongest available worker can do the task without a pill
                if worker_multiset and worker_multiset[-1] >= task:
                    # Remove the strongest worker (greedy approach)
                    strongest = worker_multiset[-1]
                    worker_counts[strongest] -= 1
                    if worker_counts[strongest] == 0:
                        worker_multiset.pop()
                        del worker_counts[strongest]
                else:
                    # Need to use a pill
                    if remaining_pills <= 0:
                        return False
                    
                    # Find the weakest worker who can do the task with a pill
                    idx = bisect.bisect_left(worker_multiset, task - strength)
                    
                    # No worker can complete the task even with a pill
                    if idx == len(worker_multiset):
                        return False
                    
                    # Use the weakest eligible worker
                    weakest_eligible = worker_multiset[idx]
                    worker_counts[weakest_eligible] -= 1
                    if worker_counts[weakest_eligible] == 0:
                        worker_multiset.pop(idx)
                        del worker_counts[weakest_eligible]
                    
                    remaining_pills -= 1
            
            return True
        
        # Binary search to find the maximum k
        left, right = 0, min(n, m)
        
        while left < right:
            mid = (left + right + 1) // 2
            if can_assign(mid):
                left = mid
            else:
                right = mid - 1
                
        return left
```

# 작업 할당 문제에서의 엣지 케이스 분석

## 중요한 엣지 케이스와 그 처리 방법

### 1. 작업자가 작업보다 적은 경우

**사례**: tasks = [1, 2, 3, 4, 5], workers = [3, 4]

**처리 방법**:

- 최대 완료 가능한 작업 수는 작업자 수(2)를 초과할 수 없음
- 이진 탐색 초기 범위를 0에서 min(n, m)으로 설정하여 자동으로 처리

### 2. 모든 작업자가 약한 경우

**사례**: tasks = [10, 20], workers = [1, 2, 3], pills = 1, strength = 5

**처리 방법**:

- 마법 약을 써도 어떤 작업도 완료할 수 없는 상황
- can\_assign 함수에서 마법 약을 써도 작업을 완료할 수 있는 작업자가 없으면 False 반환
- 이진 탐색은 0으로 수렴

### 3. 작업자 힘이 모두 동일한 경우

**사례**: tasks = [5, 10, 15], workers = [10, 10, 10]

**처리 방법**:

- 정렬 후에도 모든 작업자 힘이 같음
- worker\_multiset에 하나의 값(10)만 포함되고 worker\_counts에 개수(3) 저장
- 이 경우에도 알고리즘은 정상 작동

### 4. 마법 약으로도 어떤 작업도 완료 불가능한 경우

**사례**: tasks = [100], workers = [10], pills = 5, strength = 10

**처리 방법**:

- 가장 강한 작업자(10) + 마법 약(10) = 20 < 가장 쉬운 작업(100)
- can\_assign 함수에서 False 반환
- 이진 탐색은 0으로 수렴

### 5. 모든 작업자가 마법 약 없이 모든 작업 완료 가능한 경우

**사례**: tasks = [1, 2, 3], workers = [5, 5, 5]

**처리 방법**:

- 마법 약이 필요 없는 상황
- 모든 작업자가 모든 작업을 완료할 수 있음
- can\_assign 함수에서 가장 강한 작업자부터 할당하는 로직이 잘 작동
- 결과는 모든 작업(3개)을 완료 가능

### 6. 마법 약 개수가 0인 경우

**사례**: tasks = [3, 4, 5], workers = [3, 4, 5], pills = 0, strength = 10

**처리 방법**:

- 마법 약 없이 작업자가 작업을 완료할 수 있는지만 확인
- 작업자 힘 ≥ 작업 요구 힘인 경우에만 작업 완료 가능
- 이 사례에서는 모든 작업이 완료 가능

### 7. 작업자 힘이 0인 경우

**사례**: tasks = [5, 10], workers = [0, 0], pills = 2, strength = 6

**처리 방법**:

- 모든 작업자가 마법 약 없이는 어떤 작업도 완료 불가
- 첫 번째 작업(5)은 마법 약(6)으로 작업자 힘을 증가시키면 완료 가능
- 두 번째 작업(10)은 마법 약(6)을 써도 완료 불가
- 결과는 1개 작업 완료 가능

### 8. 동일한 힘을 가진 여러 작업이 있는 경우

**사례**: tasks = [5, 5, 5, 10], workers = [5, 6, 7], pills = 1, strength = 5

**처리 방법**:

- 정렬 후 동일한 작업들이 연속됨
- 작업을 어려운 순으로 처리하므로 이런 경우 크게 문제 없음
- 최적으로 할당하면 3개의 작업 완료 가능

### 9. 마법 약을 모두 사용할 필요가 없는 경우

**사례**: tasks = [2, 3], workers = [1, 5], pills = 3, strength = 2

**처리 방법**:

- 작업자 [5]는 마법 약 없이 작업 [3]을 완료 가능
- 작업자 [1]에게 마법 약을 1개만 사용하여 작업 [2] 완료 가능
- 나머지 마법 약 2개는 사용할 필요 없음
- 결과는 2개 작업 완료 가능

### 10. 작업 수와 작업자 수가 0인 경우 (극단적인 경우)

**사례**: tasks = [], workers = [], pills = 5, strength = 10

**처리 방법**:

- 이진 탐색 초기 범위는 [0, 0]
- 결과는 당연히 0
- 우리 알고리즘은 이런 예외 케이스도 자동으로 처리

## 알고리즘 견고성 검증

이러한 엣지 케이스들을 처리할 수 있는 이유는:

1. **초기 범위 설정**: 이진 탐색의 초기 범위를 0에서 min(n, m)으로 설정하여 작업자 수나 작업 수가 0이거나 서로 다른 경우를 자동으로 처리
2. **정렬 전처리**: 작업과 작업자를 정렬하여 동일한 힘을 가진 작업이나 작업자를 효과적으로 처리
3. **탐욕 알고리즘**: 어려운 작업부터 처리하고, 마법 약 없이 할 수 있으면 가장 강한 작업자를 사용하는 전략으로 최적의 결과 도출
4. **마법 약 최적 사용**: 마법 약이 필요할 때 그것으로 작업을 완료할 수 있는 가장 약한 작업자에게 할당
5. **이진 탐색 종료 조건**: left < right가 아닐 때 이진 탐색 종료, 결과로 left 반환하여 모든 경우에 대한 최대 작업 수 보장

이러한 전략들을 통해 알고리즘은 다양한 입력 케이스에서 안정적으로 작동하며, 언제나 최대 작업 수를 찾아낼 수 있습니다.
