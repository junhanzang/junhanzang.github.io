---
title: "2845. Count of Interesting Subarrays"
date: 2025-04-25 17:53:17
categories:
  - 알고리즘 공부
---

```
class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count_map = defaultdict(int)
        print(count_map)
        count_map[0] = 1  # 시작점
        prefix = 0
        answer = 0

        for num in nums:
            # 현재 원소가 interesting 조건 만족하는가?
            if num % modulo == k:
                prefix += 1
            
            # prefix[i] % modulo == prefix[j] % modulo + k
            target = (prefix - k) % modulo
            answer += count_map[target]
            count_map[prefix % modulo] += 1
        
        return answer
```

target = (prefix - k) % modulo가 성립하는 이유를 단계별로 설명해드리겠습니다.

먼저 이 문제가 무엇을 찾고자 하는지 정리해보겠습니다:

1. "Interesting subarray"는 nums[j+1+1] 부분 배열에서 (value % modulo == k)인 요소의 개수가 정확히 k개인 부분 배열입니다.
2. prefix는 현재까지 처리한 숫자들 중 modulo로 나눴을 때 k와 같은 값의 누적 개수입니다.

이제 왜 target = (prefix - k) % modulo가 필요한지 설명하겠습니다:

1. 현재 위치 i에서의 prefix count를 prefix[i]라고 합시다.
2. 이전 위치 j에서의 prefix count를 prefix[j]라고 합시다.
3. 부분 배열 nums[j+1+1]에 있는 "interesting한 요소"의 개수는 (prefix[i] - prefix[j])입니다.
4. 우리가 찾고자 하는 조건은 (prefix[i] - prefix[j]) % modulo = k 입니다. → 이 부분 배열 내에서 모듈로 조건을 만족하는 요소가 정확히 k개라는 의미입니다.
5. 이 식을 j에 대해 풀면:
   - (prefix[i] - prefix[j]) % modulo = k
   - prefix[i] % modulo - prefix[j] % modulo = k % modulo (모듈러 산술의 특성)
   - prefix[j] % modulo = (prefix[i] - k) % modulo
6. 따라서 우리가 찾는 이전 위치 j의 조건은 prefix[j] % modulo = (prefix[i] - k) % modulo 입니다.
7. 코드에서는 현재 위치의 prefix를 prefix라고 표현했으므로, 우리가 찾는 target은 (prefix - k) % modulo가 됩니다.

이것이 바로 target = (prefix - k) % modulo가 성립하는 이유입니다. 이 target 값과 같은 modulo 값을 가진 이전 prefix 위치들이 현재 위치와 함께 "interesting subarray"를 형성합니다.
