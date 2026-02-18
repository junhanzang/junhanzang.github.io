---
title: "4SUM problem"
date: 2025-03-26 12:43:03
categories:
  - 알고리즘 공부
---

<https://leetcode.com/problems/4sum/description/>

![](/assets/images/posts/528/tfile.svg)

![](/assets/images/posts/528/tfile_1.svg)

```
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()  # 결과 중복 제거를 위해 정렬
        n = len(nums)
        answer = []
        if n < 4:  # Not enough elements
            return answer
        for i in range(n - 3):
            # 중복된 첫 번째 요소 건너뛰기
            if i > 0 and nums[i] == nums[i-1]:
                continue
                
            for j in range(i + 1, n - 2):
                # 중복된 두 번째 요소 건너뛰기
                if j > i + 1 and nums[j] == nums[j-1]:
                    continue
                
                # 두 포인터 기법 사용 - O(n²)에서 O(n)으로 개선
                left = j + 1  # 세 번째 요소
                right = n - 1  # 네 번째 요소
                
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    
                    if current_sum == target:
                        answer.append([nums[i], nums[j], nums[left], nums[right]])
                        
                        # 중복 건너뛰기
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        
                        left += 1
                        right -= 1
                    
                    elif current_sum < target:
                        left += 1  # 합이 더 커야 하므로 왼쪽 포인터 증가
                    else:
                        right -= 1  # 합이 더 작아야 하므로 오른쪽 포인터 감소
        
        return answer
```
