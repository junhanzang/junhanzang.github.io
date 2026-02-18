---
title: "14. Longest Common Prefix"
date: 2025-03-25 22:36:59
categories:
  - 알고리즘 공부
---

<https://leetcode.com/problemset/>

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

**Example 1:**

```
Input: strs = ["flower","flow","flight"]
Output: "fl"
```

**Example 2:**

```
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
```

**Constraints:**

- 1 <= strs.length <= 200
- 0 <= strs[i].length <= 200
- strs[i] consists of only lowercase English letters if it is non-empty.

```
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) <= 0:
            return ""

        shortest = min(strs, key=len)

        for i, char in enumerate(shortest):
            for other in strs:
                if other[i] != char:
                    return shortest[:i]

        return shortest
```
