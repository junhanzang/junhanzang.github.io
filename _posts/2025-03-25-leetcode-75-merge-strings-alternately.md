---
title: "LeetCode 75 - Merge Strings Alternately"
date: 2025-03-25 22:35:24
categories:
  - 알고리즘 공부
---

You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with word1. If a string is longer than the other, append the additional letters onto the end of the merged string.

Return the merged string.

**Example 1:**

```
Input: word1 = "abc", word2 = "pqr"
Output: "apbqcr"
Explanation: The merged string will be merged as so:
word1:  a   b   c
word2:    p   q   r
merged: a p b q c r
```

**Example 2:**

```
Input: word1 = "ab", word2 = "pqrs"
Output: "apbqrs"
Explanation: Notice that as word2 is longer, "rs" is appended to the end.
word1:  a   b 
word2:    p   q   r   s
merged: a p b q   r   s
```

**Example 3:**

```
Input: word1 = "abcd", word2 = "pq"
Output: "apbqcd"
Explanation: Notice that as word1 is longer, "cd" is appended to the end.
word1:  a   b   c   d
word2:    p   q 
merged: a p b q c   d
```

**Constraints:**

- 1 <= word1.length, word2.length <= 100
- word1 and word2 consist of lowercase English letters.

<https://leetcode.com/studyplan/leetcode-75/>

```
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        # check len
        a = len(word1) - len(word2)
        answer = ""
        if a < 0:
            for i in range(len(word2)):
                if i < len(word1):
                    answer += word1[i]
                    answer += word2[i]
                else :
                    #answer += " "
                    answer += word2[i]
        elif a > 0:
            for i in range(len(word1)):
                if i < len(word2):
                    answer += word1[i]
                    answer += word2[i]
                else :
                    answer += word1[i]
                    #answer += " "
        else:
            for i in range(len(word1)):
                answer += word1[i]
                answer += word2[i]

        return answer
```
