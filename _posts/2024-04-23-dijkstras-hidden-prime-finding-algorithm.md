---
title: "Dijkstra's Hidden Prime Finding Algorithm"
date: 2024-04-23 23:11:12
categories:
  - 일상생활
tags:
  - Prime Finding Algorithm
---

<https://www.youtube.com/watch?app=desktop&v=fwxjMKBMR7s>

몰랐던걸 알았다...

[Dijkstra's Hidden Prime Finding Algorithm (fwxjMKBMR7s).srt

0.03MB](./file/Dijkstra's Hidden Prime Finding Algorithm (fwxjMKBMR7s).srt)

```
import heapq
import time

# dijkstraPrimes_heapq_2 함수 정의
def dijkstraPrimes_heapq_2(n):
    pool = [(4, 2)]
    primes = [2]
    for i in range(3, n):
        current_value, current_prime = heapq.heappop(pool)
        
        if current_value > i:
            heapq.heappush(pool, (i**2, i))
            primes.append(i)
        else:
            while current_value <= i:
                heapq.heappush(pool, (current_prime + current_value, current_prime))
                current_value, current_prime = heapq.heappop(pool)
                
        heapq.heappush(pool, (current_value, current_prime))
        
    return primes

# benchmark 함수 정의
def benchmark(func, *args, **kwargs):
    total_time = 0
    num_runs = 100  # 100번 실행하여 평균 시간 계산
    start_time = time.time()
    for _ in range(num_runs):
        func(*args, **kwargs)  # 주어진 함수 실행
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / num_runs
    return (average_time, total_time)

# dijkstraPrimes_heapq_2 함수의 실행 시간을 측정하기
average_time, total_time = benchmark(dijkstraPrimes_heapq_2, 1000)  # 1000까지 소수 찾기
print(f"100번 실행한 총 시간: {total_time:.6f} 초")
print(f"100번 실행한 평균 시간: {average_time:.6f} 초")
```

코드 수정 완료
