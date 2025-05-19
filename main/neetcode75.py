import collections

'''
1. Two Sum
Given an array of integers nums and an integer target, return the indices i and j such that nums[i] + nums[j] == target and i != j.

You may assume that every input has exactly one pair of indices i and j that satisfy the condition.
Return the answer with the smaller index first.
'''
def two_sum_return_indexes(nums, target):
  hash_map = {}

  for i, num in enumerate(nums):
      diff = target - num
      if diff in hash_map:
          return hash_map[diff], i
      else:
          hash_map[num] = i
  
  return

def two_sum_return_all_index_pairs(nums, target):
  hash_map = {}
  result = []
  for i, num in enumerate(nums):
    diff = target - num
    if diff in hash_map:
      result.append([hash_map[diff], i])
    if num not in hash_map:
      hash_map[num] = i
  
  return result

def two_sum_return_values(nums, target):
  pool = []

  for num in nums:
    diff = target - num
    if diff in pool:
      return diff, num
    else:
      pool.append(num)
  
  return

'''
53. Valid Parentheses
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
'''
closeParens = {')':'(',
                '}':'{',
                ']':'['}

def valid_parentheses(parens):
  stack = []
  for paren in parens:
    if paren in closeParens:
      if stack == []:
        return False
      elif stack.pop() != closeParens[paren]: 
        return False
    else:
      stack.append(paren)

  return stack == []

'''
2. Sliding Window: Best Time to Buy and Sell Stock

neetcode75.maxProfit([])          -> None
neetcode75.maxProfit([7,6,4,3,1]) -> 0
neetcode75.maxProfit([3,7,5,2,4]) -> 4
'''
def maxProfit(prices):
  if prices == []: return

  lowest, maxP = prices[0], 0

  for i in range(len(prices)):
    if prices[i] < lowest:
      lowest = prices[i]
    else:
      maxP = max(maxP, (prices[i] - lowest))
  
  return maxP

'''
3. Contains Duplicate

neetcode75.isDuplicate([])      -> False
neetcode75.isDuplicate([1,4,1]) -> True
neetcode75.isDuplicate([5,3,6]) -> False
'''
def isDuplicate(nums):
  mySet = set()

  for num in nums:
    if num in mySet:
      return True
    else:
      mySet.add(num)

  return False

'''
4. Product of Array Except Self

neetcode75.productExceptSelf([])        -> None
neetcode75.productExceptSelf([2,3,4,5]) -> [60,40,30,24]
'''
def productExceptSelf(nums):
  res = len(nums) * [1]
  prefix, postfix = 1, 1

  for i in range(1,len(nums)):
    prefix *= nums[i-1]
    res[i] = prefix
  
  for i in reversed(range(0,len(nums)-1)):
    postfix *= nums[i+1]
    res[i] *= postfix

  return res

'''
5. Maximum Sum Sub-array

neetcode75.maxSubArray([])                    -> 0
neetcode75.maxSubArray([-2,3,-1,4,-10,2,3,4]) -> 9
neetcode75.maxSubArray([-2,-1])               -> -1
'''
def maxSubArray(nums):
  if nums == []: return 0

  maxS, curS= nums[0], 0

  for num in nums:
    if curS < 0:
      curS = 0
    curS += num
    maxS = max(maxS, curS)

  return maxS

'''
6. Maximum Product Sub-array

neetcode75.maxProduct([2,3,-2,4]) -> 6
neetcode75.maxProduct([-2,0,-1])  -> 0
neetcode75.maxProduct([3,-1,4])   -> 4
neetcode75.maxProduct([-2,-1,-3]) -> 3
'''
def maxProduct(nums):
  res = max(nums)
  curMin, curMax = 1, 1

  for num in nums:
    tmpMax, tmpMin = curMax * num, curMin * num
    curMax, curMin = max(tmpMax, tmpMin, num), min(tmpMax, tmpMin, num)
    res = max(res, curMax)

  return res

'''
7. Find Minimum in Rotated Sorted Array

neetcode75.findMin([3,4,5,1,2])     -> 1
neetcode75.findMin([4,5,6,7,0,1,2]) -> 0
neetcode75.findMin([11,13,15,17])   -> 11
'''

def findTarget(nums, target): # binary search
  l, r = 0, len(nums)

  while(l != r):
    mid = (r - l)//2 + l
    if target <= nums[mid]:
      r = mid
    else:
      l = mid + 1

  return l # return index


def findMin(nums):
  l, r = 0, len(nums)-1

  while(l != r):
    mid = (r - l)//2 + l
    if nums[l] <= nums[mid]: 
      if nums[l] <= nums[r]:  # no rotate
        r = mid
      else:                   # rotated
        l = mid + 1
    else:
      r = mid
  
  return nums[l] # return value

'''
8. Search in Rotated Sorted Array

neetcode75.findTarget1([3,4,5,1,2], 2)    -> 4
neetcode75.findTarget1([11,13,15,17], 13) -> 1
neetcode75.findTarget1([4,5,1,2,3], 5)    -> 1
'''
def findTarget1(nums, target):
  l, r = 0, len(nums)-1

  while l != r:
    mid = (r - l)//2 + l
    if nums[l] <= nums[mid]: # left in order
      if nums[l] <= target <= nums[mid]: # within l and mid
        r = mid
      else:
        l = mid + 1
    else: # right in order
      if nums[mid] < target <= nums[r]: # within mid and r
        l = mid + 1
      else:
        r = mid
  
  return l

'''
9. 3Sum
  
nums[i] + nums[j] + nums[k] == 0
i != j, i != k, and j != k

neetcode75.threeSum([-1,0,1,2,-1,-4]) -> [[-1,-1,2],[-1,0,1]]
neetcode75.threeSum([0,1,1])          -> []
neetcode75.threeSum([0,0,0])          -> [[0,0,0]]
neetcode75.threeSum([-1,0,1,0])       -> 
'''
def threeSum(nums):
  res = set()
  for i, num in enumerate(nums):
    diff = -num
    restArr = nums[i+1:]
    pairs = twoSumPairs1(restArr, diff) 
    for pair in pairs:
      newPair = tuple(sorted([num, pair[0], pair[1]]))
      res.add(newPair)

  return res

'''
10. Container With Most Water

neetcode75.maxArea([1,8,6,2,5,4,8,3,7]) -> 49
neetcode75.maxArea([1,1]) -> 1
'''
def maxArea(nums):
  l, r = 0, len(nums)-1
  maxA = 0
  while(l != r):
    maxA = max(maxA, (r-l)*min(nums[l], nums[r]))
    if nums[l] < nums[r]:
      l += 1
    else:
      r -= 1

  return maxA

'''
11. Number of 1 Bits

neetcode75.hammingWeight(0b00000000000000000000000000001011) -> 3
'''
def hammingWeight(n):
  res = 0
  while(n != 0):
    res += n % 2
    n = n >> 1
  
  return res

'''
12. Counting Bits

neetcode75.countBits(5) -> [0,1,1,2,1,2]
'''
def countBits(n):
  res, offset, nextOffset = [0]*(n+1), 1, 2
  
  for i in range(n+1): 
    if i == 0: continue
    elif i < nextOffset: 
      res[i] = 1 + res[i-offset]
    else:
      offset *= 2
      nextOffset *= 2
      res[i] = 1 + res[i-offset]
  
  return res

'''
13. Missing Number

neetcode75.missingNumber([9,6,4,2,3,5,7,0,1]) -> 8
'''
def missingNumber(nums):
  res = 0
  for i in range(len(nums)+1):
    res = res ^ i # ^ is the XOR in Python
  
  for num in nums:
    res = res ^ num
  
  return res

'''
14. Reverse Bits

neetcode75.reverseBits(0b00000010100101000001111010011100) -> 964176192
neetcode75.reverseBits(0b00000000000000000000000000000100) -> 1
'''
def reverseBits(n):
  res = 0
  count = 0
  while(count < 32):
    bit = (n >> count) & 1 
    res = bit << (31 - count) | res
    count += 1
  
  return res

'''
15. Climbing Stairs

neetcode75.climbStairs(3)     -> 3
neetcode75.climbStairs(4)     -> 5
neetcode75.climbStairsMemo(3) -> 3
neetcode75.climbStairsMemo(4) -> 5
'''
def climbStairs(n):
  if n == 0: return 1
  elif n == 1: return 1

  return climbStairs(n-2) + climbStairs(n-1)

def climbStairsMemo(n):
  res = [1, 2]

  if n == 1: return 1
  elif n == 2: return 2

  for i in range(2, n):
    res.append(res[i-2] + res[i-1])

  return res[n-1]

'''
16. Coin Change

neetcode75.coinChangeCount([1], 0)         -> 0
neetcode75.coinChangeCount([1,2,5], 11)    -> 3
neetcode75.coinChangeCount([1,3,4,5], 7)   -> 2
neetcode75.coinChangeValues([7,5,3,2], 34) -> [3,3,7,7,7,7]
neetcode75.coinChangeCountTD([1,3,4,5], 7) -> 2
neetcode75.coinChangeCountTD([1], 0)       -> 0
'''
def coinChangeCount(coins, amount):
  res = [0]

  # bottom up solution, solving smaller solution first
  # solve DP[1], then DP[2], ..., final to DP[amount]
  for smallAmount in range(1, amount+1):
    numCoins = []
    for coin in coins:
      diff = smallAmount - coin
      if (diff == 0):
        numCoins.append(1)
      elif (diff > 0 and res[diff] != -1):
        numCoins.append(1 + res[diff])
  
    if numCoins: 
      res.append(min(numCoins))
    else:
      res.append(-1)
  
  return res.pop()

def coinChangeValues(coins, amount):
  res = [[]]

  for smallAmount in range(1, amount+1):
    changes = []
    for coin in coins:
      diff = smallAmount - coin
      if (diff == 0):
        changes.append([coin])
      elif (diff > 0 and res[diff]):
        coinList = res[diff].copy()
        coinList.append(coin)
        changes.append(coinList)
    
    if changes:
      res.append(min(changes, key=lambda x: len(x)))
    else:
      res.append([])
    
  return res.pop()

# Top down solution with cache
def coinChangeCountTD(coins, amount):
  cache = {}
  if amount == 0: return 0

  def dfs(coins, amount):
    changes = []
    for coin in coins:
      diff = amount - coin
      if diff > 0:
        change = 0
        if diff not in cache:
          change = dfs(coins, diff)
          cache[diff] = change
        else:
          change = cache[diff]
        if change != -1:
          changes.append(change + 1)
      elif diff == 0:
        changes.append(1)
    
    if changes:
      return min(changes)
    else:
      return -1

  return dfs(coins, amount)

'''
17. Longest Increasing Subsequence

neetcode75.lengthOfLIS([1,2,4,3]) -> 3 
neetcode75.lengthOfLIS([10,9,2,5,3,7,101,18]) -> 4
neetcode75.lengthOfLIS([4,10,4,3,8,9]) -> 3
neetcode75.longIncSubseq([4,10,4,3,8,9]) -> 
  [[4, 8, 9], [10], [4, 8, 9], [3, 8, 9], [8, 9], [9]]
'''
def lengthOfLIS(nums):
  res = [1 for x in range(len(nums))]
  for i in reversed(range(len(nums))):
      for j in range(i+1, len(nums)):
        if(nums[i] < nums[j]):
          res[i] = max(res[i], 1+(res[j]))

  return max(res)

def longIncSubseq(nums):
  res = [[] for x in range(len(nums))]

  for i in reversed(range(len(nums))):
    lstSeq = [[nums[i]]]
    for j in range(i+1, len(nums)):
      if(nums[i] < nums[j]):
        lstSeq.append([nums[i]]+(res[j]))
    maxLen = max(lstSeq, key=lambda x: len(x))
    res[i] = res[i] + maxLen

  return res

'''
18. Longest Common Subsequence

neetcode75.longComSubseq("adcde", "ade") = 3
'''
def longComSubseq(text1, text2):
  r, c = len(text1), len(text2)
  matrix = [[0 for x in range(c)] for y in range(r)]

  for i in reversed(range(r)):
    for j in reversed(range(c)):
      if text1[i] == text2[j]:
        diaVal = 0
        if(i+1 < r) and (j+1 < c): # in matrix
          diaVal = matrix[i+1][j+1] 
        matrix[i][j] = 1 + diaVal
      else:
        rightVal, downVal = 0, 0
        if(i+1 < r): # in matrix row
          downVal = matrix[i+1][j]
        if(j+1 < c): # in matrix column
          rightVal = matrix[i][j+1]
        matrix[i][j] = max(rightVal, downVal)

  return matrix[0][0]

'''
19. Word Break

neetcode75.wordBreak("leetcode", ["leet","code"]) = true
neetcode75.wordBreak("applepenapple", ["apple","pen"]) = true
neetcode75.wordBreak("catsandog", ["cats","dog","sand","and","cat"]) = false
neetcode75.wordBreak1("cars", ["car","ca","rs"]) = true
'''
# my version, two pointers
def wordBreak(s, wordDict):
  res = [False]*len(s)
  res.append(True)

  for i in reversed(range(len(s))):
    for j in range(i, len(s)):
      if(s[i:j+1] in wordDict and res [j+1]):
        res[i] = True
        break

  return res[0]

# neetcode version, loop the dictionary 
def wordBreak1(s, wordDict):
  res = [False]*len(s)
  res.append(True)

  for i in reversed(range(len(s))):
    for w in wordDict:
      # print(i+len(w), len(s), s[i:i+len(w)], w)
      if(i+len(w) <= len(s) and s[i:i+len(w)] == w):
        if(res[i+len(w)]):
          res[i] = True
          break
        else:
          res[i] = False 

  return res[0]

'''
20. Combination Sum

neetcode75.combinationSum([2,3,6,7], 7) -> [[2,2,3],[7]]
'''
def combinationSum(candidates, target):
  res = []

  def dfs(i, cur):
    total = sum(cur)
    if total == target: 
      res.append(cur)
      return

    if len(cur) > 150 or i>= len(candidates) or  total > target: return

    cur.append(candidates[i])
    dfs(i, cur.copy())
    cur.pop()
    dfs(i+1, cur.copy())

  dfs(0,[])

  return res

def combinationSum1(candidates, target):
  res = []

  def dfs(i, cur):
    total = sum(cur)
    if total == target: 
      res.append(cur)
      return
    if i>= len(candidates) or total > target: return

    for idx in range(i, len(candidates)):
      cur.append(candidates[idx])
      dfs(idx, cur.copy())
      cur.pop() # pop cur once its dfs done, so we move to the next candidate

  dfs(0,[])

  return res

'''
21. House Robber

neetcode75.rob([1,2,3,1])   -> 4
neetcode75.rob([2,7,9,3,1]) -> 12
'''
def rob(nums):
  res = [0]*(len(nums)+2)
  for i in reversed(range(0, len(nums))):
    res[i] = max(res[i+1], nums[i] + res[i+2])

  return res[0]

'''
22. House Robber II

neetcode75.rob2([2,3,2])   -> 4
neetcode75.rob2([1,2,3,1]) -> 4
'''

def rob2(nums):
  if len(nums) == 1:
    return nums[0]

  res1 = rob(nums[1:])
  res2 = rob(nums[0:(len(nums)-1)])

  return max(res1, res2)

'''
23. Decode Ways

neetcode75.numDecodings("1123")   -> 5
neetcode75.numDecodings("10")     -> 1
neetcode75.numDecodings("120123") -> 3
neetcode75.numDecodings("06")     -> 0
'''
def numDecodings(s):
  size = len(s)
  res = [0]*(size)
  res.append(1)

  for i in reversed(range(size)):
    if s[i] == "0":
      continue
    else:
      res[i] = res[i+1]

    if i+1<size and int(s[i:i+2]) <= 26:
      res[i] += res[i+2]

  return res[0]

def numDecodings1(s): # recursion version with cache
  dp = {len(s): 1}

  def dfs(i):
    if i in dp: 
      return dp[i]
    
    if s[i] == "0":
      return 0

    res = dfs(i+1)
    if i+1<len(s) and int(s[i:i+2]) <= 26:
      res += dfs(i+2)
    dp[i] = res
    return res

  return dfs(0)

'''
24. Unique Paths

neetcode75.uniquePaths(3, 7) -> 28
'''
def uniquePaths(m, n):
  mat = [[0 for j in range(n)] for i in range(m)]
  mat[0][0] = 1

  for i in range(m):
    for j in range(n):
      if i > 0:
        mat[i][j] += mat[i-1][j]
      if j > 0:
        mat[i][j] += mat[i][j-1]
  
  return mat[m-1][n-1]

'''
25. Jump Game

neetcode75.canJump([2,3,1,1,4]) -> True
'''
def canJump(nums):
  res = [False]*len(nums)
  res[-1] = True

  for idx in reversed(range(len(nums))):
    for j in reversed(range(1, nums[idx]+1)): # extra time complexity
      if idx+j < len(nums):
        res[idx] = res[idx+j] or res[idx]
      if res[idx]: break
  
  return res[0]

# keep track the goal
def canJumpNtime(nums):
  goal = len(nums)-1

  for idx in reversed(range(len(nums))):
    if idx + nums[idx] >= goal:
      goal = idx

  return goal == 0

'''
26. Clone Graph (need review)
'''
def cloneGraph(node):
  oldToNew = {}
  
  def clone(node):
    if node in oldToNew:
      return oldToNew[node]

    copy = Node(node.val)
    oldToNew[node] = copy

    for nei in node.neighbors:
      copy.neighbors.append(clone(nei))
    
    return copy

  return clone(node)

'''
27. Course Schedule

neetcode75.canFinish(2, [[1,0], [0,1]]) -> False
neetcode75.canFinish(3, [[0,1],[0,2]])  -> True
neetcode75.canFinish(3, [[0,1],[1,2]]) -> True
neetcode75.canFinish(3, [[1,0],[2,1],[1,2]]) -> False
'''
def canFinish(numCourses, prerequisites):
  hashMap = {course: [] for course in range(numCourses)}
  for num, pre in prerequisites:
    hashMap[num].append(pre)

  visited = set()
  def dfs(num):
    if num in visited: 
      return False

    if hashMap[num] == []:
      return True
    
    visited.add(num)
    for pre in hashMap[num]:
        if not dfs(pre): return False

    visited.remove(num)
    hashMap[num] = []
    return True

  for num in hashMap:
    if not dfs(num): return False

  return True

def isCycleGraph(numCourses, prerequisites):
  hashMap = {course: [] for course in range(numCourses)}
  for num, pre in prerequisites:
    hashMap[num].append(pre)

  visited = set()
  def dfs(num):
    adj = hashMap[num]
    if adj == []: return False

    for pre in adj:
      if pre in visited: 
        return True
      else:
        visited.add(pre)
        if dfs(pre): return True
        visited.remove(pre)
    hashMap[num] = []
    return False  

  for num in hashMap:
    if dfs(num): return True

  return False
      
'''
28. Pacific Atlantic Water Flow

neetcode75.twoSides()
'''
def twoSides():
  heights = [[1,2,2,3,5], 
             [3,2,3,4,4], 
             [2,4,5,3,1], 
             [6,7,1,4,5], 
             [5,1,1,2,4]]

  rows, columns = len(heights), len(heights[0])
  pac, alt = set(), set()
  
  def dfs(r, c, visited, preHeight):
    if r >= rows or r < 0 or c >= columns or c < 0 or \
       (r,c) in visited or \
       heights[r][c] < preHeight:
      return

    visited.add((r,c))
    dfs(r-1, c, visited, heights[r][c])
    dfs(r+1, c, visited, heights[r][c])
    dfs(r, c-1, visited, heights[r][c])
    dfs(r, c+1, visited, heights[r][c])

  for c in range(columns):
    dfs(0, c, pac, heights[0][c])
    dfs(rows-1, c, alt, heights[rows-1][c])

  for r in range(rows):
    dfs(r, 0, pac, heights[r][0])
    dfs(r, columns-1, alt, heights[r][columns-1])

  # print(pac, alt)
  return list(map(lambda x: list(x), pac.intersection(alt)))

'''
29. Numbers of Islands

neetcode75.numIslands()
'''
def numIslands(): # dfs
  grid = [["1","1","0","0","0"],
          ["1","1","0","0","0"],
          ["0","0","1","0","0"],
          ["0","0","0","1","1"]]
  rows = len(grid)
  columns = len(grid[0])

  visited = set()
  def dfs(r, c):
    if r >= rows or r < 0 or c >= columns or c < 0 or \
       (r,c) in visited or \
       grid[r][c] != "1":
      return

    visited.add((r,c))
    dfs(r-1, c)
    dfs(r+1, c)
    dfs(r, c-1)
    dfs(r, c+1)
  
  res = 0
  for r in range(rows):
    for c in range(columns):
      if grid[r][c] == "1" and (r,c) not in visited:
        dfs(r, c)
        res += 1

  return res

def numIslands2(): # bfs
  grid = [["1","1","0","0","0"],
          ["1","1","0","0","0"],
          ["0","0","1","0","0"],
          ["0","0","0","1","1"]]
  rows = len(grid)
  columns = len(grid[0])

  visited = set()
  def bfs(r, c):
    myStack = [(r,c)]

    while myStack != []:
      r, c = myStack.pop()
      if r >= rows or r < 0 or c >= columns or c < 0 or \
         (r,c) in visited or \
         grid[r][c] != "1":
            continue
      visited.add((r,c))
      myStack.append((r-1, c))
      myStack.append((r+1, c))
      myStack.append((r, c-1))
      myStack.append((r, c+1))
  
  res = 0
  for r in range(rows):
    for c in range(columns):
      if grid[r][c] == "1" and (r,c) not in visited:
        bfs(r, c)
        res += 1

  return res

'''
30. Longest Consecutive Sequence

neetcode75.longestConsecutive([100,4,200,1,3,2])
'''
def longestConsecutive(nums):
  setNums = set(nums)
  maxL = 0

  for num in setNums:
    if (num-1) in setNums:
      continue
    else:
      length = 0
      head = num
      while(head in setNums):
        head += 1
        length += 1
      maxL = max(maxL, length)

  return maxL
'''
Extra: Count Square Sub-matrices with All Ones

neetcode75.countSquares() -> 15
'''
def countSquares():
  grid = [[0,1,1,1],
          [1,1,1,1],
          [0,1,1,1]]
  # grid = [[1,1,1],
  #         [1,1,1]]
  rows, columns = len(grid), len(grid[0])

  def bfs(r, c):
    layer = 0
    que = collections.deque()
    que.append([(r,c)])

    while que:
      batch = set()
      outlayer = que.popleft()
      for r,c in outlayer:
        if r >= rows or c >= columns or grid[r][c] != 1:
          return layer

        if (r+1,c) not in outlayer:
          batch.add((r+1,c))
        if (r,c+1) not in outlayer:
          batch.add((r,c+1))
        batch.add((r+1,c+1))
      que.append(list(batch))
      layer += 1

    return layer
  
  res = 0
  for r in range(rows):
    for c in range(columns):
      if grid[r][c] == 1:
        layer = bfs(r,c)
        res += layer

  return res

'''
31. Alien Dictionary: topological ordering

neetcode75.alienOrder(["A", "BA", "BC", "C"])
neetcode75.alienOrder(["ABC", "ACDE"])
'''
# assume they give use a valid dictionary
def alienOrder(words):
  adj = {c: set() for w in words for c in w}
  
  for i in range(len(words)-1):
    w1, w2 = words[i], words[i+1]
    lenMin = min(len(w1), len(w2))
    for j in range(lenMin):
      if w1[j] != w2[j]:
        adj[w1[j]].add(w2[j])
        break

  onPath = [] # check the graph loop
  visited = []
  def dfs(c):
    if c in onPath: 
      return True

    if c in visited: 
      return False
    
    onPath.append(c)
    for nei in adj[c]:
      if dfs(nei):
        return True
    onPath.remove(c)
    visited.append(c)

  for c in adj:
    if dfs(c):
      return ""

  visited.reverse()
  return visited

'''
32. Graph Valid Tree

neetcode75.isValidTree(5, [[0, 1], [0, 2], [0, 3], [1, 4]]) -> True
neetcode75.isValidTree(2, []) -> False
neetcode75.isValidTree(1, []) -> True
'''
def isValidTree(n, edges):

  hashMap = {num: [] for num in range(n)}
  for f, s in edges:
    hashMap[f].append(s)
    hashMap[s].append(f)

  visited = set()
  def dfs(num, pre): #(if the graph has a loop)
    if num in visited:
      return True

    visited.add(num)
    adjs = hashMap[num]        
    for adj in adjs:
      if adj == pre: continue

      if dfs(adj, num): return True

    return False  

  return not dfs(0, -1) and (len(visited) == n)

'''
33. Number of Connected Components in an Undirected Graph
see leetcode 547

neetcode75.countComponents(5, [[0,1],[1,2],[3,4]]) -> 2
neetcode75.countComponents(6, [[0,1],[1,2],[0,2],[3,4]]) -> 3
neetcode75.unionFind(5, [[0,1],[1,2],[3,4],[2,3],[3,4]]) -> 2
neetcode75.unionFind(6, [[0,1],[1,2],[0,2],[3,4]]) -> 3
neetcode75.unionFind(4, [[1,2],[2,1],[0,3],[3,0],[2,3],[3,2]]) -> 1
neetcode75.unionFind(4, [[0,3],[1,2],[2,1],[2,3],[3,0],[3,2]]) -> 1
'''
# dfs version O(v+e)
def countComponents(n, edges):
  hashMap = {node: [] for node in range(n)}
  for f, s in edges:
    hashMap[f].append(s)
    hashMap[s].append(f)
  
  visited = set()
  def dfs(n):
    if n in visited: return
    visited.add(n)
    if n not in hashMap: return
    adjs = hashMap[n]
    for adj in adjs:
      dfs(adj)

  res = 0
  for node in hashMap:
    if node not in visited:
      res +=1
      dfs(node)

  return res

# union find version
# neetcode75.unionFind(4, [[0,3],[1,2],[2,1],[2,3],[3,0],[3,2]]) -> 1
def unionFind(n, edges):
  par = [i for i in range(n)] # initialize self parent nodes
  # rank = [1] * n

  # find its parent node
  def find(n1):
    res = n1
    while res != par[res]: # res == par[res] means itself is a parent node
      res = par[res]
    return res

  def union(n1, n2):
    p1, p2 = find(n1), find(n2)
    if p1 == p2: return 0

    par[p2] = p1 # union happen on the root

    return 1
  
  res = n
  for n1, n2 in edges:
    u = union(n1, n2)
    res -= u
  
  return res

# neetcode75.findNumProv([[1,1,0],[1,1,0],[0,0,1]]) -> 2
# neetcode75.findNumProv([[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]]) -> 1
def findNumProv(matrix):
  par = [i for i in range(len(matrix))]

  def find(n):
    res = n
    while res != par[res]:
      res = par[res]
    return res
  
  def union(n1, n2):
    p1, p2 = find(n1), find(n2)
    if p1 == p2: return 0
    par[p2] = p1
    return 1
  
  res = len(matrix)
  for i in range(len(matrix)):
    for j in range(len(matrix)):
      if matrix[i][j] and i != j:
        u = union(i, j)
        res -= u
  
  return res

'''
34. Insert Interval

neetcode75.insert([[1,3],[6,9]], [2,5]) -> [[1, 5], [6, 9]]
neetcode75.insert([[1,3],[6,9]], [3,6]) -> [[1, 9]]
'''
def insert(intervals, newInterval):
  insertS, insertE = newInterval
  res = []
  for i in range(len(intervals)):
    s, e = intervals[i]
    if insertE < s: 
      res.append([insertS, insertE])
      res.extend(intervals[i:])
      return res
    elif e < insertS: res.append([s,e])
    else:
      insertS = min(insertS, s)
      insertE = max(insertE, e)
  
  res.append([insertS, insertE])
  return res
  
'''
35. Merge Intervals

neetcode75.merge([[1,3],[2,6],[8,10],[15,18]]) -> [[1,6],[8,10],[15,18]]
neetcode75.merge([[2,3],[2,2],[3,3],[1,3],[5,7],[2,2],[4,6]])
neetcode75.merge([[1,3],[4,6],[5,7]])
'''
def merge(intervals):
  res = []
  intervals.sort(key=lambda x: x[0]) # O(nlogn)
  newIntervalz = None
  for s, e in intervals:
    if newInterval == None:
      newInterval = [s, e]
      continue
    insertS, insertE = newInterval

    # if overlap
    if (s >= insertS and s<=insertE) or (e >= insertS and e <=insertE):
      newInterval = [min(insertS, s), max(insertE, e)]
    else: # not overlap
      res.append(newInterval)
      newInterval = [s, e]

  if newInterval: # append leftover
    res.append(newInterval)
  
  return res

'''
36. Non-overlapping Intervals

neetcode75.eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]]) -> 1
'''
def eraseOverlapIntervals(intervals):
  # sorted by the start point
  intervals.sort(key=lambda x: x[0])
  #[1,2],[1,3],[2,3],[3,4]

  res = 0
  lastInterval = intervals[0]
  for s, e in intervals[1:]:
    lastEnd = lastInterval[1]
    if lastEnd > s:
      # over lap
      res += 1
      if e < lastEnd:
        lastInterval = [s, e]
    else:
      # not overlap
      lastInterval = [s, e]

  return res

'''
37. Meeting Rooms

neetcode75.canAttendMeetings([(0,30), (5,10), (15,20)]) -> False
'''
def canAttendMeetings(intervals):
  intervals.sort(key=lambda x: x[0])
  lastInterval = intervals[0]

  for s, e in intervals[1:]:
    lastEnd = lastInterval[1]
    if lastEnd > s:
      # overlap
      return False
  
  return True

'''
38. Meeting Rooms II

neetcode75.minMeetingRooms([(0,30), (5,10), (15,20)]) -> 2
neetcode75.minMeetingRooms([(1,3), (3,4)]) -> 1
'''
def minMeetingRooms(intervals):
  intervals.sort(key=lambda x: x[0])
  res = []

  for s, e in intervals:

    notOverlap = False
    for i in range(len(res)):
      existS, existE = res[i]
      if (s <= existS and e <= existS) or (s >= existE and e >= existS):
        # not overlap
        res[i] = (s,e)
        notOverlap = True
        break

    if not notOverlap:
      res.append((s,e))

  return len(res)

# try two pointers neetcode solution

'''
39. Reverse Linked List

neetcode75.testNode(): 1 -> 2
neetcode75.reverseListRec(neetcode75.testNode())
'''
class ListNode:
  def __init__(self, val=0, next=None):
    self.val = val
    self.next = next

def reverseList(head):
  cur = head
  res = None
  while cur != None:
    nxt = cur.next
    cur.next = res
    res = cur
    cur = nxt
  return res

# not straight forward
def reverseListRec(head):
  if head == None:
    return None

  newHead = head
  if head.next != None:
    newHead = reverseListRec(head.next)
    head.next.next = head
  head.next = None

  return newHead

# 1 -> 2
def testNode():
  two = ListNode(2)
  one = ListNode(1, two)

  return one

def printNode(head):
  while head:
    print(head.val)
    head = head.next

'''
40. Linked List Cycle
'''
# Floyd's Tortoise & Hare algorithm, they will meet (at the same point)
def hasCycle(head):
  slow, fast = head, head

  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
      return True

  return False

'''
41. Merge Two Sorted Lists
'''
def mergeTwoLists(l1, l2):
  dummy = ListNode()
  p = dummy

  while l1 and l2:
    if l1.val < l2.val:
      p.next = ListNode(l1.val)
      l1 = l1.next
      p = p.next
    else:
      p.next = ListNode(l2.val)
      l2 = l2.next
      p = p.next

  if l1: 
    p.next = l1

  if l2:
    p.next = l2

  return dummy.next

'''
42. Merge K Sorted Lists

neetcode75.mergeKLists(neetcode75.test)
'''
def mergeKLists(lists):
  if lists == []: return None

  while len(lists) > 1:
    res = []
    for i in range(0, len(lists), 2):
      l1 = lists[i]
      l2 = lists[i+1] if i+1 < len(lists) else None
      res.append(mergeTwoLists(l1, l2))
    lists = res 
  
  printNode(lists[0])

  return lists[0]

'''
43. Remove Nth Node From End of List

neetcode75.removeNthFromEnd(neetcode75.test, 2)
neetcode75.removeNthFromEnd(neetcode75.test1, 1)
'''
test =\
  ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))) 
test1 = ListNode(1) 

def removeNthFromEnd(head, n):
  dummy = ListNode(0, head)
  left = dummy
  right = head

  for i in range(n):
    if right: 
      right = right.next
    else:
      break

  while right:
    left = left.next
    right = right.next
  
  left.next = left.next.next

  return dummy.next

'''
44. Reorder List
    L0 → L1 → … → Ln - 1 → Ln
    L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

neetcode75.reorderList(neetcode75.test)
'''
def reorderList(head):
  if head == None: return head

  slow, fast = head, head.next

  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
  
  second = slow.next
  slow.next = None
  reversedList = reverseList(second)
  second = reversedList

  first = head
  while second:
    temp1, temp2 = first.next, second.next
    first.next = second
    second.next = temp1
    first, second = temp1, temp2
  
  printNode(head)

'''
45. Set Matrix Zeroes

neetcode75.setZeroes([[1,1,1],[1,0,1],[1,1,1]]) -> 
  [[1,0,1],[0,0,0],[1,0,1]]
'''
def setZeroes(matrix):
  rowZero = 1

  for i in range(len(matrix)):
    for j in range(len(matrix[i])):
      if matrix[i][j] == 0:
        matrix[0][j] = 0
        if i == 0:
          rowZero = 0
        else:
          matrix[i][0] = 0
  
  for i in range(1, len(matrix)):
    if matrix[i][0] == 0:
      matrix[i] = [0]*len(matrix[i])

  for j in range(1, len(matrix[0])):
    if matrix[0][j] == 0:
      for i in range(len(matrix)):
        matrix[i][j] = 0

  if matrix[0][0] == 0:
    for i in range(len(matrix)):
        matrix[i][0] = 0

  if rowZero == 0:
    matrix[0] = [0]*len(matrix[i])
    
  return matrix

'''
46. Spiral Matrix

neetcode75.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]) 
  -> [1,2,3,6,9,8,7,4,5]
neetcode75.spiralOrder([[1,2],[3,4],[5,6]]) ->
neetcode75.spiralOrder([[1], [2]]) ->
'''
def spiralOrder(matrix):
  rowMax, colMax = len(matrix), len(matrix[0])
  visited = [[False]*colMax for i in range(rowMax)]
  visited[0][0] = True
  res = [matrix[0][0]]
  r, c = (0,0)

  def loop(position):
    r, c = position
    resOneLoop = []

    while c < colMax and visited[r][c] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      c += 1

    c -= 1
    if r+1 < rowMax:
      r += 1

    while r < rowMax and visited[r][c] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      r += 1

    r -= 1
    if c-1 >= 0:
      c -= 1

    while c >= 0 and visited[r][c] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      c -= 1
    
    c += 1
    if r-1 >= 0:
      r -= 1
    while r >= 0 and visited[r][c] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      r -= 1
    
    res.extend(resOneLoop)

    return (r+1,c)

  if colMax == 1:
    res = []
    for lst in matrix: 
       res.extend(lst)
    return res

  while c+1 < colMax and visited[r][c+1] == False:
    r, c = loop((r,(c+1)))
  
  return res

def spiralOrder2(matrix):
  rowMax, colMax = len(matrix), len(matrix[0])
  visited = [[False]*colMax for i in range(rowMax)]
  res = []
  r, c = (0,0)

  def loop(position):
    r, c = position
    resOneLoop = []
    
    while c+1 < colMax and visited[r][c+1] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      c += 1

    while r+1 < rowMax and visited[r+1][c] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      r += 1

    while c-1 >= 0 and visited[r][c-1] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      c -= 1
    
    while r-1 >= 0 and visited[r-1][c] == False:
      resOneLoop.append(matrix[r][c])
      visited[r][c] = True
      r -= 1
    
    res.extend(resOneLoop)

    return (r,c)
  
  while c < colMax and visited[r][c] == False:
    newR, newC = loop((r,c))
    if (newR, newC) == (r,c):
      res.append(matrix[r][c])
      break
    else:
      r, c = newR, newC 

  return res

def spiralOrder3(matrix):
  top, down, left, right = 0, len(matrix), 0, len(matrix[0])
  res = []
  while left < right and top < down:
    for i in range(left, right):
      res.append(matrix[top][i])
    top += 1

    for i in range(top, down):
      res.append(matrix[i][right-1])
    right -= 1

    if not(left < right and top < down):
      break

    for i in reversed(range(left, right)):
      res.append(matrix[down-1][i])
    down -= 1

    for i in reversed(range(top, down)):
      res.append(matrix[i][left])
    left += 1

  return res

'''
47. Rotate Image

neetcode75.rotate([[1,2,3],[4,5,6],[7,8,9]]) 
  -> [[7,4,1],[8,5,2],[9,6,3]]
'''
def rotate(matrix):
  size = len(matrix) - 1
  start, end = 0, len(matrix)-1
  while start < end: 
    for i in range(start, end):
      # next1 = matrix[i][size-start]
      # next2 = matrix[size-start][size-i]
      # next3 = matrix[size-i][start]
      # next4 = matrix[start][i]

      #1
      store = matrix[i][size-start]
      matrix[i][size-start] = matrix[start][i]

      #2  
      temp = matrix[size-start][size-i]
      matrix[size-start][size-i] = store
      store = temp

      #3
      temp = matrix[size-i][start]
      matrix[size-i][start] = store
      store = temp

      #4
      matrix[start][i] = store
    
    start += 1
    end   -= 1

  return matrix

'''
48. Word Search

neetcode75.exist
([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED")
  -> True

neetcode75.exist(neetcode75.myBoard, neetcode75.words)
'''
myBoard = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
words = "ABCCED"

def exist(board, word):
  maxR, maxC, lenW = len(board), len(board[0]), len(word)
  visited = []

  def dfs(cord, i):
    row, col = cord
    if i >= lenW: return True
    if row < 0 or row >= maxR or col< 0 or col >= maxC: return False

    if(board[row][col] == words[i] and (cord not in visited)):
      visited.append(cord)
      if   dfs((row+1, col), i+1): return True
      elif dfs((row-1, col), i+1): return True
      elif dfs((row, col+1), i+1): return True
      elif dfs((row, col-1), i+1): return True
      else:
        visited.pop()
        return False
    else:
      return False

  for i in range(maxR):
    for j in range(maxC):
      visited = []
      if dfs((i,j),0):
        return True # visited is the path

  return False

def existNeet(board, words):
  # almost the same
  pass

'''

49. Longest Substring Without Repeating Characters (sliding window)

neetcode75.lenLongSubstr("dvdf") -> 3
'''
def lenLongSubstr(word):
  que, maxW = collections.deque(), 0

  for c in word:
    if c in que:
      maxW = max(maxW, len(que))

    while c in que:
      que.popleft()
    
    que.append(c)
  
  maxW = max(maxW, len(que))

  return maxW
