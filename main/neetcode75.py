# reload import, put it in the terminal
# from importlib import reload

'''
1. Two Sum
neetcode75.twoSum([2,7,11,15], 9)        -> [2, 7]
neetcode75.twoSumPairs([4,4,8,10,0,4], 8) -> {(2, 4), (5, 1), (4, 2)}
'''
def twoSumNeetcode(nums, target):
  hashMap = {}

  for i, num in enumerate(nums):
    diff = target - num
    if diff in hashMap:
      return hashMap[diff], i
    else:
      hashMap[num] = i
  
  return

def twoSum(nums, target):
  myMap = convertDictWithCount(nums)

  # find two sum in myMap
  for num in nums:
    diff = target - num
    if num == diff and myMap[num] <= 1:
        continue
    elif diff in myMap:
      return [num, diff]

def convertDictWithCount(nums):
  myMap = {} # key: num, value: count

  # build the dictionary
  for num in nums:
    if num in myMap: 
      myMap[num] += 1
    else:
      myMap[num] = 1
  return myMap

# return index of pairs
def twoSumPairs(nums, target):
  pairs = set()
  myMap = convertMapWithIndex(nums)

  # find two sum pairs
  for i in range(len(nums)):
    diff = target - nums[i]
    if nums[i] == diff:
      if len(myMap[nums[i]]) >= 2:
        pairs.add((myMap[nums[i]].pop(), myMap[nums[i]].pop()))
    elif diff in myMap and len(myMap[diff]) >= 1:
      snd = myMap[diff].pop()
      pairs.add((i, snd))

  return pairs

# return value of pairs
def twoSumPairs1(nums, target):
  pairs = set()
  myMap = convertMapWithIndex(nums)

  # find two sum pairs
  for i in range(len(nums)):
    diff = target - nums[i]
    if nums[i] == diff:
      if len(myMap[nums[i]]) >= 2:
        pairs.add((nums[myMap[nums[i]].pop()], nums[myMap[nums[i]].pop()]))
    elif diff in myMap and len(myMap[diff]) >= 1:
      snd = myMap[diff].pop()
      newPair = (min(nums[i], nums[snd]), max(nums[i], nums[snd]))        
      pairs.add(newPair)

  return pairs

# use value to store indexes
def convertMapWithIndex(nums):
  myMap = {} # key: num, value: index

  # build the dictionary
  for i in range(len(nums)):
    if nums[i] in myMap: 
      myMap[nums[i]].append(i)
    else:
      myMap[nums[i]] = [i]

  return myMap

'''
Valid Parentheses
neetcode75.valid_parens("(){[]}") -> True
'''
closeParens = {')':'(',
                '}':'{',
                ']':'['}

def validParens(parens):
  stack = []
  for paren in parens:
    if paren in closeParens:
      if stack == []:
        return False
      elif stack.pop() != closeParens[paren]: 
        return False
    else:
      stack.append(paren)

  if stack == []: 
    return True
  else: 
    return False

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
'''
def coinChangeCount(coins, amount):
  res = [0]

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
      res.append(min(changes, key=lambda x:len(x)))
    else:
      res.append([])
    
  return res.pop()
