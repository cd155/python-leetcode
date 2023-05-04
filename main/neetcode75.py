# reload import, put it in the terminal
# from importlib import reload

'''
1. Two Sum
neetcode75.twoSum([2,7,11,15], 9)        -> [2, 7]
neetcode75.twoSumPairs([4,4,8,10,0,4], 8) -> [(5, 1), (2, 4), (4, 2)]
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
      newPair = min(i, snd), max(i,snd)
      pairs.add(newPair)

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
  res = set{}
  for i, num in enumerate(nums):
    diff = -num
    restArr = nums[i+1:]
    pairs = twoSumPairs1(restArr, diff) 
    for pair in pairs:
      res.add(sorted([num, pair[0], pair[1]]))

  return res
