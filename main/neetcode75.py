# reload import, put it in the terminal
# from importlib import reload

'''
1. Two Sum
neetcode75.twoSum ([2,7,11,15], 9)        -> [2, 7]
neetcode75.twoSumPairs([4,4,8,10,0,4], 8) -> [(5, 1), (2, 4)]
'''
def twoSumNeetcode(nums, target):
  pass

def twoSum(nums, target):
  myMap = convertDictWithCount(nums)

  # find two sum in myMap
  for num in nums:
    otherHalf = target - num
    if num == otherHalf and myMap[num] <= 1:
        continue
    elif otherHalf in myMap:
      return [num, otherHalf]

def convertDictWithCount(nums):
  myMap = {} # key: num, value: count

  # build the dictionary
  for num in nums:
    if num in myMap: 
      myMap[num] += 1
    else:
      myMap[num] = 1
  return myMap

def twoSumPairs(nums, target):
  pairs = []
  myMap = convertMapWithIndex(nums)

  # find two sum pairs
  for i in range(len(nums)):
    otherHalf = target - nums[i]
    if nums[i] == otherHalf:
      if len(myMap[nums[i]]) >= 2:
        pairs.append((myMap[nums[i]].pop(), myMap[nums[i]].pop()))
      else: continue
    elif otherHalf in myMap and len(myMap[nums[i]]) >= 1:
      pairs.append((i, myMap[otherHalf].pop()))

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

neetcode75.isDuplicate([])        -> None
neetcode75.isDuplicate([1,2,3,4]) -> [24,12,8,6]
'''
def productExceptSelf(nums):
  pass