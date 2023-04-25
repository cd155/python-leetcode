# reload import, put it in the terminal
# from importlib import reload

# 1. Two Sum

# weekone.two_sum ([2,7,11,15], 9) -> [2,7]
def two_sum(nums, target):
  my_dict = convert_dict(nums)

  # find two sum in my_dict
  for num in nums:
    other_half = target - num
    if num == other_half and my_dict[num] <= 1:
        continue
    elif other_half in my_dict:
      return [num, other_half]

def convert_dict(nums):
  my_dict = {}

  # build the dictionary
  for num in nums:
    if num in my_dict: 
      my_dict[num] += 1
    else:
      my_dict[num] = 1
  return my_dict

# weekone.two_sum_pairs([4,4,8,10,0,4], 8) -> [(1,0), (2,4)]
def two_sum_pairs(nums, target):
  pairs = []
  my_dict = convert_dict_index(nums)

  # find two sum pairs
  for i in range(len(nums)):
    other_half = target - nums[i]
    if nums[i] == other_half:
      if len(my_dict[nums[i]]) >= 2:
        pairs.append((my_dict[nums[i]].pop(), my_dict[nums[i]].pop()))
      else: continue
    elif other_half in my_dict and len(my_dict[nums[i]]) >= 1:
      pairs.append((i, my_dict[other_half].pop()))

  return pairs

# use value to store indexes
def convert_dict_index(nums):
  my_dict = {}

  # build the dictionary
  for i in range(len(nums)):
    if nums[i] in my_dict: 
      my_dict[nums[i]].append(i)
    else:
      my_dict[nums[i]] = [i]
  return my_dict

# 2. Valid Parentheses

close_parens = {')':'(',
                '}':'{',
                ']':'['}

# weekone.valid_parens("(){[]}") -> True
def valid_parens(parens):
  stack = []
  for paren in parens:
    if paren in close_parens:
      if stack == []:
        return False
      elif stack.pop() != close_parens[paren]: 
        return False
    else:
      stack.append(paren)

  if stack == []: 
    return True
  else: 
    return False
