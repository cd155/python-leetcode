# reload import, put it in the terminal
# from importlib import reload

# 1. Two Sum

# weekone.two_sum ([2,7,11,15], 9) -> [0,1]
def two_sum(nums, target):
  my_dict = convert_dict(nums)

  for num in nums:
    other_half = target - num
    if num == other_half and my_dict[num] <= 1:
        continue
    elif other_half in my_dict:
      return [num, other_half]

def convert_dict(nums):
  my_dict = {}

  for num in nums:
    if num in my_dict: 
      my_dict[num] += 1
    else:
      my_dict[num] = 1
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
