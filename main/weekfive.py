# reload import, put it in the terminal
# from importlib import reload

# 49. Sort Colors

colors = [0,1,2,1,0,0,1,2]

# put nums in buckets, replace the colors base on buckets
def bucket_sort(colors):
  bucket = [0,0,0]
  for color in colors:
    bucket[color] += 1

  track = 0 # the color is index
  for i in range(len(colors)):
    colors[i] = track
    bucket[track] -= 1

    if bucket[track] == 0:
      track += 1

# move 0 to left pointer, move 2 to right pointer
def two_pointers_sort(colors):
  left = 0
  right = len(colors) - 1
  i = 0
  
  while (i <= right):
    if colors[i] == 0: 
      swap(colors, i, left)
      left += 1
      i += 1
    elif colors[i] == 2: # moving 2 to right doesn't increase i
      swap(colors, i, right)
      right -= 1
    else:
      i += 1

def swap(arr, i, j):
  temp = arr[i]
  arr[i] = arr[j]
  arr[j] = temp