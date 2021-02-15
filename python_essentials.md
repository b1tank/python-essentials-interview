# Python 3 essentials for tech coding interview


## Reference
- Python: https://medium.com/@ratulsaha/preparing-for-programming-interview-as-a-phd-student-with-python-5f8af8b40d5f
- BFS/DFS: https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
- Graph/BFS/Dijkstra/A*: https://www.redblobgames.com/pathfinding/a-star/implementation.html


## String
```py
a = 'ab cbe'
len(a)
for c in a:
a.find('b') => -1 on failure
a.index('b') => ValueError on failure
a.rfind('b')
a.rindex('b')
a.strip('e')
a.rstrip('e')
a.split()
a.split('c')
'/'.join(['a', 'b', 'c'])
'a = {}'.format(a)
'a'.isalnum() # alphanumeric
# isalpha, isascii, isdecimal, isdigit, islower, isnumeric, isspace, isupper
'a'.lower() => new copy
'a'.upper()
```


## Number
```py
float('inf'), float('-inf')
import sys
int_max = sys.maxsize # 2**63-1
int_min = -sys.maxsize - 1 # -2**63
# INT_MAX in Java = 2**31-1 = 2147483647
# INT_MIN in Java = -2**31 = -2147483648
```


## Conditional check
```py
a is b
a is None
a is not None
a == b
a != b
a in d
a not in d
```


## Loops
```py
counter = 0
while counter <= 5:
    print(counter)
    counter += 1
else:
    print "loop exited normally"
    # Output: 0 1 2 3 4 5 loop exited normally

for i in range(5):
    print(i)
    if i > 3:
        break
else:
    print "loop exited normally"
    # Output: 0 1 2 3 4

for i, val in enumerate(li):
```


## itertools

### itertools.chain
```py
>>> import itertools
>>> list2d = [[1,2,3], [4,5,6], [7], [8,9]]
>>> merged1 = tuple(itertools.chain.from_iterable(list2d))
>>> merged1
(1, 2, 3, 4, 5, 6, 7, 8, 9)
>>> merged2 = list(itertools.chain(*list2d))
>>> merged2
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Data structures

### List
#### List traversal
```py
range(start, stop, hop)
range(n) # [0,1,...,n-1]
range(1,n) # [1,...,n-1]
range(1,n,2) # [1,3,5,...,n-1] if n is even, or [1,3,5,...,n-2] if n is odd
range(n,-1,-1) # [n,n-1,n-2,...,0]
range(len(arr)) # Provides indices of an array arr
range(len(arr)-1,-1,-1) # Provides indices of arr backwards
```
#### List slicing
```py
arr[w:s] # Wait w elements, start copy (:), stop before reaching index s
arr = [1,2,3,4]
arr[1:] == [2,3,4]
arr[:2] == [1,2]
arr[::2] == [1,3] # 2 is the step
arr[::-1] == [4,3,2,1] # reverse a list (or string)
```
#### List manipulation and comprehension
```py
arr = [1,2,3]
arr.append(4)
arr.extend([5,6])
arr.insert(0, 7)
arr.insert(len(arr), 8) # equivalent to arr.append(8)
arr.remove(3) # remove first number 3
arr.index(3) # return index of first number 3
arr.count(3) # count the occurence of number 3
arr.pop(1) # pop the element with index 1
arr.pop() # like stack.pop()
arr.sort() # in-place
arr.reverse() # in-place
[str(x) for x in arr] # Output: ['1','2','3']
[str(x) for x in arr if x%2] # Output: ['1','3']
list(map(lambda x: str(x), arr)) # Output: ['1','2','3']; have to manually convert to list in python3. map() is iterable without explicit list conversion
```
#### List as queue
```py
arr = [1,2,3]   
arr.append(x) # queue.push(x)
arr.pop(0) #queue.pop()
arr[0] #queue.peek()
```
#### List as stack
```py
arr = [1,2,3]
arr.append(x) #stack.push(x)
arr.pop() # stack.pop()
arr[-1] # stack.peek()
```

### Set
```py
set()
s = {0, 1, 2}
s.add(4)
ss = {2, 3, 4}
>>> s - ss
{0, 1}
s >= > < <= | & ^ ss
>>> s ^ ss
{0, 1, 3, 4}
s.discard(100)    // no error
s.remove(100)     // Error: 100 not in s
```

### Dict 
```py
d = {}
d[3] = 'ha'
if 3 in d:
del 3
d.clear()
d.copy() # shallow copy
d.get(k, default=None)
d.pop(k), d.pop(k, default)
d.popitem()
d.keys()
d.values()
for key, val in d.items():
```

### Collections
#### deque()
```py
from collections import deque
queue = deque([1,2,3])
queue.append(4)
queue.appendleft(0)
queue.popleft()
queue.pop()
queue.clear()
queue.count(3)
queue.extend([3,4])
queue.extendleft([1,2]) # [X,Y,Z] => [2,1,X,Y,Z], NOTE: the series of left appends results in reversing the order of elements in the iterable argument
queue.remove(2) # remove (by value) first number 2
queue.reverse()
queue.rotate(2) # positive => right
queue.rotate(-1)
```
#### Counter()
```py
from collections import Counter
>>> c = Counter('ha')
>>> c
Counter({'h': 1, 'a': 1})
>>> c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
>>> c = Counter(cats=4, dogs=8)             # a new counter from keyword args
>>> c = Counter(['eggs', 'ham'])
>>> c['bacon']                              # count of a missing element is zero
0
>>> c['sausage'] = 0                        # counter entry with a zero count
>>> del c['sausage']                        # del actually removes the entry
```
#### defaultdict()
```py
from collections import defaultdict
>>> s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
>>> d = defaultdict(list)
>>> for k, v in s:
...     d[k].append(v)
...
>>> d.items()
[('blue', [2, 4]), ('red', [1]), ('yellow', [1, 3])]
>>> ice_cream = defaultdict(lambda: 'Vanilla')
>>> ice_cream['Sarah'] = 'Chunky Monkey'
>>> ice_cream['Abdul'] = 'Butter Pecan'
>>> print ice_cream['Sarah']
Chunky Monkey
>>> print ice_cream['Joe']
Vanilla
```

### LinkedList
```py
class ListNode(object):
    def __init__(self,x):
        self.val = x
        self.next = None
```

### Tree
```py
class TreeNodeoObject):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
```

### Priority Queue
```py
import heapq
pq = [(0, src)]
heapq.heapify(pq)
heapq.heappop(pq)
heapq.heappush(pq, (cost, cur_node))
heapq.nsmallest(n, li) # for small n, it’s efficient; for large n, use sorted
heapq.nlargest(n, li)
# maintain a fixed-size heap
if len(h) < capacity:
    heapq.heappush(h, thing)
else:
    # Equivalent to a push, then a pop, but faster
    spilled_value = heapq.heappushpop(h, thing)
    do_whatever_with(spilled_value)
heapq.heapreplace # "heapq.heappoppush"
```


## Algorithms

### Sort
```py
sorted([3,2,1])
>>> sorted("This is a test string from Andrew".split(), key=str.lower, reverse=False)
['a', 'Andrew', 'from', 'is', 'string', 'test', 'This']

l = [(3, 4), (2, 5), (6, 1)]
l.sort(key=lambda x : x[1], reverse=True)
>>> l
[(2, 5), (3, 4), (6, 1)]
```

### Binary search
```py
arr, target = [1,2,3,4,5], 4
start, end = 0, len(n)-1
while start <= end:
    mid = start + (end-start)/2 # (end+start)/2 might cause overflow (in JAVA/C++)!
    if arr[mid] == target:
        return mid # The value is found
    else:
        if arr[mid] < target:
            start = mid+1
        else:
            end = mid-1
return -1 # The value is not found
```

### Graph
#### Adjacency lists
- Unweighted graph
```py
graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}
```
- Weighted graph
```py
graph = {'A': set([('B', 1), ('C', 3)]),
         'B': set([('A', 9), ('D', 4), ('E', 2)]),
         'C': set([('A', 21), ('F', 6)]),
         'D': set([('B', 10)]),
         'E': set([('B', 6), ('F', 8]),
         'F': set([('C', 2), ('E', 5)])}
```
#### Depth first search
- stack
```py
def dfs(graph, start):
    stack = [start]
    visited = {start}
    while stack:
        cur_node = stack.pop()
        for nei in graph[cur_node]:
            if nei not in visited:
                visited.add(nei)
                stack.append(nei)
    return visited
```
- recursion
```py
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for nei in graph[start] - visited:
        dfs(graph, next, visited)
    return visited
```
#### Breadth first search
```py
import collections

queue = collections.deque([(start, 0)])
visited = {start}
while queue:
    cur_node, depth = queue.popleft()
    if cur_node == target_node: return depth
    for nei in graph[cur_node]: # NOTE: non-existing node can bring KeyError! May need to check "if node in graph"
        if nei not in visited:
            visited.add(nei)
            queue.append((nei, depth+1))
```
#### Dijkstra algorithm
```py
import heapq

pq = [(0, start)]
heap.heapify(pq)
while pq:
    cur_cost, cur_node = heapq.heappop(pq)
    if cur_node == target_node:
        return cur_cost
    for nei, weight in cur_node.neighbors():
        nei_cost = cur_cost + weight
        heapq.heappush(pq, (nei_cost, nei))
```
#### A* algorithm
```py
import heapq

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, target_node):
    pq = [(0, start)]
    
    came_from = {}
    cost = {}
			
    came_from[start] = None
    cost[start] = 0
    
    while pq:
        priority, cur_node = heapq.heappop(pq)
        
        if cur_node == target_node:
            break
        
        for nei, weight in graph[cur_node]:
            new_cost = cost[cur_node] + weight
            if nei not in cost or new_cost < cost[nei]:
                cost[next] = new_cost
                priority = new_cost + heuristic(nei, target_node)
                heapq.heappush(pq, (priority, nei))
                came_from[nei] = cur_node
    
    return came_from, cost
```



# Python Cheatsheet
## Data structures in python
[5. Data Structures — Python 2.7.15 documentation](https://docs.python.org/2/tutorial/datastructures.html)
* Stack: 
* use list
* append, pop
* Queue
* from collections import deque
* append, popleft
* Set
* fruit = set()
* fruit = set([list_member1, list_member2])
* fruit = {list_member1, list_member2}
* member in fruit, add, remove/discard
* Remove assumes object exist and will raise an error when it doesn’t find it. Discard will neglect the error.
* PriorityQueue
* [Python Tutorial: Data Structure - Priority Queue & heapq - 2018](https://www.bogotobogo.com/python/python_PriorityQueue_heapq_Data_Structure.php)
* Bisect

# Useful designs in Python
* String builder equivalent
* Just like other string builders
```py
def method4():
str_list = []
for num in xrange(loop_count):
str_list.append(`num`)
return ''.join(str_list)    
```
* List comprehension
```py
def method6():
return ''.join([`num` for num in xrange(loop_count)])
```
* Nested dictionary for trie
```py
self.root = {}
def addWord(self, word):
node = self.root
for char in word:
node = node.setdefault(char, {})
node[None] = None
```
* Disjoint Set Union (DSU)
```py
class DSU:
def __init__(self):
self.p = range(10001) # a large number
def find(self, x):
if self.p[x] != x:
self.p[x] = self.find(self.p[x])
return self.p[x]
def union(self, x, y):
self.p[self.find(x)] = self.find(y)
```

# Other useful notes
* hasattr(x, ‘foo’) is a built-in binary function that checks whether object x has an attribute x.foo
* string.lowercase.index(‘b’) = 1; string.lowercase[2]=‘c’
* [Reading and Writing CSV Files in Python – Real Python](https://realpython.com/python-csv/)
* Python string formatting good reference: [Python String Formatting Best Practices – Real Python](https://realpython.com/python-string-formatting/)

* Python list will be modified (in place) inside the function, but values won’t 
>>> a = [1,2,3]
>>> def modify(b):
…   b[0] = 100
… 
>>> modify(a)
>>> a
[100, 2, 3]
* lambda x, y: x2+y
* import random       
* random.randint(1,5): random int from 1-5 (both inclusive)
* random.random(): random float from 0-1
* random.choice([1,2,3,4])
* list(map(lambda x:int(x), “1234”))

# English Words
XOR: exclusive or
Logarithmic

Python throw exception

C++ vector initialization

#academics/Self-learning
#Flag

# swap two objects
```py
x = [1,3]
y = [2,4]
x, y = y, x

a = [[1,3],[2,4]]
a[0], a[1] = a[1], a[0]
```
