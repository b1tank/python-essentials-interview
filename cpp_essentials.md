# C++ essentials for tech coding interview


## Best practices & heads-up
1. Always initialize types.
2. ```const``` is required for accessor member functions of classes; so think about ```const``` at the beginning in case you need to add it to every single member function later. 


## Inline function
```cpp
// compiler will insert the body of the function in that location as opposed to making a function call
// only for small functions that are used frequently, not for large functions
inline void swap(int & a, int & b) // inline is a compiler directive
{
    // code needs to be recompiled when the function is changed
    int temp = a;
    a = b;
    b = temp;
}
```


## Throw exception
* Common exceptions: http://stdcxx.apache.org/doc/stdlibref/2-3.html
```cpp
try {
    throw std::invalid_argument( "Pop an empty stack!" );
}
catch( const std::invalid_argument& e ) {
    std::cerr << "exception: " << e.what() << std::endl;
}
```


## String
```cpp
char c = 'c'
string a = "ab cbe"
a.length()      // Also, a.size()
for(auto c:a)   // c will be char
a.find('b')     //=> string::npos on failure
a.find('b',3)   //=> start search from index 3. Default at 0
a.rfind('b') 
a.rfind('b',3)  //=> start search from index 3 to the left. Default at npos
a.substr(1,2)   //=> start index and substring length
a.substr(1)     //=> from start index to end
// Split a string: use find and substr
std::string s = "scott>=tiger";
std::string delimiter = ">=";
std::string token = s.substr(0, s.find(delimiter));
// Joining a string: iterating through the vector
std::isalnum('2') // alphanumeric. Positive values mean true. Doesn't have to be 1.
# isalpha, isdigit, islower, isupper, isspace
char(tolower('A')) => new copy. Also see toupper()
std::transform(a.begin(), a.end(), a.begin(), ::tolower); // #include <algorithm>
std::to_string(42);
std::stoi("42");
ind = 'c'-'a';  // ind=2
char x = 'b'+2; // x = 'd'
x = 'A'-'a'+x;  // x = 'D'
```


## Number
```cpp
INT_MAX, INT_MIN // #include <limits.h>
# INT_MAX = 2**31-1 = 2147483647
# INT_MIN = -2**31 = -2147483648
c = a%b;  // Take the mod
// First (C style) method for random number generation
#include <cstdlib>
int v3 = rand() % 30 + 1985;   // v3 in the range 1985-2014
// Second method
#include <random>
std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0,9); // [0,9]: both inclusive
int number = distribution(generator); 
// Third method
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 999);
```


## Conditional check
```cpp
a == b
a != b
// Find element in a vector. If not available, return v.end()
if(std::find(v.begin(), v.end(), x) != v.end())
// Check container empty
v.empty()
// Bool literal
true
false
```


## Loops
```cpp
for(auto c:v)  // auto&, auto&& if you want references
for(auto it = v.begin();it!=v.end();it++)
```


## Vector (#include <vector>)
### Vector creation
```cpp
vector<int> arr (2,10)  // 2 elements with value 10
vector<int> arr {1,2,3}
vector<vector<int>> b;
b.push_back(vector<int>(3,2));  # b.size()=1, b[0].size()=3 
sum = vector<vector<int>> (N1+1, vector<int>(N2+1));
```
### Vector slicing
```cpp
v2 = std::vector<int>(v1.begin() + 1, v1.end());
std::reverse(v.begin(),v.end()) // reverse a container
```
### Vector manipulation and comprehension
```cpp
arr.push_back(4)
arr.insert(arr.end(),v2.begin(),v2.end()) // return the begin location of the newly added list part
arr.insert(arr.begin(),v2.begin(),v2.end())
arr.insert(arr.begin()+1, 7) // Insert before the pointer location. Return the pointer to the new element
it = arr.erase(it) // Erase the object at it and return the updated iterator
it = arr.erase(st, ed) // Erase a range: [st,ed)
vec.erase(std::remove(vec.begin(), vec.end(), 3), vec.end()); // erase-remove idiom. Remove all 3s. remove/remove_if are in algorithm.h
it = std::find (vec.begin(), vec.end(), 30);    // From algorithm.h
count = std::count(vec.begin(), vec.end(), 30)
a1 = arr.front()
aN = arr.back()
arr.pop_back()  // Return void!!
std::sort(arr.begin(),arr.end())                // in place sort
std::sort(arr.begin(),arr.end(),compareFunc)
```

## Queue (#include <queue>)
* Queue, stack, and PQ has no begin/end. Can't do loop. Can't modify elements by clear, reverse.
```cpp
std::queue<int> Q;  
Q.push(3); 
Q.pop();     // Return void!!!
a = Q.front();   // Return reference
a = Q.back();
a = Q.size();
```

## Stack (#include <stack>)
* Stack has no front/back. Only has top.
```cpp
std::stack<int> S;  
S.push(3) 
S.pop()     // Return void!!!
a = S.top() // Return reference
a = S.size();
```

## Priority Queue (#include <queue>)
```cpp
std::priority_queue<int> pq1;   // Max heap by default
std::priority_queue<int> pq1(arr.begin(),arr.end());
std::priority_queue<int, std::vector<int>, std::greater<int> > pq2; // Min heap
Q.push(3); 
Q.pop();     // Return void!!!
a = Q.top();
a = Q.size();
```
* Customized class and comparitor
```cpp
#include <iostream>
#include <queue>
#include <vector>

using namespace std;
class point{
public:
    int x;
    int y;
    point(int x, int y):x(x),y(y) {}
};

class pointComp{
public:
    bool operator()(point& a, point& b) {  // Operator()!!!
        return a.x<b.x;
    }
};

int main() {
    priority_queue<point,std::vector<point>,pointComp> Q;
    Q.push(point(1,2));
    Q.push(point(9,6));
    Q.push(point(3,0));

    while (!Q.empty()){
        cout << Q.top().x <<"\t"<< Q.top().y << endl;
        Q.pop();
    }

    return 0;
}
// Using a max heap
// 9       6                                                                                                                         
// 3       0                                                                                                                         
// 1       2        
```

## Hashset (#include <unordered_set>)
* Can be looped through using iterators from begin() to end()
* Array/vector are not hashable. One way to work around is to convert it to strings.
```cpp
std::unordered_set<std::string> S1;
std::unordered_set<std::string> S2 ( {"red","green","blue"} );
std::unordered_set<std::string> S3 (S2.begin(), S2.end());
std::unordered_set<std::string> S4 (S2);
S1.insert("black");
S1.insert(S2.begin(), S2.end());
S2.erase("green");      // Doesn't matter if the element doesn't exist in set
S2.erase(it);
std::unordered_set<std::string>::const_iterator it = S1.find ("nobody"); // it=S1.end()
S2.clear();
```

## Hashmap (#include <unordered_map>)
```cpp
std::unordered_map<int, std::string> D1;
std::unordered_map<int, std::string> D2( {{1,"red"},{2,"yellow"}} );
std::unordered_map<int, std::string> D3(D2.begin(), D2.end());
std::unordered_map<int, std::string> D4(D2);
D1[9] = "blue";
D2.insert( {{3,"black"},{4,"white"}} );
std::unordered_map<std::string,double>::const_iterator it = D3.find (7); // it=mymap.end()
std::unordered_map<std::string,double>::const_iterator it = D3.find (2); // it->first=2, it->second="yellow"
D1.erase(9);
D4.clear();
std::unordered_map<int, vector<std::string>> D5;
D5[11].push_back("word"); // Valid
```



## Pair ((#include <utility>)) and tuple
```cpp
std::pair <std::string,double> p1 ("tomatoes",2.30);
p1.first
p1.second
```
```cpp
#include <tuple>

std::tuple<int, int, int> three;

std::get<0>( three) = 0;
std::get<1>( three) = 1;
std::get<2>( three) = 2;
```

## Custom structures
### Tree
```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
```

### Singly linked list
```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
```
* Cautious: `node->next` instead of `node.next` in python. It's a pointer now.

### Doubly linked list
```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode *prev;
    ListNode(int x) : val(x), prev(NULL), next(NULL) {}
};
```

### Graph
```cpp
class Node {
public:
    int val;
    vector<Node*> neighbors;

    Node() {}

    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
```

## Split text
* Most elegant solution if we can use boost
```cpp
#include <boost/algorithm/string.hpp>
std::string text = "Let me split this into words";
std::vector<std::string> results;

boost::split(results, text, [](char c){return c == ' ';});
```
* A somehow magical solution by initializaing vector with istringstream
```cpp
#include <sstream>
std::string text = "Let me split this into words";
std::istringstream iss(text);
std::vector<std::string> results((std::istream_iterator<std::string>(iss)),std::istream_iterator<std::string>());
```
* The most straight-forward solution using stringstream and getline function
```cpp
#include <sstream>
stringstream ss(str);       // Can also be istringstream
vector<string> results;
string tok;
char delimiter = '/';
while(getline(ss, tok, delimiter)) {
    results.push_back(tok);
}
```

## Join text
```cpp
std::ostringstream ss;
if(begin != end)
{
    ss << *begin++; // see 3.
}    
while(begin != end) // see 3.
{
    ss << separator;
    ss << *begin++;
}
ss << concluder;
return ss.str();

```

## Algorithms

### Sort
* Custom comparison function
```cpp
int main()
{
    vector<point> a{point(1,2),point(9,6),point(3,0)};
    std::sort(a.begin(),a.end(),[](point& a, point& b){return a.y<b.y;});
    for (auto p:a) {
        cout << p.x <<"\t" <<p.y <<endl;
    }
    
    return 0;
}
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
