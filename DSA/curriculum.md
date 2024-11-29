# Data Structures and Algorithms: A Comprehensive Curriculum

## Week 1: Foundations of Algorithm Analysis and Complexity

### 1.1 Introduction to Algorithm Analysis [3 hours]
#### 1.1.1 Basic Terminology
- Definition of an algorithm
- Properties of algorithms
  - Finiteness
  - Definiteness
  - Input/Output
  - Effectiveness
- Relationship between algorithms and programs
- Role of algorithms in problem-solving

#### 1.1.2 Algorithm Specification Methods
- Natural language
- Pseudocode conventions
- Flow charts
- Implementation patterns
- Program verification techniques

#### 1.1.3 Performance Measurement
- Machine-dependent measurements
  - Clock time
  - CPU cycles
  - Memory usage patterns
- Machine-independent measurements
  - Operation counts
  - Basic operations identification
  - Input size metrics

### 1.2 Complexity Analysis [3 hours]
#### 1.2.1 Time Complexity
- Counting primitive operations
- Best case analysis
- Average case analysis
- Worst case analysis
- Amortized analysis introduction
- Example analysis:
```go
// Analysis of linear search
func linearSearch(arr []int, target int) int {
    operations := 0
    for i := 0; i < len(arr); i++ {
        operations++ // Comparison operation
        if arr[i] == target {
            return operations
        }
    }
    return operations
}
```

#### 1.2.2 Space Complexity
- Memory layout analysis
  - Stack space
  - Heap allocations
  - Static/global memory
- Auxiliary space concepts
- In-place algorithms
- Space-time tradeoffs

### 1.3 Asymptotic Notation [3 hours]
#### 1.3.1 Big O Notation
- Formal definition
- Mathematical properties
- Common growth rates
  - O(1): Constant
  - O(log n): Logarithmic
  - O(n): Linear
  - O(n log n): Linearithmic
  - O(n²): Quadratic
  - O(2ⁿ): Exponential

#### 1.3.2 Omega and Theta Notations
- Omega (Ω) notation properties
- Theta (Θ) notation properties
- Relationships between notations
- Practical implications

#### 1.3.3 Analysis Examples
```go
// Example 1: O(n²)
func bubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
}

// Example 2: O(log n)
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        }
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}
```

### Week 1 Lab/Recitation [3 hours]
#### Problem Set 1: Algorithm Analysis
1. Implement and analyze three different solutions for finding maximum subarray sum:
   - Cubic time solution (O(n³))
   - Quadratic time solution (O(n²))
   - Linear time solution (O(n)) using Kadane's algorithm

2. Space-Time Tradeoff Analysis:
   - Implement two solutions for finding duplicates:
     - Using sorting (Time: O(n log n), Space: O(1))
     - Using hash set (Time: O(n), Space: O(n))
   - Analyze and compare tradeoffs

3. Asymptotic Analysis Practice:
   - Analyze and improve complex algorithms
   - Identify bottlenecks
   - Apply optimization techniques

## Week 2: Basic Data Structures and Memory Management

### 2.1 Memory Management Fundamentals [3 hours]
#### 2.1.1 Computer Memory Organization
- Memory hierarchy
  - Registers
  - Cache levels
  - Main memory
  - Virtual memory
- Memory access patterns
- Cache-friendly programming

#### 2.1.2 Dynamic Memory Allocation
- Stack vs Heap allocation
- Memory allocation patterns
- Memory leaks and prevention
- Garbage collection concepts
- Example:
```go
// Memory allocation patterns
type Node struct {
    data int
    next *Node
}

// Stack allocation
func stackExample() {
    var x int        // Stack allocated
    var arr [5]int   // Stack allocated
}

// Heap allocation
func heapExample() *Node {
    return &Node{    // Heap allocated
        data: 42,
        next: nil,
    }
}
```

### 2.2 Arrays and Slices [3 hours]
#### 2.2.1 Array Fundamentals
- Memory layout
- Access patterns
- Bounds checking
- Multi-dimensional arrays
- Cache performance considerations

#### 2.2.2 Slice Implementation
- Internal structure
- Growth patterns
- Copy operations
- Memory efficiency
- Common pitfalls

#### 2.2.3 Advanced Operations
```go
// Slice internals demonstration
func demonstrateSliceGrowth() {
    s := make([]int, 0)
    cap := 0
    for i := 0; i < 1000; i++ {
        s = append(s, i)
        if cap != cap(s) {
            fmt.Printf("Length: %d, New Capacity: %d\n", len(s), cap(s))
            cap = cap(s)
        }
    }
}

// Efficient slice operations
func optimizedOperations() {
    // Pre-allocating with make
    s := make([]int, 0, 1000)
    
    // Efficient append
    for i := 0; i < 1000; i++ {
        s = append(s, i)
    }
    
    // Slice window operations
    window := s[10:20:20] // Third index controls capacity
}
```

### 2.3 Basic Linear Data Structures [3 hours]
#### 2.3.1 Stack Implementation
- Array-based implementation
- Linked list-based implementation
- Growth strategies
- Performance comparison
- Applications:
  - Expression evaluation
  - Function call management
  - Backtracking algorithms

```go
// Stack ADT implementation
type Stack struct {
    items []interface{}
}

func NewStack() *Stack {
    return &Stack{items: make([]interface{}, 0)}
}

func (s *Stack) Push(item interface{}) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() (interface{}, error) {
    if s.IsEmpty() {
        return nil, errors.New("stack is empty")
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, nil
}
```

#### 2.3.2 Queue Implementation
- Array-based circular queue
- Linked list-based queue
- Double-ended queue
- Priority queue introduction
- Applications:
  - Process scheduling
  - Resource pooling
  - BFS implementations

### Week 2 Lab/Recitation [3 hours]
#### Problem Set 2: Implementation Challenges

1. Stack Applications:
   - Implement a calculator that evaluates postfix expressions
   - Implement parentheses matching
   - Convert infix to postfix expressions

2. Queue Applications:
   - Implement a process scheduler using priority queue
   - Implement a sliding window maximum using deque
   - Implement a circular buffer

3. Memory Management:
   - Implement a memory pool allocator
   - Create a garbage collection simulator
   - Analyze memory usage patterns

#### Programming Project:
Implement a text editor buffer with the following features:
- Efficient insert/delete operations
- Undo/redo functionality
- Memory-efficient string storage
- Optimal cursor movement
- Copy/paste operations

# Week 3: Advanced Linear Data Structures

## Learning Objectives
Students will be able to:
1. Implement various types of linked lists
2. Understand memory management in linked structures
3. Perform complex list operations
4. Optimize linked list implementations
5. Apply linked lists to solve real-world problems

### 3.1 Linked Lists Fundamentals [3 hours]
#### 3.1.1 Core Concepts
- Node structure and memory layout
- Pointer manipulation basics
- Memory allocation patterns
- Traversal techniques
- Comparison with arrays/slices
- Memory efficiency considerations

#### 3.1.2 Implementation Variations
- Singly linked lists
- Doubly linked lists
- Circular linked lists
- XOR linked lists
- Skip lists
- Unrolled linked lists

### 3.2 Advanced Linked List Operations [3 hours]
#### 3.2.1 Complex Operations
- List reversal (iterative and recursive)
- Cycle detection and removal
- Finding intersections
- Merging sorted lists
- Deep copy with random pointers
- Two-pointer techniques

#### 3.2.2 Optimization Strategies
- In-place operations
- Pointer manipulation efficiency
- Memory access patterns
- Cache considerations
- Performance analysis

### 3.3 Memory-Efficient Implementations [3 hours]
#### 3.3.1 Advanced Techniques
- Memory pooling
- Reference counting
- Cache-conscious structures
- Lock-free implementations
- Memory barriers
- Atomic operations

### Week 3 Lab/Recitation [3 hours]
#### Problem Set 3: Advanced List Manipulation
1. Implementation Challenges:
   - Build LRU Cache using doubly linked list
   - Create skip list with probabilistic balancing
   - Develop XOR linked list implementation
   
2. Algorithm Practice:
   - K-way merge of sorted lists
   - In-place block reversal
   - Cycle detection algorithms
   
3. Performance Tasks:
   - Implementation comparison
   - Memory profiling
   - Cache analysis

#### Programming Project: Document Editor Backend
Build a text editor's underlying data structure that supports:
- Efficient insertion/deletion
- Undo/redo operations
- Copy/paste functionality
- Multiple cursor support
- Line splitting/merging
- Memory-efficient storage
- Performance monitoring
- Crash recovery

# Week 4: Trees and Binary Search Trees

### 4.1 Tree Fundamentals [3 hours]
#### 4.1.1 Basic Concepts
- Tree terminology
- Tree properties
- Tree traversal methods
- Binary trees
- N-ary trees
- Tree representations
- Space efficiency

#### 4.1.2 Tree Traversals
- Inorder traversal
- Preorder traversal
- Postorder traversal
- Level-order traversal
- Constant space traversals
- Iterator patterns

### 4.2 Binary Search Trees [3 hours]
#### 4.2.1 BST Fundamentals
- BST properties
- Insertion methods
- Deletion algorithms
- Balancing concepts
- Height analysis
- Augmentation techniques

#### 4.2.2 Tree Operations
- Search optimization
- Range queries
- Predecessor/Successor
- Floor/Ceiling operations
- Selection algorithms

### 4.3 Advanced Tree Algorithms [3 hours]
#### 4.3.1 Complex Operations
- Lowest common ancestor
- Serialization techniques
- Path problems
- Tree isomorphism
- Diameter calculation
- Tree reconstruction

### Week 4 Lab/Recitation [3 hours]
#### Problem Set 4: Tree Implementation
1. Basic Operations:
   - Implement AVL tree
   - Create tree iterator
   - Build range query system

2. Advanced Problems:
   - Tree serialization/deserialization
   - K-th element algorithms
   - Tree visualization

3. Special Cases:
   - BST recovery
   - Tree-to-list conversion
   - Threaded tree implementation

#### Programming Project: Tree-based Database Index
Build a database indexing system that features:
- Multiple index support
- Range query optimization
- Concurrent access handling
- Memory-efficient storage
- Auto-balancing capability
- Performance monitoring
- Recovery mechanisms
- Query optimization

## Week 5: Advanced Tree Structures and Self-Balancing Trees

### 5.1 Red-Black Trees [3 hours]
#### 5.1.1 Fundamental Concepts
- Red-black tree properties
  1. Every node is either red or black
  2. Root is always black
  3. No two adjacent red nodes
  4. Every path has same number of black nodes
- Color flipping
- Rotation operations
- Invariant maintenance

```go
type Color bool

const (
    RED   Color = true
    BLACK Color = false
)

type RBNode[T comparable] struct {
    data   T
    color  Color
    left   *RBNode[T]
    right  *RBNode[T]
    parent *RBNode[T]
}

// Red-black tree insertion
func (tree *RedBlackTree[T]) Insert(data T) {
    node := &RBNode[T]{
        data:   data,
        color:  RED,
        left:   nil,
        right:  nil,
        parent: nil,
    }
    
    // Standard BST insert
    tree.bstInsert(node)
    
    // Fix Red-Black properties
    tree.fixInsertion(node)
}

func (tree *RedBlackTree[T]) fixInsertion(node *RBNode[T]) {
    for node != tree.root && node.parent.color == RED {
        if node.parent == node.parent.parent.left {
            uncle := node.parent.parent.right
            if uncle != nil && uncle.color == RED {
                // Case 1: Uncle is red
                node.parent.color = BLACK
                uncle.color = BLACK
                node.parent.parent.color = RED
                node = node.parent.parent
            } else {
                if node == node.parent.right {
                    // Case 2: Uncle is black, node is right child
                    node = node.parent
                    tree.leftRotate(node)
                }
                // Case 3: Uncle is black, node is left child
                node.parent.color = BLACK
                node.parent.parent.color = RED
                tree.rightRotate(node.parent.parent)
            }
        } else {
            // Mirror cases for right parent
            // [Implementation similar to above with left/right swapped]
        }
    }
    tree.root.color = BLACK
}
```

### 5.2 B-Trees and B+ Trees [3 hours]
#### 5.2.1 Multi-way Search Trees
- B-tree properties
- Minimum degree
- Node splitting
- Key redistribution
- Practical applications
  - Database indexing
  - File systems
  - Disk-based storage

```go
type BTreeNode[T comparable] struct {
    keys     []T
    children []*BTreeNode[T]
    leaf     bool
    n        int  // Number of keys
}

type BTree[T comparable] struct {
    root *BTreeNode[T]
    t    int  // Minimum degree
}

func (tree *BTree[T]) splitChild(x *BTreeNode[T], i int) {
    t := tree.t
    y := x.children[i]
    z := &BTreeNode[T]{
        keys:     make([]T, 2*t-1),
        children: make([]*BTreeNode[T], 2*t),
        leaf:     y.leaf,
        n:        t - 1,
    }
    
    // Copy the second half of y's keys to z
    for j := 0; j < t-1; j++ {
        z.keys[j] = y.keys[j+t]
    }
    
    // If not leaf, copy the second half of y's children to z
    if !y.leaf {
        for j := 0; j < t; j++ {
            z.children[j] = y.children[j+t]
        }
    }
    
    y.n = t - 1
    
    // Move x's children to make room for z
    for j := x.n; j >= i+1; j-- {
        x.children[j+1] = x.children[j]
    }
    
    x.children[i+1] = z
    
    // Move x's keys to make room for y's middle key
    for j := x.n-1; j >= i; j-- {
        x.keys[j+1] = x.keys[j]
    }
    
    x.keys[i] = y.keys[t-1]
    x.n = x.n + 1
}
```

### 5.3 Advanced Tree Applications [3 hours]
#### 5.3.1 Specialized Tree Structures
- Interval trees
- Segment trees
- Range trees
- Quadtrees
- Octrees

```go
// Segment tree for range queries
type SegmentTree[T any] struct {
    tree    []T
    n       int
    combine func(T, T) T  // Function to combine values
}

func NewSegmentTree[T any](arr []T, combine func(T, T) T) *SegmentTree[T] {
    n := len(arr)
    tree := make([]T, 4*n)
    st := &SegmentTree[T]{
        tree:    tree,
        n:       n,
        combine: combine,
    }
    st.build(arr, 0, 0, n-1)
    return st
}

func (st *SegmentTree[T]) build(arr []T, node int, start, end int) T {
    if start == end {
        st.tree[node] = arr[start]
        return st.tree[node]
    }
    
    mid := (start + end) / 2
    leftVal := st.build(arr, 2*node+1, start, mid)
    rightVal := st.build(arr, 2*node+2, mid+1, end)
    st.tree[node] = st.combine(leftVal, rightVal)
    return st.tree[node]
}

func (st *SegmentTree[T]) query(node, start, end, l, r int) T {
    if r < start || end < l {
        return st.getIdentity()
    }
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    mid := (start + end) / 2
    leftVal := st.query(2*node+1, start, mid, l, r)
    rightVal := st.query(2*node+2, mid+1, end, l, r)
    return st.combine(leftVal, rightVal)
}
```

### Week 5 Lab/Recitation [3 hours]
#### Problem Set 5: Advanced Tree Problems
1. Red-Black Tree Implementation:
   - Complete RB-tree with deletion
   - Augment for order statistics
   - Range query operations

2. B-Tree Applications:
   - Implement disk-based B-tree
   - Design caching strategy
   - Range-based operations

3. Specialized Trees:
   - Implement interval tree
   - Design spatial index
   - Range update operations

## Week 6: Priority Queues and Heaps

### 6.1 Binary Heaps [3 hours]
#### 6.1.1 Core Concepts
- Heap properties
- Array representation
- Build heap analysis
- Heapify operations
- Priority queue operations

```go
type BinaryHeap[T comparable] struct {
    items []T
    less  func(T, T) bool
}

func (h *BinaryHeap[T]) buildHeap() {
    n := len(h.items)
    for i := n/2 - 1; i >= 0; i-- {
        h.heapifyDown(i)
    }
}

func (h *BinaryHeap[T]) heapifyDown(i int) {
    smallest := i
    left := 2*i + 1
    right := 2*i + 2
    
    if left < len(h.items) && h.less(h.items[left], h.items[smallest]) {
        smallest = left
    }
    if right < len(h.items) && h.less(h.items[right], h.items[smallest]) {
        smallest = right
    }
    
    if smallest != i {
        h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
        h.heapifyDown(smallest)
    }
}
```

### 6.2 Advanced Heap Structures [3 hours]
#### 6.2.1 Specialized Heaps
- Fibonacci heaps
- Binomial heaps
- Leftist heaps
- Skew heaps
- Pairing heaps

```go
// Fibonacci Heap Node
type FibNode[T comparable] struct {
    key      T
    degree   int
    marked   bool
    parent   *FibNode[T]
    child    *FibNode[T]
    left     *FibNode[T]
    right    *FibNode[T]
}

type FibonacciHeap[T comparable] struct {
    min   *FibNode[T]
    size  int
    less  func(T, T) bool
}

func (fh *FibonacciHeap[T]) consolidate() {
    if fh.min == nil {
        return
    }
    
    // Calculate max degree
    phi := (1 + math.Sqrt(5)) / 2
    maxDegree := int(math.Log(float64(fh.size)) / math.Log(phi))
    degrees := make([]*FibNode[T], maxDegree+1)
    
    // List of roots
    rootList := make([]*FibNode[T], 0)
    curr := fh.min
    for {
        rootList = append(rootList, curr)
        curr = curr.right
        if curr == fh.min {
            break
        }
    }
    
    for _, w := range rootList {
        x := w
        d := x.degree
        for degrees[d] != nil {
            y := degrees[d]
            if fh.less(y.key, x.key) {
                x, y = y, x
            }
            fh.link(y, x)
            degrees[d] = nil
            d++
        }
        degrees[d] = x
    }
    
    // Reconstruct the root list
    fh.min = nil
    for _, w := range degrees {
        if w != nil {
            if fh.min == nil {
                fh.min = w
                w.left = w
                w.right = w
            } else {
                // Insert w into root list
                w.right = fh.min.right
                w.left = fh.min
                fh.min.right.left = w
                fh.min.right = w
                if fh.less(w.key, fh.min.key) {
                    fh.min = w
                }
            }
        }
    }
}
```

### 6.3 Applications and Optimizations [3 hours]
#### 6.3.1 Real-world Applications
- Event scheduling
- Task prioritization
- Dijkstra's algorithm
- Median maintenance
- Huffman coding

```go
// Median Finder using two heaps
type MedianFinder[T Number] struct {
    maxHeap *BinaryHeap[T]  // Lower half
    minHeap *BinaryHeap[T]  // Upper half
}

func (mf *MedianFinder[T]) AddNum(num T) {
    if mf.maxHeap.Len() == 0 || num < mf.maxHeap.Top() {
        mf.maxHeap.Push(num)
    } else {
        mf.minHeap.Push(num)
    }
    
    // Balance heaps
    if mf.maxHeap.Len() > mf.minHeap.Len()+1 {
        mf.minHeap.Push(mf.maxHeap.Pop())
    } else if mf.minHeap.Len() > mf.maxHeap.Len() {
        mf.maxHeap.Push(mf.minHeap.Pop())
    }
}

func (mf *MedianFinder[T]) FindMedian() T {
    if mf.maxHeap.Len() == mf.minHeap.Len() {
        return (mf.maxHeap.Top() + mf.minHeap.Top()) / 2
    }
    return mf.maxHeap.Top()
}
```

### Week 6 Lab/Recitation [3 hours]
#### Problem Set 6: Priority Queue Applications
1. Advanced Heap Operations:
   - Implement k-way merge
   - Running median
   - Top-k frequent elements

2. Specialized Implementations:
   - Fibonacci heap
   - Double-ended priority queue
   - External memory heap

3. Real-world Applications:
   - Job scheduler
   - Network packet prioritization
   - Event-driven simulation

# Week 7: Graph Theory and Graph Traversal

## Learning Objectives
Students will be able to:
1. Define and classify different types of graphs
2. Understand graph properties and representations
3. Implement basic graph traversal algorithms
4. Analyze graph connectivity problems
5. Apply graph algorithms to solve practical problems

### 7.1 Introduction to Graph Theory [3 hours]
#### 7.1.1 Basic Terminology and Concepts
- Graph definition
- Graph types
  - Directed vs undirected
  - Weighted vs unweighted
  - Simple vs multigraph
- Graph properties
  - Connectivity
  - Cycles
  - Paths
  - Degree
- Graph components

#### 7.1.2 Graph Representation
- Adjacency Matrix
- Adjacency List
- Edge List
- Tradeoffs between representations
```go
// Adjacency List representation
type Graph struct {
    vertices int
    adjList  map[int][]int
}

// Adjacency Matrix representation
type GraphMatrix struct {
    vertices int
    matrix   [][]int
}

// Edge List representation
type Edge struct {
    from, to, weight int
}
type EdgeGraph struct {
    vertices int
    edges    []Edge
}
```

### 7.2 Graph Traversal Algorithms [3 hours]
#### 7.2.1 Depth-First Search (DFS)
- Recursive implementation
- Iterative implementation
- Applications
  - Path finding
  - Cycle detection
  - Topological sorting
- Time and space complexity analysis

#### 7.2.2 Breadth-First Search (BFS)
- Queue-based implementation
- Level tracking
- Applications
  - Shortest paths in unweighted graphs
  - Connected components
  - Network distance
- Time and space complexity analysis

### 7.3 Graph Connectivity [3 hours]
#### 7.3.1 Connected Components
- Finding components in undirected graphs
- Strong components in directed graphs
- Bridges and articulation points
- Applications in network analysis

### Week 7 Lab/Recitation [3 hours]
#### Problem Set 7: Graph Implementation
1. Basic Graph Operations:
   - Implement graph using all three representations
   - Convert between representations
   - Compare memory usage and operation efficiency

2. Traversal Implementation:
   - Implement both DFS and BFS
   - Find paths between vertices
   - Detect cycles
   - Calculate component sizes

3. Connectivity Analysis:
   - Find connected components
   - Identify bridges
   - Detect articulation points
   - Analyze network vulnerability

#### Programming Project: Social Network Analyzer
Build a social network analysis tool with:
- Friend connection graph implementation
- Community detection using DFS/BFS
- Shortest path between users
- Influence measurement
- Connection suggestions
- Visual representation of communities
- Performance analysis tools

# Week 8: Advanced Graph Algorithms

### 8.1 Shortest Path Algorithms [3 hours]
#### 8.1.1 Single Source Shortest Path
- Dijkstra's algorithm
- Bellman-Ford algorithm
- Special cases
  - DAG shortest paths
  - Unweighted graphs
- Implementation considerations
```go
// Dijkstra's Algorithm template
func Dijkstra(graph *Graph, source int) map[int]int {
    // Implementation guide provided in lab
}

// Bellman-Ford template
func BellmanFord(graph *Graph, source int) (map[int]int, bool) {
    // Implementation guide provided in lab
}
```

### 8.2 Minimum Spanning Trees [3 hours]
#### 8.2.1 MST Algorithms
- Prim's algorithm
- Kruskal's algorithm
- Union-Find data structure
- Applications in network design

#### 8.2.2 Implementation Strategies
- Priority queue usage
- Efficient edge handling
- Performance optimization techniques

### 8.3 Network Flow [3 hours]
#### 8.3.1 Maximum Flow
- Ford-Fulkerson method
- Edmonds-Karp implementation
- Push-relabel approach
- Applications
  - Bipartite matching
  - Assignment problems
  - Network capacity

### Week 8 Lab/Recitation [3 hours]
#### Problem Set 8: Advanced Graph Problems
1. Shortest Path Implementation:
   - Implement Dijkstra's algorithm
   - Handle negative weights with Bellman-Ford
   - Compare performance with different graph densities
   - Optimize for specific graph types

2. MST Problems:
   - Implement both Prim's and Kruskal's algorithms
   - Build efficient Union-Find structure
   - Handle edge cases
   - Compare performance characteristics

3. Network Flow:
   - Implement Ford-Fulkerson method
   - Solve bipartite matching problems
   - Handle maximum flow scenarios
   - Optimize for large networks

#### Programming Project: Route Planning System
Implement a route planning system with:
- Map representation using weighted graphs
- Multiple routing algorithms
- Traffic condition simulation
- Alternative path suggestions
- Distance/time optimization
- Real-time updates
- Path visualization
- Performance monitoring dashboard

# Week 9: Fundamental Sorting Algorithms

## Learning Objectives
Students will be able to:
1. Understand and implement basic sorting algorithms
2. Analyze time and space complexity of sorting methods
3. Choose appropriate sorting algorithms for different scenarios
4. Optimize sorting implementations for various data types
5. Handle special cases and variations in sorting

### 9.1 Elementary Sorting Methods [3 hours]
#### 9.1.1 Basic Sorting Algorithms
- Bubble Sort
  - Basic implementation
  - Optimized version
  - Best/worst cases
  - Stability analysis
- Selection Sort
  - Implementation strategy
  - Memory efficiency
  - Performance characteristics
- Insertion Sort
  - In-place sorting
  - Adaptive behavior
  - Partial sorting
```go
func insertionSort(arr []int) {
    for i := 1; i < len(arr); i++ {
        key := arr[i]
        j := i - 1
        for j >= 0 && arr[j] > key {
            arr[j+1] = arr[j]
            j--
        }
        arr[j+1] = key
    }
}
```

#### 9.1.2 Algorithm Analysis
- Time complexity analysis
- Space complexity analysis
- Stability considerations
- In-place vs. out-of-place
- Adaptive behavior analysis

### 9.2 Efficient Sorting Algorithms [3 hours]
#### 9.2.1 Divide and Conquer Sorting
- Merge Sort
  - Divide-and-conquer strategy
  - Merge operation optimization
  - Space complexity considerations
  - Stability preservation
- Quick Sort
  - Partition schemes
  - Pivot selection strategies
  - Randomization benefits
  - Handling duplicates
```go
func quickSort(arr []int, low, high int) {
    if low < high {
        pivot := partition(arr, low, high)
        quickSort(arr, low, pivot-1)
        quickSort(arr, pivot+1, high)
    }
}
```

### 9.3 Special Purpose Sorting [3 hours]
#### 9.3.1 Linear Time Sorting
- Counting Sort
  - Range requirements
  - Memory usage
  - Stability considerations
- Radix Sort 
  - LSD vs MSD
  - Base selection
  - Performance factors
- Bucket Sort
  - Distribution analysis
  - Bucket size selection
  - Memory-performance tradeoffs

### Week 9 Lab/Recitation [3 hours]
#### Problem Set 9: Sorting Implementation
1. Elementary Sorting:
   - Implement and optimize bubble, selection, insertion sorts
   - Compare performance with different input sizes
   - Analyze best/worst case scenarios
   - Measure cache performance

2. Advanced Sorting:
   - Implement merge sort and quick sort
   - Optimize memory usage
   - Handle edge cases
   - Compare different pivot strategies

3. Linear Time Sorting:
   - Implement counting and radix sort
   - Handle different data distributions
   - Optimize for specific ranges
   - Analyze memory efficiency

#### Programming Project: Sorting Algorithm Visualizer
Build a sorting algorithm visualization system with:
- Visual representation of sorting process
- Step-by-step execution
- Performance metrics display
- Multiple algorithm support
- Custom input generation
- Comparison analytics
- Algorithm animation controls
- Performance profiling tools

# Week 10: Searching Algorithms and Advanced Sorting

### 10.1 Basic Searching Algorithms [3 hours]
#### 10.1.1 Sequential Search
- Linear search implementation
- Optimization techniques
- Special case handling
- Performance analysis
```go
func linearSearch(arr []int, target int) int {
    for i, v := range arr {
        if v == target {
            return i
        }
    }
    return -1
}
```

#### 10.1.2 Binary Search
- Iterative implementation
- Recursive implementation
- Variations for different scenarios
- Error handling
```go
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        }
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}
```

### 10.2 Advanced Searching Techniques [3 hours]
#### 10.2.1 Advanced Search Algorithms
- Interpolation Search
- Jump Search
- Exponential Search
- Fibonacci Search
- Meta-binary Search

#### 10.2.2 Performance Optimization
- Cache considerations
- Memory access patterns
- CPU optimization
- Space-time tradeoffs

### 10.3 Hybrid Sorting Algorithms [3 hours]
#### 10.3.1 Advanced Sorting Techniques
- Timsort
  - Natural run detection
  - Merge strategy
  - Galloping mode
- Introsort
  - Quicksort/Heapsort hybrid
  - Depth limit handling
  - Fallback strategy
- Shell Sort
  - Gap sequence selection
  - Performance analysis
  - Implementation optimization

### Week 10 Lab/Recitation [3 hours]
#### Problem Set 10: Advanced Search and Sort
1. Search Implementation:
   - Implement various search algorithms
   - Compare performance characteristics
   - Optimize for different data distributions
   - Handle edge cases

2. Hybrid Sorting:
   - Implement Timsort
   - Create custom hybrid sort
   - Performance testing
   - Memory optimization

3. Real-world Applications:
   - Database indexing simulation
   - Text search implementation
   - Sorting large datasets
   - Performance profiling

#### Programming Project: Search Engine Core
Implement a basic search engine core with:
- Efficient indexing system
- Multiple search algorithms
- Ranking system
- Query optimization
- Result caching
- Partial matching support
- Performance metrics
- Scalability features

# Week 11: Dynamic Programming

## Learning Objectives
Students will be able to:
1. Understand dynamic programming principles
2. Identify problems suitable for dynamic programming
3. Apply optimization techniques
4. Implement efficient memoization and tabulation
5. Solve complex recursive problems efficiently

### 11.1 Introduction to Dynamic Programming [3 hours]
#### 11.1.1 Core Concepts
- Optimal substructure
- Overlapping subproblems
- State definition
- Transition functions
- Memoization vs Tabulation

#### 11.1.2 Basic Problems
- Fibonacci sequence 
```go
// Memoization approach
func fib(n int, memo map[int]int) int {
    if val, exists := memo[n]; exists {
        return val
    }
    if n <= 1 {
        return n
    }
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
}

// Tabulation approach
func fibDP(n int) int {
    dp := make([]int, n+1)
    dp[0], dp[1] = 0, 1
    for i := 2; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    return dp[n]
}
```

### 11.2 Classic DP Problems [3 hours]
#### 11.2.1 One-dimensional DP
- Longest Increasing Subsequence
- Maximum Subarray Sum
- Climbing Stairs
- Coin Change
- Rod Cutting

#### 11.2.2 Two-dimensional DP
- Matrix Chain Multiplication
- Longest Common Subsequence
- Edit Distance
- Minimum Path Sum
- 0/1 Knapsack

### 11.3 Advanced DP Concepts [3 hours]
#### 11.3.1 Optimization Techniques
- State reduction
- Space optimization
- Rolling array technique
- Path reconstruction
- State compression

### Week 11 Lab/Recitation [3 hours]
#### Problem Set 11: DP Implementation
1. Basic DP Problems:
   - Implement both recursive and iterative solutions
   - Compare space-time tradeoffs
   - Optimize memory usage
   - Handle large inputs

2. Advanced Problems:
   - Solve matrix-based DP problems
   - Implement state compression
   - Optimize space complexity
   - Handle path reconstruction

3. Real-world Applications:
   - Text similarity algorithms
   - Resource allocation problems
   - Optimization scenarios
   - Performance analysis

#### Programming Project: Resource Optimization System
Build a system that:
- Solves complex resource allocation problems
- Handles multiple constraints
- Provides optimal solutions
- Visualizes decision process
- Supports real-time updates
- Includes performance monitoring
- Allows constraint modification
- Generates solution reports

# Week 12: Advanced Algorithms

### 12.1 Greedy Algorithms [3 hours]
#### 12.1.1 Greedy Strategy
- Greedy choice property
- Optimal substructure
- Safe moves
- Exchange arguments
- Proof of correctness

#### 12.1.2 Classic Problems
- Activity Selection
- Huffman Coding
- Fractional Knapsack
- Minimum Spanning Trees
- Job Scheduling
```go
type Activity struct {
    start, finish int
}

func activitySelection(activities []Activity) []Activity {
    // Sort by finish time
    sort.Slice(activities, func(i, j int) bool {
        return activities[i].finish < activities[j].finish
    })
    
    selected := []Activity{activities[0]}
    lastSelected := 0
    
    for i := 1; i < len(activities); i++ {
        if activities[i].start >= activities[lastSelected].finish {
            selected = append(selected, activities[i])
            lastSelected = i
        }
    }
    return selected
}
```

### 12.2 Divide and Conquer [3 hours]
#### 12.2.1 Advanced Applications
- Strassen's Matrix Multiplication
- Closest Pair of Points
- Fast Fourier Transform
- Karatsuba Multiplication
- Master Theorem Applications

### 12.3 Advanced Problem-Solving Techniques [3 hours]
#### 12.3.1 Algorithm Design Patterns
- Sliding Window
- Two Pointers
- Meet in the Middle
- Bit Manipulation
- Segment Trees

### Week 12 Lab/Recitation [3 hours]
#### Problem Set 12: Advanced Algorithm Implementation
1. Greedy Problems:
   - Implement classical greedy algorithms
   - Prove correctness
   - Handle edge cases
   - Optimize performance

2. D&C Problems:
   - Implement advanced D&C solutions
   - Analyze recursive complexity
   - Handle large datasets
   - Optimize memory usage

3. Design Patterns:
   - Implement sliding window problems
   - Solve two-pointer scenarios
   - Apply bit manipulation
   - Create efficient solutions

#### Programming Project: Algorithm Toolkit
Create a comprehensive algorithm toolkit that:
- Implements multiple algorithm paradigms
- Provides performance comparison tools
- Includes visualization components
- Supports problem analysis
- Features automatic testing
- Generates optimization reports
- Allows algorithm customization
- Includes benchmarking tools

# Week 13: String Algorithms

## Learning Objectives
Students will be able to:
1. Implement efficient string matching algorithms
2. Understand pattern searching techniques
3. Apply string processing optimizations
4. Analyze string algorithm complexity
5. Build practical text processing applications

### 13.1 Pattern Matching Algorithms [3 hours]
#### 13.1.1 Basic String Matching
- Naive pattern matching
- Rabin-Karp algorithm
- KMP algorithm
- Boyer-Moore algorithm
```go
// KMP Pattern Matching
func computeLPS(pattern string) []int {
    lps := make([]int, len(pattern))
    length := 0
    i := 1

    for i < len(pattern) {
        if pattern[i] == pattern[length] {
            length++
            lps[i] = length
            i++
        } else {
            if length != 0 {
                length = lps[length-1]
            } else {
                lps[i] = 0
                i++
            }
        }
    }
    return lps
}
```

### 13.2 Advanced String Processing [3 hours]
#### 13.2.1 Suffix Arrays and Trees
- Suffix array construction
- Longest common prefix array
- Suffix tree applications
- String searching optimization

#### 13.2.2 String Distance Algorithms
- Edit Distance variations
- Longest Common Subsequence
- String similarity metrics
- Approximate matching

### 13.3 Text Processing Applications [3 hours]
#### 13.3.1 Real-world Applications
- Text compression
- DNA sequence matching
- Spell checkers
- Auto-complete systems

### Week 13 Lab/Recitation [3 hours]
#### Programming Project: Text Analysis System
Build a comprehensive text analysis tool with:
- Multiple pattern matching algorithms
- String similarity comparisons
- Auto-complete functionality
- Spell checking capabilities
- Performance analytics
- Memory usage optimization
- Visualization components

# Week 14: Network Flow and Advanced Graph Topics

### 14.1 Advanced Network Flow [3 hours]
#### 14.1.1 Flow Algorithms
- Maximum Bipartite Matching
- Min-Cost Max-Flow
- Multi-commodity Flow
- Push-Relabel Algorithm

#### 14.1.2 Applications
- Assignment Problems
- Circulation with Demands
- Project Selection
- Network Reliability

### 14.2 Advanced Graph Topics [3 hours]
#### 14.2.1 Graph Coloring
- Vertex coloring algorithms
- Edge coloring
- Map coloring
- Chromatic number

#### 14.2.2 Path Algorithms
- Hamilton paths
- Euler paths
- Chinese Postman Problem
- Traveling Salesman Problem

### 14.3 Graph Applications [3 hours]
#### 14.3.1 Real-world Problems
- Network Design
- Resource Allocation
- Circuit Design
- Transportation Networks

### Week 14 Lab/Recitation [3 hours]
#### Programming Project: Network Optimization System
Create a network optimization tool that:
- Solves complex flow problems
- Handles multiple commodities
- Optimizes resource allocation
- Visualizes network states
- Analyzes bottlenecks
- Suggests improvements
- Monitors performance
- Generates reports

# Week 15: Advanced Problem-Solving Techniques

### 15.1 Computational Geometry [3 hours]
#### 15.1.1 Basic Concepts
- Point location
- Line intersection
- Convex hull algorithms
- Sweep line algorithms
```go
type Point struct {
    x, y float64
}

// Graham Scan for Convex Hull
func grahamScan(points []Point) []Point {
    // Implementation details provided in lab
}
```

### 15.2 Advanced Data Structures [3 hours]
#### 15.2.1 Specialized Structures
- Skip Lists
- Rope data structure
- Persistent data structures
- Cache-oblivious algorithms

### 15.3 Problem-Solving Strategies [3 hours]
#### 15.3.1 Advanced Techniques
- Two-pointer technique
- Sliding window
- Meet in the middle
- Heavy-light decomposition

### Week 15 Lab/Recitation [3 hours]
#### Programming Project: Algorithm Visualization Platform
Build a comprehensive visualization platform that:
- Demonstrates multiple algorithms
- Shows step-by-step execution
- Compares different approaches
- Measures performance
- Provides interactive learning
- Supports custom inputs
- Includes performance profiling
- Features algorithm animation
