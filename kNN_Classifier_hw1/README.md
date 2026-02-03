# k-Nearest Neighbors Implementation

Implementation of kNN classifier from scratch as part of Stanford CS231n-inspired assignment.

## What I Implemented

### **Completed:**
- **`compute_distances_two_loops()`** - Naive implementation with nested loops
- **`compute_distances_one_loop()`** - Partial vectorization with one loop
- **`compute_distances_no_loops()`** - Fully vectorized using broadcasting and dot product
- **`predict_labels()`** - Prediction with majority voting and tie-breaking
- **`predict()`** - Main interface supporting different computation modes

### **Analysis Performed:**
- Validation against scikit-learn's kNN implementation
- Accuracy testing on digit classification dataset (95%+ accuracy)
- Performance comparison between three implementations
- Exploration of distance metric properties (L1 vs L2 invariance)

## Key Results
- **Accuracy**: Achieved 95% on test set with k=1
- **Speedup**: Vectorized implementation significantly faster than naive loops
- **Verification**: Results match scikit-learn implementation for majority of cases

## Concepts Demonstrated
- **Vectorization techniques** in NumPy (broadcasting, matrix operations)
- **Algorithm optimization** from O(n²) to more efficient implementations
- **Distance metrics** understanding (Euclidean/L2 distance computation)
- **kNN algorithm** fundamentals (lazy learning, nearest neighbor search)

## Project Structure
