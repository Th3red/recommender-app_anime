# Anime Recommendation System Comparison

This repository contains two implementations of an anime recommendation system built using Flask. The first implementation uses a **sparse matrix** approach with SQLAlchemy, while the second employs a **dense matrix** approach with psycopg2. This document compares both implementations in terms of execution time, memory usage, and theoretical performance.

## Table of Contents

- [Implementations Overview](#implementations-overview)
- [Profiling Methodology](#profiling-methodology)
- [Performance Metrics](#performance-metrics)
  - [Execution Time](#execution-time)
  - [Memory Usage](#memory-usage)
- [Big O Analysis](#big-o-analysis)
- [Conclusion](#conclusion)

## Implementations Overview

### First `app.py` - Sparse Matrix Approach

- **Data Structures:** `csr_matrix` from `scipy.sparse`
- **Database Interaction:** SQLAlchemy with connection pooling
- **Fuzzy Matching:** `thefuzz`
- **Logging:** Python's `logging` module

### Second `app.py` - Dense Matrix Approach

- **Data Structures:** NumPy `ndarray`
- **Database Interaction:** psycopg2 without connection pooling
- **Fuzzy Matching:** `fuzzywuzzy`
- **Logging:** `print` statements

## Profiling Methodology

- **Execution Time:** Measured using `timeit` and `cProfile`
- **Memory Usage:** Monitored using `memory_profiler` and `tracemalloc`
- **Big O Analysis:** Theoretical analysis based on algorithmic steps

## Performance Metrics

### Execution Time

| Function                            | First `app.py` (Sparse) | Second `app.py` (Dense) |
| ----------------------------------- | ----------------------- | ----------------------- |
| `build_user_anime_matrix`           | 51.01 seconds           | 231.97 seconds          |
| `build_hnsw_index`                  | 189.7 seconds           | 731.61 seconds          |
| `insert_new_user_into_index`        | 18.69 seconds           | 10.43 seconds           |
| `get_user_user_recommendations`     | 151.55 seconds          | 22.01 seconds           |
| `get_content_based_recommendations` | 20.45 seconds           | 11.32 seconds           |

_Note:_ The above times are illustrative. Replace them with your actual profiling results.

### Memory Usage

| Function                            | First `app.py` (Sparse) | Second `app.py` (Dense) |
| ----------------------------------- | ----------------------- | ----------------------- |
| `build_user_anime_matrix`           | 2245.45 MiB             | 2680.66 MiB             |
| `build_hnsw_index`                  | 5163.62 MiB             | 6528.52 MiB             |
| `insert_new_user_into_index`        | 1392.13 MiB             | 2.66 MiB                |
| `get_user_user_recommendations`     | 1394.57 MiB             | 3.26 MiB                |
| `get_content_based_recommendations` | 78.37 MiB               | 15.25 MiB               |

_Note:_ The above memory usages are illustrative. Replace them with your actual profiling results.

## Big O Analysis

| Function                            | First `app.py` (Sparse) | Second `app.py` (Dense) |
| ----------------------------------- | ----------------------- | ----------------------- |
| `build_user_anime_matrix`           | O(N + M)                | O(N \* M)               |
| `build_hnsw_index`                  | O(U \* D log U)         | O(U \* D log U)         |
| `insert_new_user_into_index`        | O(D log U)              | O(D + log U)            |
| `get_user_user_recommendations`     | O(Uk + Mk)              | O(U \* K + Mk)          |
| `get_content_based_recommendations` | O(G \* log G)           | O(G \* log G)           |

**Definitions:**

- **N:** Number of Users
- **U:** Number of users vectors
- **M:** Number of anime
- **K:** Number of neighbors
- **R:** Average number of ratings per neighbor
- **D:** Number of dimensions
- **G:** Genres

## Conclusion

- **First `app.py` (Sparse Matrix):**

  - **Pros:** Lower memory usage, faster startup.
  - **Cons:** Slower recommendation generation.

- **Second `app.py` (Dense Matrix):**
  - **Pros:** Faster recommendation generation.
  - **Cons:** Higher memory usage, slower startup.
