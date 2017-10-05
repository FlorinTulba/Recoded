### Expanding the zeros found in a matrix on the corresponding rows and columns

* * *

Using parallelism and concurrency was the motivation behind checking this problem.

The C++ implementation provides:

- an algorithm that uses [CUDA](https://en.wikipedia.org/wiki/CUDA) to ask the GPU for finding the zeros within vertical chunks of the matrix, while the CPU updates the previously detected columns. The GPU transmits the found columns in parallel with finding new ones (There are different streams for those tasks). Updating the rows is supposed to go rather fast and is kept at the final for the CPU. Despite the high degree of introduced parallelism, the obtained performance is inferior to the other CPU-only approaches (around 80ns per analyzed element versus the better times listed at the bottom of the page). The reason is the increase in cache misses when working (row-wise) on column bands (on each row within the column band, it sets to zero only a few selected columns). Furthermore, the update of the rows performs some redundant work, as some elements were already set to zero during the update of the column bands
- an [OpenMP](http://www.openmp.org/) algorithm choosing the number of threads tackling the detection of zeros (first phase) and separately, the update of the matrix (the last phase). However, the update is performed in a single row-major traversal: either an entire row or just the marked columns from that row. The update phase is also designed to prevent false sharing. Recording the identified columns keeps the false sharing to a minimum by letting each thread record locally the columns with zeros from their part of the data and then merging those local recordings together. Barriers and locks were avoided whenever possible

Empirically, the multithreading implementations perform better when each used thread has enough elements to analyze, thus there are cases when only a few busy cores are better than several more with less to do. This was more obvious for the Java implementation:

- Java: 3.5ns per analyzed element for a single thread versus 5ns per analyzed element when using 2 threads
- the C++ implementation based on OpenMP shows for both cases times around 3ns per analyzed element

* * *

&copy; 2017 Florin Tulba (florintulba@yahoo.com)