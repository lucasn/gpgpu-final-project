#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "error.h"

#define THREADS_PER_BLOCK 512
#define FLOAT_ERROR 1e-5
#define DEBUG 0

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

int main(void) {
    int n_rows = 10;
    int n_cols = 10;
    matrix_t *a = alloc_matrix(n_rows, n_cols);
    matrix_t *b = alloc_matrix(n_rows, n_cols);
    matrix_t *c = alloc_matrix(n_rows, n_cols);
    matrix_t *d = alloc_matrix(n_rows, n_cols);

    for (int i = 0; i < n_rows * n_cols; i++) {
        a->m[i] = (double) i;
        b->m[i] = (double) i;
    }

    printf(">>> Testing matrix dot\n");

    matrix_dot(a, b, c);

    dim3 threads_per_block(16, 16);
    dim3 n_blocks(MAX(1, ceil(n_rows / threads_per_block.x)), MAX(1, n_cols / threads_per_block.y));
    gpu_matrix_dot<<< n_blocks, threads_per_block >>>(a, b, d);

    cudaGetLastError();
    CHECK_ERROR(cudaDeviceSynchronize());

    for (int i = 0; i < n_cols * n_rows; i++) {
        if (DEBUG)
            printf(">>> DEBUG: c[%d]->m=(%f) != d[%d]->m=(%f)\n", i, c->m[i], i, d->m[i]);
        if (abs(c->m[i] - d->m[i]) > FLOAT_ERROR) {
            printf(">>> Test failed: matrices aren't equal c[%d]->m=(%f) != d[%d]->m=(%f)\n", i, c->m[i], i, d->m[i]);
            return 0;
        }
    }
    printf(">>> Test passed\n");
    return 0;
}