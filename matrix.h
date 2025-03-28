#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;

matrix_t *alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1, double s, matrix_t *res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

void gpu_matrix_dot_wrapper(matrix_t *m1, matrix_t *m2, matrix_t *res, int synchronize);

void gpu_matrix_dot_wrapper_stream(matrix_t *m1, matrix_t *m2, matrix_t *res, cudaStream_t stream);

__global__ void gpu_matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void gpu_matrix_dot_no_shared(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void gpu_matrix_scalar_minus(matrix_t *m1, matrix_t *m2, double s, matrix_t *res);

void gpu_matrix_scalar_minus_wrapper(matrix_t *m1, matrix_t *m2, double s, matrix_t *res);
#endif