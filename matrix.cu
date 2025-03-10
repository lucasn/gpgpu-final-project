#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "error.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define TILE_SIZE 16

matrix_t *alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res;

    CHECK_ERROR(cudaMallocManaged(&res, sizeof(matrix_t)));

    CHECK_ERROR(cudaMallocManaged(&res->m, columns * rows * sizeof(double)));
    // CHECK_ERROR(cudaMemset(res->m, 0, columns * rows * sizeof(double)));

    res->columns = columns;
    res->rows = rows;

    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    CHECK_ERROR(cudaFree(m->m));
    CHECK_ERROR(cudaFree(m));
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

void gpu_matrix_dot_wrapper(matrix_t *m1, matrix_t *m2, matrix_t *res, int synchronize) {
    dim3 threads_per_block(16, 16);

    int n_blocks_x = (res->columns + threads_per_block.x - 1) / threads_per_block.x;
    int n_blocks_y = (res->rows + threads_per_block.y - 1) / threads_per_block.y;

    dim3 n_blocks(n_blocks_x, n_blocks_y);
    gpu_matrix_dot<<< n_blocks, threads_per_block >>>(m1, m2, res);

    if (synchronize) {
        cudaGetLastError();
        CHECK_ERROR(cudaDeviceSynchronize());
    }
}

__global__
void gpu_matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res) {
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    __shared__ double shared_m1[TILE_SIZE][TILE_SIZE];
    __shared__ double shared_m2[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y ;
    int col = blockIdx.x * blockDim.x + threadIdx.x ;

    double sum = 0;

    int n_tiles = (m1->columns + TILE_SIZE - 1) / TILE_SIZE;

    for (int p = 0; p < n_tiles; p++) {

        if (row < m1->rows && p * TILE_SIZE + threadIdx.x < m1->columns){
            shared_m1[threadIdx.y][threadIdx.x] = m1->m[row * m1->columns + (p * TILE_SIZE + threadIdx.x)];
        }
        else {
            shared_m1[threadIdx.y][threadIdx.x] = 0;
        }

        if (p * TILE_SIZE + threadIdx.y < m2->rows && col < m2->columns) {
            shared_m2[threadIdx.y][threadIdx.x] = m2->m[(p * TILE_SIZE + threadIdx.y) * m2->columns + col];
        }
        else {
            shared_m2[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();

        if (row < res->rows && col < res->columns) {
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += shared_m1[threadIdx.y][i] * shared_m2[i][threadIdx.x];
            }
        }
        __syncthreads();
        
    }

    if (row < res->rows && col < res->columns) {
        res->m[row * res->columns + col] = sum;
    }
}


void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}