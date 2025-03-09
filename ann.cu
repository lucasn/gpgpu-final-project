#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include "error.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    bool generate;
    double z1;

	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t *nn;
    CHECK_ERROR(cudaMallocManaged(&nn, sizeof(ann_t)));
    CHECK_ERROR(cudaMallocManaged(&nn->layers, number_of_layers * sizeof(layer_t *)));

    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t *layer;

    CHECK_ERROR(cudaMallocManaged(&layer, sizeof(layer_t)));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn, double (*activation_function)(double))
{
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        // matrix_dot(nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)

        dim3 threads_per_block(16, 16);
        dim3 n_blocks1(
            MAX(1, 
                MAX(ceil(nn->layers[l]->weights->rows / threads_per_block.x), ceil(nn->layers[l-1]->activations->rows / threads_per_block.x))
            ), 
            MAX(1, 
                MAX(ceil(nn->layers[l]->weights->columns / threads_per_block.y), ceil(nn->layers[l-1]->activations->columns / threads_per_block.y))
            )
        );
        gpu_matrix_dot<<< n_blocks1, threads_per_block >>>(nn->layers[l]->weights, nn->layers[l-1]->activations, z1);

        //matrix_dot(nn->layers[l]->biases, one, z2); // z2 <- b^l x 1        

        cudaGetLastError();
        CHECK_ERROR(cudaDeviceSynchronize());

        dim3 n_blocks2(
            MAX(1, 
                MAX(ceil(nn->layers[l]->biases->rows / threads_per_block.x), ceil(one->rows / threads_per_block.x))
            ), 
            MAX(1, 
                MAX(ceil(nn->layers[l]->biases->columns / threads_per_block.y), ceil(one->columns / threads_per_block.y))
            )
        );

        gpu_matrix_dot<<< n_blocks2, threads_per_block >>>(nn->layers[l]->biases, one, z2);

        cudaGetLastError();
        CHECK_ERROR(cudaDeviceSynchronize());

        matrix_sum(z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1      

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
     
        destroy_matrix(z1);
        destroy_matrix(z2);
        destroy_matrix(one);
    }
}

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw, *delta_tmp, *dfz;
        tw = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(nn->layers[l]->weights, tw); // (w^l)T        
        // matrix_dot(tw, nn->layers[l]->delta, delta_tmp); // (w^l)T x delta^l

        dim3 threads_per_block(16, 16);
        dim3 n_blocks(
            MAX(1, 
                MAX(ceil(tw->rows / threads_per_block.x), ceil(nn->layers[l]->delta->rows / threads_per_block.x))
            ), 
            MAX(1, 
                MAX(ceil(tw->columns / threads_per_block.y), ceil(nn->layers[l]->delta->columns / threads_per_block.y))
            )
        );

        gpu_matrix_dot<<< n_blocks, threads_per_block >>>(tw, nn->layers[l]->delta, delta_tmp);

        cudaGetLastError();
        CHECK_ERROR(cudaDeviceSynchronize());

        matrix_function(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        destroy_matrix(tw);
        destroy_matrix(delta_tmp);
        destroy_matrix(dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        matrix_transpose(nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T
        // matrix_dot(nn->layers[l]->delta, ta, w1); // w1 <- delta^l x (a^(l-1))^T

        dim3 threads_per_block(16, 16);
        dim3 n_blocks(
            MAX(1, 
                MAX(ceil(nn->layers[l]->delta->rows / threads_per_block.x), ceil(ta->rows / threads_per_block.x))
            ), 
            MAX(1, 
                MAX(ceil(nn->layers[l]->delta->columns / threads_per_block.y), ceil(ta->columns / threads_per_block.y))
            )
        );

        gpu_matrix_dot<<< n_blocks, threads_per_block >>>(nn->layers[l]->delta, ta, w1);

        cudaGetLastError();
        CHECK_ERROR(cudaDeviceSynchronize());

        matrix_scalar(w1, nn->alpha / nn->minibatch_size, w1); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus(nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        destroy_matrix(w1);
        destroy_matrix(ta);

        matrix_t *one, *b1;
        b1 = alloc_matrix(nn->layers[l]->number_of_neurons, 1);
        one = alloc_matrix(nn->minibatch_size, 1);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        // matrix_dot(nn->layers[l]->delta, one, b1); // b1 <- delta^l x 1^T

        dim3 threads_per_block2(16, 16);
        dim3 n_blocks2(
            MAX(1, 
                MAX(ceil(nn->layers[l]->delta->rows / threads_per_block2.x), ceil(one->rows / threads_per_block2.x))
            ), 
            MAX(1, 
                MAX(ceil(nn->layers[l]->delta->columns / threads_per_block2.y), ceil(one->columns / threads_per_block2.y))
            )
        );

        gpu_matrix_dot<<< n_blocks2, threads_per_block2 >>>(nn->layers[l]->delta, one, b1);

        cudaGetLastError();
        CHECK_ERROR(cudaDeviceSynchronize());

        matrix_scalar(b1,  nn->alpha / nn->minibatch_size, b1); // b1 <- alpha / m . delta^l x 1^T
        matrix_minus(nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T
        
        destroy_matrix(one);
        destroy_matrix(b1);
    }
}