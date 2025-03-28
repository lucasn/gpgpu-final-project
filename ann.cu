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

#define true 1
#define false 0

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

    nn->one = alloc_matrix(1, nn->minibatch_size);
    for (int idx = 0; idx < nn->one->columns*nn->one->rows; idx++)
        nn->one->m[idx] = 1.0;

    nn->one_t = alloc_matrix(nn->minibatch_size, 1);
    for (int idx = 0; idx < nn->one_t->columns*nn->one_t->rows; idx++)
        nn->one_t->m[idx] = 1.0;

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
    int max_number_of_neurons = 0;
    for (int l = 1; l < nn->number_of_layers; l++) {
        max_number_of_neurons = MAX(max_number_of_neurons, nn->layers[l]->number_of_neurons);
    }

    matrix_t *z1 = alloc_matrix(max_number_of_neurons, nn->minibatch_size);
    matrix_t *z2 = alloc_matrix(max_number_of_neurons, nn->minibatch_size);

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        z1->rows = nn->layers[l]->number_of_neurons;
        z2->rows = nn->layers[l]->number_of_neurons;
        
        gpu_matrix_dot_wrapper(nn->layers[l]->weights, nn->layers[l-1]->activations, z1, true);// z1 <- w^l x a^(l-1)
        gpu_matrix_dot_wrapper(nn->layers[l]->biases, nn->one, z2, true); // z2 <- b^l x 1        

        matrix_sum(z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1      

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
    }

    destroy_matrix(z1);
    destroy_matrix(z2);
}

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix(dfzL);

    int max_number_of_neurons = 0;
    for (int l = L; l > 1; l--) {
        max_number_of_neurons = MAX(max_number_of_neurons, nn->layers[l]->number_of_neurons);
    }

    matrix_t *tw, *delta_tmp, *dfz;
    tw = alloc_matrix(max_number_of_neurons, max_number_of_neurons);
    delta_tmp = alloc_matrix(max_number_of_neurons, nn->minibatch_size);
    dfz = alloc_matrix(max_number_of_neurons, nn->minibatch_size);

    for (int l = L; l > 1; l--)
    {
        tw->rows = nn->layers[l-1]->number_of_neurons;
        tw->columns = nn->layers[l]->number_of_neurons;
        delta_tmp->rows = nn->layers[l-1]->number_of_neurons;
        dfz->rows = nn->layers[l-1]->number_of_neurons;

        matrix_transpose(nn->layers[l]->weights, tw); // (w^l)T        

        gpu_matrix_dot_wrapper(tw, nn->layers[l]->delta, delta_tmp, true);

        matrix_function(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))
    }

    destroy_matrix(tw);
    destroy_matrix(delta_tmp);
    destroy_matrix(dfz);

    matrix_t *b1;
    b1 = alloc_matrix(max_number_of_neurons, 1);

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        matrix_transpose(nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T

        gpu_matrix_dot_wrapper(nn->layers[l]->delta, ta, w1, true);

        matrix_scalar(w1, nn->alpha / nn->minibatch_size, w1); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus(nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        //gpu_matrix_scalar_minus_wrapper(nn->layers[l]->weights, w1, nn->alpha / nn->minibatch_size, nn->layers[l]->weights);

        destroy_matrix(w1);
        destroy_matrix(ta);

        b1->rows =nn->layers[l]->number_of_neurons; 

        gpu_matrix_dot_wrapper(nn->layers[l]->delta, nn->one_t, b1, true);

        matrix_scalar(b1,  nn->alpha / nn->minibatch_size, b1); // b1 <- alpha / m . delta^l x 1^T
        matrix_minus(nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T

    }
    destroy_matrix(b1);
}