#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

/**
 * A simple and pure C implementation of back propagation, 
 * training and forward propagation of an artificial  neural network.
 * Author > Pedro Lara
 *
 */

typedef struct __layer {

    double ** W;
    double *  delta;
    double *  output;

    int n_neurons_curr;
    int n_neurons_prev;

    double (*activation)( double );
    double (*activation_derivative)( double );
    
} layer;



typedef struct __network {

    layer ** layers; // we need only store (n_layers - 1)
    int    n_layers;
    
    double lrate; // learning rate

    double (*cost_loss)( double *, double *, int );

} network;


/**
 * Gera um aleatório entre (a,b)
 */

double randr(double a, double b) {
    return ((rand() % 10000000) / 10000000.)*(b-a)+a;
}


layer * layer_new( int n_neurons_curr, int n_neurons_prev, double (activation)( double ), double (activation_derivative)( double ) ) {
    layer * L = (layer *) malloc( sizeof( layer) );

    L->n_neurons_curr = n_neurons_curr;
    L->n_neurons_prev = n_neurons_prev;
    L->activation = activation;
    L->activation_derivative = activation_derivative;
    L->delta = (double *) malloc( sizeof( double ) * n_neurons_curr );
    L->output = (double *) malloc( sizeof( double ) * n_neurons_curr );
    
    L->W = (double **) malloc( sizeof( double * ) * n_neurons_curr );

    for( int i = 0; i < n_neurons_curr; i++ ) {
        L->W[i] = (double *) malloc( sizeof( double ) * (n_neurons_prev + 1) ); // + 1 because last position in reserved to bias
    }
    /**
     * Random inicialization of weights
     */ 
    for( int i = 0; i < n_neurons_curr; i ++ ) {
        for( int j = 0; j < n_neurons_prev+1; j++ ) {
            L->W[i][j] = randr(0.0, 0.5);    
        }
    }
    return L;
}

void layer_free( layer * L ) {
    for( int i = 0; i < L->n_neurons_curr; i++ ) {
        free( L->W[i] );
    }

    free( L->W );
    free( L->output );
    free( L->delta );
    free( L );
}


void layer_processing( double * X, layer * L ) {
    for( int i = 0; i < L->n_neurons_curr; i ++ ) {
        double s = 0.;
        for( int j = 0; j < L->n_neurons_prev; j++ ) {
            s += L->W[i][j] * X[j];
        }
        s += L->W[i][L->n_neurons_prev]; // Sum the bias
        L->output[i] = L->activation( s );
    }
}


void network_free( network * n ) {
    for( int i = 0; i < n->n_layers - 1; i++ ) {
        layer_free( n->layers[i] );
        
    }
    free( n->layers );
    free( n );
}

network * network_new(  int n_neurons_input, 
                        int n_neurons_hide, 
                        int n_neurons_output, 
                        double lrate,
                        double (activation)( double ),
                        double (activation_derivative)( double ),
                        double (cost_loss)( double *, double *, int ) ) {
    network * n =  (network *) malloc( sizeof( network )  );
    n->lrate = lrate;
    n->n_layers = 3;
    n->layers = (layer **) malloc( sizeof( layer * ) * (n->n_layers - 1) );
    n->layers[0] = layer_new( n_neurons_hide, n_neurons_input,  activation, activation_derivative );
    n->layers[1] = layer_new( n_neurons_output, n_neurons_hide, activation, activation_derivative );
    n->cost_loss = cost_loss;
    return n;
}

/**
 * result is in last layer output
 */

void network_predict( double * input, network * n ) {
    layer_processing( input               , n->layers[0] );
    layer_processing( n->layers[0]->output, n->layers[1] );
}



double network_backpropagation( double ** X, double ** Y, int train_size, network * n ) {
    double total_loss = 0.;
    double error = 0.;
    for( int i = 0; i < train_size; i++ ) {
        network_predict( X[i], n );
        total_loss += n->cost_loss( Y[i], n->layers[1]->output, n->layers[1]->n_neurons_curr );
        int n_layer = (n->n_layers - 2); 
        for( int j = 0; j < n->layers[n_layer]->n_neurons_curr; j++ ) {
            n->layers[n_layer]->delta[j] = (n->layers[n_layer]->output[j] - Y[i][j]) * ( n->layers[n_layer]->output[j] );
        }

        for( int l = (n->n_layers - 2); l >= 1; l-- ) {
            for( int j = 0; j < n->layers[l]->n_neurons_prev; j++ ) {
                error = 0.0;
                for( int k = 0; k < n->layers[l]->n_neurons_curr; k++ ) {
                    error += n->layers[l]->W[k][j] * n->layers[l]->delta[k];
                }
                n->layers[l-1]->delta[j] = error * n->layers[l-1]->activation_derivative( n->layers[l-1]->output[j] ); 
            }
        }

        for( int l = (n->n_layers - 2); l >= 0; l-- ) {
            for( int j = 0; j < n->layers[l]->n_neurons_curr ; j++ ) {
                for( int k = 0; k < n->layers[l]->n_neurons_prev; k++ ) {
                    double * X_out_prev;
                    if( l == 0 )
                        X_out_prev = X[i];
                    else
                        X_out_prev = n->layers[l-1]->output;
                    
                    n->layers[l]->W[j][k] -= n->lrate * n->layers[l]->delta[j] * X_out_prev[k]; 
                    
                }
                n->layers[l]->W[j][n->layers[l]->n_neurons_curr] -= n->lrate * n->layers[l]->delta[j];
            }
        }
    }
    return total_loss;
}


void network_train( double ** X, double ** Y, int train_size, network * n, int max_epocs, double min_error ) {
    double error = min_error + 1;
    int i = 0;
    while(  i < max_epocs &&  error > min_error ) {
        error = network_backpropagation( X, Y, train_size, n );
        printf("[%d] MSE %lf\n", i,  error );
        i++;
    }
}

double sigmoid( double x ) {
    return 1./(1. + exp(-x) );
}

double sigmoid_derivative( double x ) {
    return x*(1.-x);
}

double mse( double * x, double * y, int size ) {
    double s = 0.;
    for( int i = 0; i < size; i++ ) {
        s += (x[i] - y[i])*(x[i] - y[i]);
    }    
    return s / size;
}

int main() {
    
    srand( time(NULL) );
    
    
    int ni = 3;     // número de neurônios na camada de entrada
    int nh = 3;     // número de neurônios na camada de oculta
    int no = 1;     // número de neurônios na camada de saída
    
    double lrate = 0.6;
    
    int train_size = 8;
    
    double ** X = (double **) malloc( sizeof(double *) * train_size );
    double ** Y = (double **) malloc( sizeof(double *) * train_size );
    
    
    for( int i = 0; i < train_size; i++ ) {
        X[i] = (double *) malloc( sizeof(double) * ni );
        Y[i] = (double *) malloc( sizeof(double) * no );
    }
    
    
    /**
     * XOR 3 bit PORT
     */
   
     X[0][0] = 0; X[0][1] = 0; X[0][2] = 0; Y[0][0] = 0;
     X[1][0] = 0; X[1][1] = 0; X[1][2] = 1; Y[1][0] = 1;
     X[2][0] = 0; X[2][1] = 1; X[2][2] = 0; Y[2][0] = 1;
     X[3][0] = 0; X[3][1] = 1; X[3][2] = 1; Y[3][0] = 0;
     X[4][0] = 1; X[4][1] = 0; X[4][2] = 0; Y[4][0] = 1;
     X[5][0] = 1; X[5][1] = 0; X[5][2] = 1; Y[5][0] = 0;
     X[6][0] = 1; X[6][1] = 1; X[6][2] = 0; Y[6][0] = 0;
     X[7][0] = 1; X[7][1] = 1; X[7][2] = 1; Y[7][0] = 1;
     
    network * net = network_new( ni, 
                                 nh, 
                                 no, 
                                 lrate,
                                 sigmoid,
                                 sigmoid_derivative,
                                 mse );

    network_train(  X,  Y,  train_size, net, 1000000, 10e-8 );

    
    for( int i = 0; i < train_size; i++ ) {

        network_predict( X[i], net );

        printf("%lf\n", net->layers[1]->output[0] );
    }



    network_free( net );
    
    for( int i = 0; i < train_size; i++ ) {
        free( X[i] ); free( Y[i] );
    }
    free( X ); free( Y );

    return 0;
}
