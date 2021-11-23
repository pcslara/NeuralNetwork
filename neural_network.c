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




/**
 * Gera um aleatório entre [0,1)
 */

double randd() {
    return (rand() % 10000000) / 10000000.;
}

/**
 * Gera um aleatório entre (a,b)
 */

double randr(double a, double b) {
    return ((rand() % 10000000) / 10000000.)*(b-a)+a;
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

void processing_layer( double * Y, double * X, double ** W,  double (activation)(double), int nlines, int ncols ) {
    for( int i = 0; i < nlines; i ++ ) {
        double s = 0.;
        for( int j = 0; j < ncols; j++ ) {
            s += W[i][j] * X[j];
        }
        s += W[i][ncols]; // Sum the bias
        Y[i] = activation( s );
    }  
}

void fit( double * Y, double * X, double ** Wi, double ** Wh,  double (activation)(double), int ni, int nh, int no ) {
    
    double * Xh = (double *) malloc( sizeof( double ) * nh );
    processing_layer( Xh, X, Wi, activation, nh, ni );
    processing_layer( Y, Xh, Wh, activation, no, nh );
    free( Xh );

}


double backpropagation(   double * deltaH, 
                          double * deltaO, 
                          double ** X, 
                          double ** Y, 
                          double ** Wh, 
                          double ** Wo,   
                          int training_size, 
                          int ni, 
                          int nh, 
                          int no,
                          double (loss)(double*, double *, int),
                          double (activation)(double),
                          double (activation_derivative)(double),
                          double lrate
                       ) {
    double error = 0.;
    
    double * Xh = (double *) malloc( sizeof( double ) * nh );
    
    double * O = (double *) malloc( sizeof( double ) * no );
    
    double total_error = 0.;
    
    for( int i = 0; i < training_size; i++ ) {
        
        /**
         * Forward propagation
         */
     
        processing_layer( Xh, X[i], Wh, activation, nh, ni);

        
        processing_layer( O, Xh, Wo, activation, no, nh);
        
        
        total_error += loss( Y[i], O, no ); 
        
        /** 
         * Backward propagation
         */
        
        for( int j = 0; j < no; j++ ) {
            deltaO[j] = (O[j] - Y[i][j]) * ( O[j] );
        }
    
        for( int j = 0; j < nh; j++ ) {
            error = 0.0;
            for( int k = 0; k < no; k++ ) {
                error += Wo[k][j] * deltaO[k];
            }
            deltaH[j] = error * activation_derivative( Xh[j] ); 
        }
        
        
        /**
         * Update the weights
         */
        for( int j = 0; j < no; j++ ) {
            error = 0.0;
            for( int k = 0; k < nh; k++ ) {
                Wo[j][k] -=  lrate * deltaO[j] * Xh[k];   
            }
        
            Wo[j][nh] -=  lrate * deltaO[j];  // Update the bias
        }
    
        for( int j = 0; j < nh; j++ ) {
            error = 0.0;
            for( int k = 0; k < ni; k++ ) {
                Wh[j][k] -= lrate * deltaH[j] * X[i][k];
            }
            Wh[j][ni] -=  lrate * deltaH[j]; // Update the bias
        }
        
       

    }

    free( O );
    free( Xh );
    
    return total_error;
}


void train(  double ** X, 
             double ** Y,
             int ni,
             int nh,
             int no,
             int training_size
           ) {
           
    double total_error;
    
    double ** Wh = (double **) malloc( sizeof( double * ) * nh );
    double ** Wo = (double **) malloc( sizeof( double * ) * no );
    
    double * deltaH = (double *) malloc( sizeof( double ) * nh );
    double * deltaO = (double *) malloc( sizeof( double ) * no );
               
    double lrate = 0.7;
    
    int epoch = 100000;
    
    /** 
     * Alocação das matrizes de peso
     */
    for( int i = 0; i < nh; i ++ ) {
        Wh[i] = (double *) malloc( sizeof( double ) * (ni+1) );
    }
    
    for( int i = 0; i < no; i ++ ) {
        Wo[i] = (double *) malloc( sizeof( double ) * (nh+1) );
    }
    
    
    for( int i = 0; i < nh; i ++ ) {
        for( int j = 0; j < ni+1; j++ ) {
            Wh[i][j] = randr(0.0, 0.5);    
        }
    }
    
    for( int i = 0; i < no; i ++ ) {
        for( int j = 0; j < nh+1; j++ ) {
            Wo[i][j] = randr(0.0, 0.5);    
        }
    }
    
    for( int i = 0; i < epoch; i++ ) {
        total_error = backpropagation(    deltaH, 
                                          deltaO, 
                                          X, 
                                          Y, 
                                          Wh, 
                                          Wo, 
                                          training_size, 
                                          ni, 
                                          nh, 
                                          no,
                                          mse,
                                          sigmoid,
                                          sigmoid_derivative,
                                          lrate
                                       );
    
        printf("[%d] MSE = %lf\n", i, total_error );
    
    }       
    
    double * Y_test = (double *) malloc( sizeof( double ) * no );
    
    for( int i = 0; i < training_size; i++ ) {
        
        fit( Y_test, X[i], Wh, Wo, sigmoid, ni, nh, no );
        printf("Expected: %lf Predicted: %lf\n", Y[i][0], Y_test[0] );
    }
    
    for( int i = 0; i < nh; i ++ ) {
        free( Wh[i] );
    }
    
    for( int i = 0; i < no; i ++ ) {
        free( Wo[i] );
    }
    
    free( Wh );
    free( Wo );
    free( Y_test );
    
    free( deltaH );
    free( deltaO );

} 
            



int main() {
    
    srand( time(NULL) );
    
    
    int ni = 3;     // número de neurônios na camada de entrada
    int nh = 3;     // número de neurônios na camada de oculta
    int no = 1;     // número de neurônios na camada de saída
    
    
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
     
     train(  X, 
             Y,
             ni,
             nh,
             no,
             train_size
          );
    
    for( int i = 0; i < train_size; i++ ) {
        free( X[i] ); free( Y[i] );
    }
    free( X ); free( Y );
    
    return 0;
}
