#include <microtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>  

/*---------ATTENTION PLEASE:----------


The following code contains all the implementations used for the matVecMult function,
and they are all COMMENTED EXCEPT THE BEST IN TERMS OF EFFICIENCY.



----------------------------------------
*/
typedef float* Matrix;

Matrix createMatrix(int rows, int cols) {
  Matrix M;

  M = (Matrix)malloc(rows * cols * sizeof(M[0]));
  if (M == 0)
    fprintf(stderr, "Matrix allocation failed in file %s, line %d\n", __FILE__,
            __LINE__);

  return M;
}

void freeMatrix(Matrix M) {
  if (M) free(M);
}

void initMatrix(Matrix A, int rows, int cols) {
  int i, j;

  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) A[i * cols + j] = 1.0 / (i + j + 2);
}

/*
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
  int i, k;

  memset(C, 0, rows * sizeof(C[0]));

  for (k = 0; k < cols; k++)
    for (i = 0; i < rows; i++)
      C[i] += A[i * cols + k] * B[k];
}
*/

/*
//loop unrolling by factor 16
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        C[i] = 0.0;
    }

    for (j = 0; j < cols; j++) {
        // Loop unrolling by a factor of 16 with result accumulation
        for (i = 0; i < rows; i += 16) {
            float result[16] = {0.0}; 
            for (int k = 0; k < 16; k++) {
                if (i + k < rows) {
                    result[k] = A[(i + k) * cols + j] * B[j];
                }
            }
            for (int k = 0; k < 16; k++) {
                if (i + k < rows) {
                    C[i + k] += result[k];  
                }
            }
        }
    }
}
*/

/*
// Define a struct to pass parameters to each thread
typedef struct {
    int start;
    int end;
    Matrix A;
    Matrix B;
    Matrix C;
    int cols;
} ThreadData;

// Function for each thread to perform part of the computation
void* threadMultiply(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < data->cols; j++) {
            data->C[i] += data->A[i * data->cols + j] * data->B[j];
        }
    }
    pthread_exit(NULL);
}

void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    int numThreads = 4;  // Adjust the number of threads based on your CPU and requirements
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int chunkSize = rows / numThreads;

    for (int i = 0; i < rows; i++) {
        C[i] = 0.0;
    }

    for (int t = 0; t < numThreads; t++) {
        threadData[t].start = t * chunkSize;
        threadData[t].end = (t == numThreads - 1) ? rows : (t + 1) * chunkSize;
        threadData[t].A = A;
        threadData[t].B = B;
        threadData[t].C = C;
        threadData[t].cols = cols;

        pthread_create(&threads[t], NULL, threadMultiply, &threadData[t]);
    }

    for (int t = 0; t < numThreads; t++) {
        pthread_join(threads[t], NULL);
    }
}



*/
/*
//Strassen's algorithm first version, block size=16
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    int i, j, k;
    const int blockSize = 16;  
    for (i = 0; i < rows; i++) {
        C[i] = 0.0;
    }

    for (k = 0; k < cols; k++) {
        for (i = 0; i < rows; i += blockSize) {
            for (j = 0; j < cols; j += blockSize) {
                for (int ii = i; ii < i + blockSize; ii++) {
                    for (int jj = j; jj < j + blockSize; jj++) {
                        if (ii < rows && jj < cols) {
                            C[ii] += A[ii * cols + jj] * B[jj];
                        }
                    }
                }
            }
        }
    }
}
*/
/*
//Strassen's algorithm first version, block size based on my l1d cache size
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    int i, j, k;
    const int L1dCacheSize = 64 * 1024;  // L1d cache size in bytes
    const int blockSize = (L1dCacheSize / (2 * sizeof(float))) / cols;  // Adjusted for L1d cache size

    for (i = 0; i < rows; i++) {
        C[i] = 0.0;
    }

    for (k = 0; k < cols; k++) {
        for (i = 0; i < rows; i += blockSize) {
            for (j = 0; j < cols; j += blockSize) {
                for (int ii = i; ii < i + blockSize; ii++) {
                    for (int jj = j; jj < j + blockSize; jj++) {
                        if (ii < rows && jj < cols) {
                            C[ii] += A[ii * cols + jj] * B[jj];
                        }
                    }
                }
            }
        }
    }
}
*/
/*
void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        C[i] = 0.0;
    }

    for (j = 0; j < cols; j++) {
        for (i = 0; i < rows; i += simdWidth) {
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < simdWidth; k++) {
                if (i + k < rows) {
                    __m128 a = _mm_set1_ps(A[(i + k) * cols + j]);
                    __m128 b = _mm_load_ps(&B[j]);
                    sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
                }
            }
            if (i + simdWidth < rows) {
                _mm_store_ps(&C[i], sum);
            } else {
                // Handle remaining elements when the row count is not a multiple of simdWidth
                float temp[simdWidth];
                _mm_store_ps(temp, sum);
                for (int k = 0; k < simdWidth; k++) {
                    if (i + k < rows) {
                        C[i + k] += temp[k];
                    }
                }
            }
        }
    }
}
*/

void matVecMult(Matrix A, Matrix B, Matrix C, int rows, int cols) {
  int i, k;

  for (i = 0; i < rows; i++) {
    __m128 sum = _mm_setzero_ps();
    for (k = 0; k < cols; k += 4) {
      __m128 aVec = _mm_loadu_ps(&A[i * cols + k]);
      __m128 bVec = _mm_loadu_ps(&B[k]);
      sum = _mm_add_ps(sum, _mm_mul_ps(aVec, bVec));
    }
    float result[4];
    _mm_storeu_ps(result, sum);
    C[i] = result[0] + result[1] + result[2] + result[3];
  }
}


int main(int argc, char** argv) {
  int n, m, p = 1;
  Matrix A, B, C;
  double t, time1, time2;

  if (argc != 3) {
    fprintf(stderr, "USAGE: %s rows cols\n", argv[0]);
    exit(1);
  }

  n = atoi(argv[1]);
  m = atoi(argv[2]);

  A = createMatrix(n, m);
  B = createMatrix(m, p);
  C = createMatrix(n, p);

  initMatrix(A, n, m);
  initMatrix(B, m, p);
  memset(C, 0, n * p * sizeof(C[0]));

  // measure time
  time1 = microtime();
  matVecMult(A, B, C, n, m);
  time2 = microtime();

  t = time2 - time1;

  // Print results
  printf("\nTime = %g us\n", t);
  printf("Timer Resolution = %g us\n", getMicrotimeResolution());
  printf("Performance = %g Gflop/s\n", 2.0 * n * n * 1e-3 / t);
  printf("C[N/2] = %g\n\n", (double)C[n / 2]);

  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);

  return 0;
}
