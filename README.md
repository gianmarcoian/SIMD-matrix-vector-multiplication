# Matrix-Vector Multiplication Optimization

This repository aims to compute and compare different implementations of cache parallelization to compute matrix-vector multiplication. All the different methodologies are compared and presented in a report and in an Excel file for performance comparison.

## Contents

- [Description](#description)
- [Contents](#contents)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Description

Matrix-vector multiplication is a fundamental operation in linear algebra and many scientific computing applications. This repository focuses on optimizing this operation through various cache parallelization techniques.

## Contents

- `Makefile`: Compiles the code.
- `microtime.c` and `microtime.h`: Compute the time of processing.
- `optimized.c`: Main file containing the matrix-vector multiplication to optimize.
- `Report.docx`: Detailed report comparing the results achieved by several implementations.
- `Results.xlsx`: Log for the experimental results.

## Requirements

- GCC (GNU Compiler Collection)
- Microtime library (included)

## Usage

To compile the code, use the provided Makefile:

->bash
make

This will generate the executable file 'optimized'. Then run ./optimized.
The program will execute and output the reuslts.


## Results

The best result achieved was through the use of SIMD instructions (Intel's SSE technology). The detailed performance comparison is provided in the Report.docx and Results.xlsx.

## License
Feel free to adjust the content as needed.






