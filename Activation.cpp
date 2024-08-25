#include "Activation.h"


Matrix activation::relu(const Matrix& matrix){
    Matrix temp(matrix.get_rows(), matrix.get_cols());
    for (int k = 0; k < temp.get_rows() * matrix.get_cols(); k++) {
        if (matrix[k] >= 0){
            temp[k] = matrix[k];
        }
        else{
            temp[k] = 0;
        }
    }
    return temp;
}

Matrix activation::softmax(const Matrix& matrix){
    // create an exp coordinated version of the matrix
    Matrix exp_res(matrix.get_rows(), matrix.get_cols());
    float sigma = 0;
    for (int k = 0; k < matrix.get_rows() * matrix.get_cols(); k++) {
        sigma += (std::exp(matrix[k]));
        exp_res[k] = std::exp(matrix[k]);
    }
    return (1/sigma) * exp_res;
}

