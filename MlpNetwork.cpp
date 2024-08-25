#include "MlpNetwork.h"
#define OUTPUT_ARR_SIZE 10

MlpNetwork::MlpNetwork(const Matrix weights[MLP_SIZE],
                       const Matrix biases[MLP_SIZE]){
    for (int i = 0; i < MLP_SIZE - 1; i++){
        // check if sizes are the same
        if (weights[i].get_rows() !=  weights[i+1].get_cols()){
            throw std::domain_error("error in sizes");
        }
    }
    for (int i = 0; i < MLP_SIZE; i++) {
        _weights_arr[i] = weights[i];
        _biases_arr[i] = biases[i];
    }
}

Matrix MlpNetwork::get_dense(const Matrix& weight, const Matrix& bias,
                         activation::act_type act_func, Matrix& input) const {
    Dense dense = Dense(weight, bias, act_func);
    Matrix res = dense(input);
    return res;
}


digit MlpNetwork::operator()(Matrix& img) const{
    // setting the weights
    Matrix w1 = _weights_arr[0], w2 = _weights_arr[1],
            w3 = _weights_arr[2], w4 = _weights_arr[3];
    // setting the biases
    Matrix b1 = _biases_arr[0], b2 = _biases_arr[1],
            b3 = _biases_arr[2], b4 = _biases_arr[3];
    // setting the layers
    Matrix r1 = get_dense(w1, b1, activation::relu, img);
    Matrix r2 = get_dense(w2, b2, activation::relu, r1);
    Matrix r3 = get_dense(w3, b3, activation::relu, r2);
    Matrix r4 = get_dense(w4, b4, activation::softmax, r3);
    // returning the highest likely digit
    digit max;
    max.probability = r4[0];
    max.value = 0;
    for (int i = 0; i < OUTPUT_ARR_SIZE; i++) {
        if (r4[i] > max.probability){
            max.probability = r4[i];
            max.value = i;
        }
    }
    return max;
}
