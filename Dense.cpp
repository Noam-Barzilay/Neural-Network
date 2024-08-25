#include "Dense.h"

Dense::Dense(const Matrix& weights, const Matrix& bias,
             activation::act_type other_func){
    if (bias.get_cols() != 1){
        throw std::domain_error("Error: bias is not a vector");
    }
    if (weights.get_rows() != bias.get_rows()){
        throw std::domain_error("Error: "
                                "weights and bias row's should be equal");
    }
    _weights = weights; _bias = bias; func = other_func;
}

Matrix Dense::get_weights() const{
    return _weights;
}

Matrix Dense::get_bias() const{
    return _bias;
}

activation::act_type Dense::get_activation() const{
    return func;
}

Matrix Dense::operator()(Matrix& input) const{
    input.vectorize();
    return func((_weights*input) + _bias);
}

