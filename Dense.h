#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"


class Dense
{
private:
    Matrix _weights; Matrix _bias; activation::act_type func;
public:
    Dense(const Matrix& weights, const Matrix& bias,
          activation::act_type other_fun);
    Matrix get_weights() const;
    Matrix get_bias() const;
    activation::act_type get_activation() const;
    Matrix operator()(Matrix& input) const;
};

#endif //DENSE_H
