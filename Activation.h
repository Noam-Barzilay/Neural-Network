#include "Matrix.h"

#ifndef ACTIVATION_H
#define ACTIVATION_H


namespace activation
{
    typedef Matrix (*act_type)(const Matrix &matrix);
    Matrix relu (const Matrix &matrix);
    Matrix softmax (const Matrix &matrix);
}

#endif //ACTIVATION_H
