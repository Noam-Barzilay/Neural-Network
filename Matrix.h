// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H
#include <cmath>
#include <fstream>
#include <iostream>

/**
 * @struct matrix_dims
 * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
 */

class Matrix
{
private:
    int _rows, _cols;
    float *arr;
public:
    struct dims
    {
        int rows, cols;
    };
    Matrix(int rows=1, int cols=1);
    Matrix (const Matrix& other_matrix);
    ~Matrix();
    int get_rows() const;
    int get_cols() const;
    Matrix& transpose();
    Matrix& vectorize();
    void plain_print() const;
    Matrix dot(const Matrix& m) const;
    float sum() const;
    float norm() const;
    int argmax() const;
    // operators
    Matrix operator+(const Matrix& other) const;
    Matrix& operator=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float c) const;
    friend Matrix operator*(float c, const Matrix& other);
    Matrix& operator+=(const Matrix& other);
    float& operator()(int i, int j) ;
    float operator()(int i, int j) const;
    float& operator[](int k);
    float operator[](int k) const;
    friend std::ostream& operator<<(std::ostream& os, const Matrix& other);
    friend std::istream& operator>>(std::istream& is, const Matrix& other);
};

#endif //MATRIX_H//
