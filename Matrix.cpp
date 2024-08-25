#include <cmath>
#include <fstream>
#include <iostream>
#include "Matrix.h"
#define OUTPUT_CONST 0.1

int Matrix::get_rows() const{
    return _rows;
}

int Matrix::get_cols() const{
    return _cols;
}

Matrix::Matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0){
        throw std::domain_error("matrix should be at least 1x1");
    }
    _rows = rows; _cols = cols;
    // create a 1d array
    arr = new float[_rows * _cols];
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            (*this)(i, j) = 0.0;
        }
    }
}

Matrix::Matrix(const Matrix& other_matrix): Matrix(other_matrix.get_rows(),
                                                   other_matrix.get_cols())
{
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++){
            (*this)(i, j) = other_matrix(i, j);
        }
    }
}

Matrix::~Matrix(){
    delete[] arr;
}

Matrix& Matrix::transpose(){
    Matrix transposed(_cols, _rows);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            transposed(j, i) = (*this)(i, j);
        }
    }
    *this = transposed;
    return *this;
}

Matrix& Matrix::vectorize(){
    _rows *= _cols;
    _cols = 1;
    return *this;
}

void Matrix::plain_print() const{
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::dot(const Matrix& m) const{
    if (m.get_rows() != _rows || m.get_cols() != _cols){
        throw std::domain_error("matrices should be of the same size");
    }
    Matrix result(_rows, _cols);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            result(i, j) = ((*this)(i, j) * m(i, j));
        }
    }
    return result;
}

float Matrix::sum() const{
    float sum = 0;
    for (int i = 0; i < _rows; i++){
        for (int j = 0; j < _cols; ++j) {
            sum += (*this)(i, j);
        }
    }
    return sum;
}


float Matrix::norm() const{
    double sum = 0;
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            sum += std::pow((*this)(i, j), 2);
        }
    }
    return std::sqrt(sum);
}

int Matrix::argmax() const{
    // first, find the largest number in the matrix
    float max_element = (*this)(0, 0);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            if ((*this)(i, j) > max_element){
                max_element = (*this)(i, j);
            }
        }
    }
    // check if A[a.argmax()] is the largest element in matrix A
    for (int idx = 0; idx < _rows * _cols; idx++) {
        if ((*this)[idx] == max_element){
            return idx;
        }
    }
    return 0;
}

Matrix Matrix::operator+(const Matrix& other) const{
    if (other.get_rows() != _rows || other.get_cols() != _cols){
        throw std::domain_error("matrices should be of the same size");
    }
    Matrix result(_rows, _cols);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            result(i, j) = ((*this)(i, j) + other(i, j));
        }
    }
    return result;
}

Matrix& Matrix::operator=(const Matrix& other){
    if (&other != this){
        delete[] arr;
        _rows = other.get_rows();
        _cols = other.get_cols();
        arr = new float [_rows * _cols];
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++){
                (*this)(i, j) = other(i, j);
            }
        }
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& other) const{
    // to multiply matrices, number of columns in first matrix should
    // be equal to the number of rows in the second matrix
    if (_cols != other.get_rows()){
        throw std::domain_error(
                "to multiply matrices, number of columns in first "
        "matrix should be equal to the number of rows in the second matrix");
    }
    Matrix result(_rows, other.get_cols());
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < other.get_cols(); j++) {
            result(i, j) = 0;  // by default
            // number of columns in 1st matrix
            // == number of rows in 2nd matrix
            for (int k = 0; k < _cols; k++) {
                result(i, j) += ((*this)(i, k) * other(k, j));
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(float c) const{
    Matrix result(_rows, _cols);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            result(i, j) = (*this)(i, j) * c;
        }
    }
    return result;
}

Matrix operator*(float c, const Matrix& other){
    return other * c;
}

Matrix& Matrix::operator+=(const Matrix& other){
    if (other.get_rows() != _rows || other.get_cols() != _cols){
        throw std::domain_error("matrices should be of the same size");
    }
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            (*this)(i, j) += other(i, j);
        }
    }
    return *this;
}

float& Matrix::operator()(int i, int j) {
    if ((i >= 0 && i < _rows) && (j >= 0 && j < _cols)){
        return arr[i*_cols + j];
    }
    else{
        throw std::out_of_range("indices must be in range");
    }
}

float Matrix::operator()(int i, int j) const{
    if ((i >= 0 && i < _rows) && (j >= 0 && j < _cols)){
        return arr[i*_cols + j];
    }
    else{
        throw std::out_of_range("indices must be in range");
    }
}

float& Matrix::operator[](int k) {
    if (k >= 0 && k < _rows*_cols){
        // isomorphic mapping
        return arr[k];
    }
    else{
        throw std::out_of_range("indices must be in range");
    }
}

float Matrix::operator[](int k) const{
    if (k >= 0 && k < _rows*_cols){
        // isomorphic mapping
        return arr[k];
    }
    else{
        throw std::out_of_range("indices must be in range");
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& other){
    for (int i = 0; i < other.get_rows(); i++) {
        for (int j = 0; j < other.get_cols(); j++) {
            if (other(i, j) > OUTPUT_CONST){
                os << "**";
            }
            else{
                os << "  ";
            }
        }
        os << std::endl;
    }
    return os;
}

std::istream& operator>>(std::istream& is, const Matrix& other){
    long int array_size_bytes = (other.get_rows()
            * other.get_cols()) * sizeof (float);
    is.seekg (0, std::ios_base::end);
    long int file_size_bytes = is.tellg();
    if (file_size_bytes != array_size_bytes)
    {
        throw std::domain_error("length error");
    }
    is.seekg (0, std::ios_base::beg);
    if (!is.read ((char *)other.arr, array_size_bytes))
    {
        throw std::out_of_range("out of range");
    }
    return is;
}

