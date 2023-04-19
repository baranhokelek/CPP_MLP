// matrix.h
#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>

namespace linalg
{
    template <typename Type>
    class Matrix
    {
        size_t cols;
        size_t rows;

    public:
        std::vector<Type> data;
        std::tuple<size_t, size_t> shape;
        int numel = rows * cols;

        /* constructors */
        Matrix(size_t rows, size_t cols) : cols(cols), rows(rows), data({})
        {
            data.resize(cols * rows, Type());
            shape = {rows, cols};
        }

        Matrix() : cols(0), rows(0), data({})
        {
            shape = {rows, cols};
        }

        void print_shape()
        {
            std::cout << "Matrix Size([" << rows << ", " << cols << "])" << std::endl;
        }

        void print()
        {
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    std::cout << (*this)(r, c) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        Type &operator()(size_t row, size_t col)
        {
            return data[row * cols + col];
        }

        Matrix matmul(Matrix &target)
        {
            assert(cols == target.rows);
            Matrix output(rows, target.cols);
            for (size_t r = 0; r < output.rows; ++r)
            {
                for (size_t c = 0; c < output.cols; ++c)
                {
                    for (size_t k = 0; k < target.rows; ++k)
                    {
                        output(r, c) += (*this)(r, k) * target(k, c);
                    }
                }
            }
            return output;
        }

        Matrix multiply_elementwise(Matrix &target)
        {
            assert(rows == target.rows);
            assert(cols == target.cols);
            Matrix output(rows, cols);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    output(r, c) += (*this)(r, c) * target(r, c);
                }
            }
            return output;
        }

        Matrix square()
        {
            return multiply_elementwise(*this);
        }

        Matrix multiply_scalar(Type scalar)
        {
            Matrix output(rows, cols);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    output(r, c) += (*this)(r, c) * scalar;
                }
            }
            return output;
        }

        Matrix add(Matrix &target)
        {
            assert(shape == target.shape);
            Matrix output(rows, cols);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    output(r, c) += (*this)(r, c) + target(r, c);
                }
            }
            return output;
        }

        Matrix operator+(Matrix &target)
        {
            return add(target);
        }

        Matrix operator-()
        {
            Matrix output(rows, cols);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    output(r, c) = -(*this)(r, c);
                }
            }
            return output;
        }

        Matrix sub(Matrix &target)
        {
            Matrix neg_target = -target;
            return add(neg_target);
        }

        Matrix operator-(Matrix &target)
        {
            return sub(target);
        }

        Matrix transpose()
        {
            Matrix output(cols, rows);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    output(c, r) += (*this)(r, c);
                }
            }
            return output;
        }

        Matrix T()
        {
            return transpose();
        }

        Matrix apply_function(const std::function<Type(const Type &)> &function)
        {
            Matrix output(rows, cols);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    output(r, c) = function((*this)(r, c));
                }
            }
            return output;
        }

        Matrix clip()
        {
            Matrix output(rows, cols);
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    if (abs((double)(*this)(r, c)) < 0.0001)
                        output(r, c) = 0;
                    else
                        output(r, c) = (*this)(r, c);
                }
            }
            return output;
        }

        bool check_nan()
        {
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    if ((bool)isnan((*this)(r, c)))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        bool check_abnormal()
        {
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    if (((bool)!isnormal((*this)(r, c)) && (double)(*this)(r, c) != 0.0) || (double)(*this)(r, c) >= 3.0)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        void fill_(Type val)
        {
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    (*this)(r, c) = val;
                }
            }
        }
    };

    template <typename T>
    struct mtx
    {
        static Matrix<T> rand(size_t rows, size_t cols)
        {
            Matrix<T> M(rows, cols);
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::uniform_real_distribution<T> d{0, 1};
            for (size_t r = 0; r < rows; r++)
            {
                for (size_t c = 0; c < cols; c++)
                {
                    M(r, c) = d(gen);
                }
            }
            return M;
        }

        static Matrix<T> randn(size_t rows, size_t cols)
        {
            Matrix<T> M(rows, cols);
            std::random_device rd{};
            std::mt19937 gen{rd()};
            T n(M.numel);
            T stdev{1 / sqrt(n)};
            std::normal_distribution<T> d{0, stdev};
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    M(r, c) = d(gen);
                }
            }
            return M;
        }

        static Matrix<T> zeros(size_t rows, size_t cols)
        {
            Matrix<T> M(rows, cols);
            M.fill_(T(0));
            return M;
        }

        static Matrix<T> ones(size_t rows, size_t cols)
        {
            Matrix<T> M(rows, cols);
            M.fill_(T(1));
            return M;
        }
    };
}