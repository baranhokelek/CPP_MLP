#pragma once
#include "matrix.h"
#include <random>
#include <utility>
#include <cassert>
#include <math.h>

using namespace linalg;
namespace nn
{
    inline auto sigmoid(double x)
    {
        double result = 1.0 / (1 + exp(-x));
        return result;
    }

    inline auto d_sigmoid(double x)
    {
        return (x * (1 - x));
    }

    template <typename T>
    class MLP
    {
    public:
        std::vector<size_t> units_per_layer;
        std::vector<Matrix<T>> bias_vectors;
        std::vector<Matrix<T>> weight_matrices;
        std::vector<Matrix<T>> activations;

        double lr;

        /* Constructor */
        explicit MLP(std::vector<size_t> units_per_layer, double lr = 0.001) : units_per_layer(units_per_layer),
                                                                               lr(lr),
                                                                               bias_vectors(),
                                                                               weight_matrices(),
                                                                               activations()
        {
            for (size_t i = 0; i < units_per_layer.size() - 1; i++)
            {
                size_t out_channels = units_per_layer[i + 1];
                size_t in_channels = units_per_layer[i];

                Matrix<T> W = mtx<T>::randn(out_channels, in_channels);
                weight_matrices.push_back(W);
                Matrix<T> b = mtx<T>::randn(out_channels, 1);
                bias_vectors.push_back(b);
                activations.resize(units_per_layer.size()); //??? why are we doing this in each iteration? size doesn't depend on i
            }
        }
        /* Forward */

        auto forward(Matrix<T> x)
        {
            assert(get<0>(x.shape) == units_per_layer[0] && get<1>(x.shape));

            activations[0] = x;
            Matrix prev(x);
            for (int i = 0; i < units_per_layer.size() - 1; ++i)
            {
                Matrix y = weight_matrices[i].matmul(prev);
                y = y + bias_vectors[i];
                y = y.apply_function(nn::sigmoid);
                activations[i + 1] = y;
                prev = y;
            }
            return prev;
        }
        Matrix<T> operator()(Matrix<T> x)
        {
            return forward(x);
        }

        /* Backward */

        void backprop(Matrix<T> target)
        {
            assert(get<0>(target.shape) == units_per_layer.back());
            auto y = target;
            auto y_hat = activations.back();
            auto error = (target - y_hat);

            for (int i = weight_matrices.size() - 1; i >= 0; --i)
            {
                auto Wt = weight_matrices[i].T();
                auto prev_errors = Wt.matmul(error);
                auto d_outputs = activations[i + 1].apply_function(d_sigmoid);
                auto gradients = error.multiply_elementwise(d_outputs);
                // gradients = gradients.clip();
                gradients = gradients.multiply_scalar(lr);
                auto a_trans = activations[i].T();
                auto weight_gradients = gradients.matmul(a_trans);

                bias_vectors[i] = bias_vectors[i].add(gradients);
                weight_matrices[i] = weight_matrices[i].add(weight_gradients);
                error = prev_errors;
            }

            int j{5};
        }
    };
}