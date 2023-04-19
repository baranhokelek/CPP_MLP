#include "matrix.h"
#include "nn.h"
#include <fstream>
#include <deque>

auto make_model(size_t in_channels, size_t out_channels, size_t hidden_units_per_layer, int hidden_layers, double lr)
{
    std::vector<size_t> units_per_layer;
    units_per_layer.push_back(in_channels);
    for (int i = 0; i < hidden_layers; i++)
    {
        units_per_layer.push_back(hidden_units_per_layer);
    }
    units_per_layer.push_back(out_channels);
    nn::MLP<double> model(units_per_layer, lr);
    return model;
}

void log(std::ofstream &file, const linalg::Matrix<double> &x, const linalg::Matrix<double> &y, const linalg::Matrix<double> &y_hat)
{
    auto mse = (y.data[0] - y_hat.data[0]);
    mse = mse * mse;
    file << mse << " " << x.data[0] << " " << y.data[0] << " " << y_hat.data[0] << " \n";
}

int main()
{
    int in_channels{1}, out_channels{1};
    int hidden_units_per_layer{8}, hidden_layers{3};
    double lr{0.2};

    auto model = make_model(in_channels, out_channels, hidden_units_per_layer, hidden_layers, lr);
    std::ofstream my_file;
    my_file.open("data.txt");

    int max_iter{20000};
    double mse;

    const double PI{3.14159};
    for (int i = 1; i <= max_iter; ++i)
    {
        auto x = mtx<double>::rand(in_channels, 1).multiply_scalar(PI);
        auto y = x.apply_function([](double v) -> double
                                  { return sin(v) * sin(v); });
        auto y_hat = model.forward(x);
        model.backprop(y);
        log(my_file, x, y, y_hat);
    }

    my_file.close();
}