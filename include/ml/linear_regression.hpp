
#pragma once
#include <vector>

namespace ml {

struct LinearRegression {
    std::vector<double> w;
    double b = 0.0;
    double lr = 0.01;
    int epochs = 500;

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y){
        size_t n = X.size(); if(n==0) return;
        size_t d = X[0].size();
        w.assign(d, 0.0); b = 0.0;
        for(int e=0;e<epochs;++e){
            std::vector<double> gradw(d,0.0); double gradb=0.0;
            for(size_t i=0;i<n;++i){
                double pred = b;
                for(size_t j=0;j<d;++j) pred += w[j]*X[i][j];
                double err = pred - y[i];
                for(size_t j=0;j<d;++j) gradw[j] += err * X[i][j];
                gradb += err;
            }
            double invN = 1.0 / (double)n;
            for(size_t j=0;j<d;++j) w[j] -= lr * invN * gradw[j];
            b -= lr * invN * gradb;
        }
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const{
        std::vector<double> out; out.reserve(X.size());
        for(const auto& row: X){
            double pred = b;
            for(size_t j=0;j<row.size();++j) pred += w[j]*row[j];
            out.push_back(pred);
        }
        return out;
    }
};

} // namespace ml
