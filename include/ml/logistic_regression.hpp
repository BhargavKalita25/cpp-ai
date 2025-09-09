
#pragma once
#include <vector>
#include <cmath>

namespace ml {

inline double sigmoid(double z){ return 1.0/(1.0+std::exp(-z)); }

struct LogisticRegression {
    std::vector<double> w;
    double b = 0.0;
    double lr = 0.1;
    int epochs = 500;

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y){
        size_t n=X.size(); if(n==0) return;
        size_t d=X[0].size();
        w.assign(d, 0.0); b=0.0;
        for(int e=0;e<epochs;++e){
            std::vector<double> gradw(d,0.0); double gradb=0.0;
            for(size_t i=0;i<n;++i){
                double z=b; for(size_t j=0;j<d;++j) z += w[j]*X[i][j];
                double p = sigmoid(z);
                double err = p - (double)y[i];
                for(size_t j=0;j<d;++j) gradw[j] += err * X[i][j];
                gradb += err;
            }
            double invN = 1.0/(double)n;
            for(size_t j=0;j<d;++j) w[j] -= lr * invN * gradw[j];
            b -= lr * invN * gradb;
        }
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X, double threshold=0.5) const{
        std::vector<int> out; out.reserve(X.size());
        for(const auto& row: X){
            double z=b; for(size_t j=0;j<row.size();++j) z+=w[j]*row[j];
            out.push_back(sigmoid(z)>=threshold ? 1:0);
        }
        return out;
    }
};

} // namespace ml
