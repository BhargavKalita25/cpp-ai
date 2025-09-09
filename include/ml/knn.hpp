
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace ml {

inline double l2(const std::vector<double>& a, const std::vector<double>& b){
    double s=0; for(size_t i=0;i<a.size();++i){ double d=a[i]-b[i]; s+=d*d; } return std::sqrt(s);
}

struct KNN {
    int k=3;
    std::vector<std::vector<double>> X;
    std::vector<int> y;

    void fit(const std::vector<std::vector<double>>& X_, const std::vector<int>& y_){
        X = X_; y = y_;
    }

    int predict_one(const std::vector<double>& x) const{
        std::vector<std::pair<double,int>> d;
        d.reserve(X.size());
        for(size_t i=0;i<X.size();++i) d.push_back({l2(x, X[i]), y[i]});
        std::nth_element(d.begin(), d.begin()+std::min((size_t)k,d.size())-1, d.end(),
                         [](auto& a, auto& b){ return a.first < b.first; });
        int vote0=0, vote1=0;
        for(int i=0;i<k && i<(int)d.size(); ++i) (d[i].second==1 ? vote1:vote0)++;
        return vote1>=vote0 ? 1:0;
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& Xq) const{
        std::vector<int> out; out.reserve(Xq.size());
        for(const auto& row: Xq) out.push_back(predict_one(row));
        return out;
    }
};

} // namespace ml
