
#pragma once
#include <vector>
#include <cmath>

namespace ml {

inline double mse(const std::vector<double>& y, const std::vector<double>& yhat){
    double s=0; size_t n=y.size();
    for(size_t i=0;i<n;++i){ double d=y[i]-yhat[i]; s+=d*d; }
    return s / (n? n:1);
}

inline double accuracy(const std::vector<int>& y, const std::vector<int>& yhat){
    size_t c=0, n=y.size();
    for(size_t i=0;i<n;++i) if(y[i]==yhat[i]) ++c;
    return n? (double)c/n : 0.0;
}

} // namespace ml
