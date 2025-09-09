
#include "ml/linear_regression.hpp"
#include "ml/metrics.hpp"
#include <vector>
#include <cassert>

int main(){
    using namespace ml;
    std::vector<std::vector<double>> X; std::vector<double> y;
    for(int i=0;i<20;++i){ X.push_back({(double)i}); y.push_back(3.0*i - 2.0); }
    LinearRegression lr; lr.lr = 0.02; lr.epochs = 600;
    lr.fit(X, y);
    auto yhat = lr.predict(X);
    assert(mse(y, yhat) < 1e-2);
    return 0;
}
