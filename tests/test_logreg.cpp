
#include "ml/logistic_regression.hpp"
#include "ml/metrics.hpp"
#include <vector>
#include <cassert>

int main(){
    using namespace ml;
    std::vector<std::vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<int> y = {0,0,0,1}; // AND
    LogisticRegression clf; clf.lr=0.5; clf.epochs=1000;
    clf.fit(X, y);
    auto yhat = clf.predict(X);
    assert(accuracy(y, yhat) >= 0.99);
    return 0;
}
