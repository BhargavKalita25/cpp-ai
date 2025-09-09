
#include <iostream>
#include "ml/dataset.hpp"
#include "ml/linear_regression.hpp"
#include "ml/logistic_regression.hpp"
#include "ml/knn.hpp"
#include "ml/metrics.hpp"

int main(){
    using namespace ml;

    // Linear Regression demo: y ≈ 2x + 1
    std::vector<std::vector<double>> X; std::vector<double> y;
    for(int i=0;i<50;++i){ X.push_back({(double)i}); y.push_back(2.0*i + 1.0); }
    LinearRegression lr; lr.lr = 0.01; lr.epochs = 800;
    lr.fit(X, y);
    auto yhat = lr.predict(X);
    std::cout << "LR mse: " << mse(y, yhat) << "  w=" << lr.w[0] << " b=" << lr.b << "\n";

    // Logistic Regression demo (AND gate)
    std::vector<std::vector<double>> Xc = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<int> yc = {0,0,0,1};
    LogisticRegression logreg; logreg.lr=0.5; logreg.epochs=800;
    logreg.fit(Xc, yc);
    auto ypred = logreg.predict(Xc);
    std::cout << "LogReg acc (AND): " << accuracy(yc, ypred) << "\n";

    // KNN demo
    KNN knn; knn.k = 3; knn.fit(Xc, yc);
    auto yk = knn.predict(Xc);
    std::cout << "KNN acc (AND): " << accuracy(yc, yk) << "\n";

    return 0;
}
