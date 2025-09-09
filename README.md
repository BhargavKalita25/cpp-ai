# C++ ML Starter (Bhargav)

Header-only **C++17** mini-ML toolkit with linear regression, logistic regression, k-NN, simple CSV loader, and basic metrics.

## Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest   # run tests
```

## Run demo
```bash
./ml_demo
```

## Features
- `LinearRegression` (batch gradient descent; MSE)
- `LogisticRegression` (binary classification; sigmoid)
- `KNN` classifier (Euclidean)
- `dataset.hpp` CSV loader + train/test split
- `metrics.hpp` (mse, accuracy)