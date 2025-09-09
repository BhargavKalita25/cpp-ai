
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace ml {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

inline std::vector<std::string> split(const std::string& s, char sep=','){
    std::vector<std::string> out; std::string cur; std::stringstream ss(s);
    while(std::getline(ss, cur, sep)) out.push_back(cur);
    return out;
}

inline void load_csv(const std::string& path, Matrix& X, Vector& y, bool header=false){
    std::ifstream f(path);
    if(!f) throw std::runtime_error("cannot open csv: " + path);
    std::string line;
    if(header && std::getline(f, line)) { /* skip header */ }
    while(std::getline(f, line)){
        if(line.empty()) continue;
        auto toks = split(line, ',');
        if(toks.size() < 2) continue;
        std::vector<double> row;
        for(size_t i=0;i+1<toks.size();++i) row.push_back(std::stod(toks[i]));
        X.push_back(row);
        y.push_back(std::stod(toks.back()));
    }
}

inline void train_test_split(const Matrix& X, const Vector& y,
                             Matrix& Xtr, Vector& ytr, Matrix& Xte, Vector& yte,
                             double test_ratio=0.2, unsigned seed=42){
    std::vector<size_t> idx(X.size());
    for(size_t i=0;i<idx.size();++i) idx[i]=i;
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    size_t testN = static_cast<size_t>(X.size()*test_ratio);
    for(size_t k=0;k<idx.size();++k){
        size_t i = idx[k];
        if(k<testN){ Xte.push_back(X[i]); yte.push_back(y[i]); }
        else { Xtr.push_back(X[i]); ytr.push_back(y[i]); }
    }
}

} // namespace ml
