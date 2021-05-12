#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
    this->n_neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix X, IVector y)
{
    this->images = X;
    this->keys = y;
}


IVector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    unsigned tests = X.rows();
    unsigned data = this->images.rows();
    auto ret = IVector(tests);

    for(unsigned k = 0; k < tests; ++k) {
        std::vector<pair<double, int>> neighbors(data);
        for(unsigned i = 0; i < data; ++i) {
            neighbors[i] = {(X.row(k) - this->images.row(i)).norm(), this->keys[i]};
        }
        sort(neighbors.begin(), neighbors.end());
        neighbors.erase(neighbors.begin() + this->n_neighbors, neighbors.end());

        std::map<int, unsigned> histogram;
        for(auto p : neighbors) histogram[p.second]++;
        int mode;
        unsigned amount = 0;
        for(auto p : histogram) if(p.second > amount) {
            amount = p.second;
            mode = p.first;
        }

        ret[k] = mode;
    }

    return ret;
}
