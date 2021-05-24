#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <queue>

using namespace std;

bool paircmp(pair<double,int> a, pair<double,int> b){
    return a.first < b.first;
}


KNNClassifier::KNNClassifier(unsigned int n_neighbors, bool conPeso)
{
    this->n_neighbors = n_neighbors;
    this->conPeso = conPeso;
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
   
    #pragma omp parallel for
    for(unsigned k = 0; k < tests; ++k) {
        std::vector<pair<double, int>> neighbors(data);
        for(unsigned i = 0; i < data; ++i) {
            neighbors[i] = {(X.row(k) - this->images.row(i)).squaredNorm(), this->keys[i]};
        }
        
        std::nth_element (neighbors.begin(), neighbors.begin()+this->n_neighbors, neighbors.end(),paircmp);
        std::map<int, double> histogram;
        for(unsigned i = 0; i < this->n_neighbors; i++){
            auto p = neighbors[i];
            if (!conPeso){
                histogram[p.second]++;
            }
            else{
                histogram[p.second] += 1/(p.first);
            }
        } 
        int mode;
        double amount = 0;
        for(auto p : histogram) if(p.second > amount) {
            amount = p.second;
            mode = p.first;
        }

        ret[k] = mode;
    }

    return ret;
}
