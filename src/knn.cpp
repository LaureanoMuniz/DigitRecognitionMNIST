#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <queue>

using namespace std;

bool paircmp(pair<double,int> a, pair<double,int> b){
    return a.first > b.first;
}


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
        for(unsigned i = 0; i < this->n_neighbors; ++i) {
            neighbors[i] = {(X.row(k) - this->images.row(i)).squarednorm(), this->keys[i]};
        }
        
        std::priority_queue<pair<double,int>,std::vector<pair<double,int>>,bool (&)(std::pair<double, int>, std::pair<double, int>)> priorityQ_neighbors(paircmp,neighbors);

        for(unsigned i = this->n_neighbors; i < data; ++i) {
            priorityQ_neighbors.push({(X.row(k) - this->images.row(i)).squarednorm(), this->keys[i]});
            priorityQ_neighbors.pop();
        }
        
        std::map<int, unsigned> histogram;
        for(unsigned i = 0; i < priorityQ_neighbors.size(); i++){
            auto p = priorityQ_neighbors.top();
            histogram[p.second]++;  
            priorityQ_neighbors.pop();
        } 
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
