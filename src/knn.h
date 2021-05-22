#pragma once

#include "types.h"
#include <vector>


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors, bool conPeso);

    void fit(Matrix X, IVector y);

    IVector predict(Matrix X);
private:
    
    bool conPeso;
    unsigned n_neighbors;
    Matrix images;
    IVector keys;
};
