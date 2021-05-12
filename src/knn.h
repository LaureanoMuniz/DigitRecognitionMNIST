#pragma once

#include "types.h"
#include <vector>


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, IVector y);

    IVector predict(Matrix X);
private:
    unsigned n_neighbors;
    Matrix images;
    IVector keys;
};
