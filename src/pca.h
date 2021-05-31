#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Matrix transform(Matrix X);

    Matrix get_projection();

    static PCA from_proj(int, Matrix);
private:
    Matrix projection;
    unsigned int n_components;
};
