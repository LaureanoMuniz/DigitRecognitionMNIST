#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components)
{
    this->n_components = n_components;
}

void PCA::fit(Matrix X)
{
    unsigned images = X.rows();
    auto mu = X.colwise().mean();// = (prom c1, prom c2 ,..., )
    X.rowwise() -= mu;
    auto cov_matrix = X.transpose() * X / (images-1);
    this->projection = get_first_eigenvalues(cov_matrix, this->n_components).second;
}



Matrix PCA::transform(Matrix X)
{
    return X * this->projection;
}

Matrix PCA::get_projection() {
    return this->projection;
}

PCA PCA::from_proj(int alpha, Matrix proj) {
    PCA res(alpha);
    res.projection = proj;
    return res;
}
