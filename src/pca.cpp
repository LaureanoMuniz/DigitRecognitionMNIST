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
    this->projection = get_first_eigenvalues(cov_matrix, this->n_components, 5000, 1e-6).second; // Valor en 1e-6 para ser mas permisivo. No estaba parando antes con 1e-16
}



MatrixXd PCA::transform(Matrix X)
{
    return X * this->projection;
}
