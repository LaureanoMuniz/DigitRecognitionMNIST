#include <algorithm>
#include <chrono>
#include <iostream>
#include <tuple>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{ 
    Vector b = Vector::Random(X.cols());
    Vector b_anterior;
    double coseno;
    b /= b.norm();
    while(num_iter --> 0) {
        b_anterior = b;
        b = X * b;
        b /= b.norm();
        coseno = b.dot(b_anterior);

        if ((1-eps) < coseno && coseno <= 1) {
            //cout << "Termine en la iteracion" << num_iter << endl;
            break;
        }
    }
    return make_pair(b.dot(X*b), b);
    
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double eps)
{
    assert(X.cols() == X.rows());
    unsigned n = X.cols();
    Matrix A(n, n);

    double delta = Vector::Random(1)[0];
    for(int i = 0; i < X.cols(); ++i) {
        A(i, i) += delta;
    }

    Vector eigvalues(num);
    Matrix eigvectors(n, num);

    for(unsigned i = 0; i < num; ++i) {
        // std::tie(eigvalues(i), eigvectors.col(i)) = power_iteration(X, num_iter, eps);
        auto par = power_iteration(A, num_iter, eps);
        eigvectors.col(i) = par.second;
        eigvalues(i) = par.first;

        A -= (eigvalues(i) * eigvectors.col(i) * eigvectors.col(i).transpose());
        eigvalues(i) -= delta;
    }
    return make_pair(eigvalues, eigvectors);
}

