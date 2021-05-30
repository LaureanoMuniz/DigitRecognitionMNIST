#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <sstream>

#include <Eigen/Core>
#include <Eigen/LU>
#include "types.h"
#include "knn.h"
#include "pca.h"

using namespace emscripten;

struct Shape {
  int rows, cols;
};

EMSCRIPTEN_BINDINGS(Metnum) {
  value_array<Shape>("Shape")
    .element(&Shape::rows)
    .element(&Shape::cols)
    ;

  class_<Matrix>("EigenMatrix")
    .constructor<int, int>()
    .class_function("fromArray", optional_override([](const emscripten::val &v, Shape shape) {
      // This function uses the technique from this issue but without first passing through a std::vector.
      //    https://github.com/emscripten-core/emscripten/pull/5655#issuecomment-520378179
      const auto l = v["length"].as<unsigned>();
      assert(l == shape.rows * shape.cols);
      Matrix *m = new Matrix(shape.rows, shape.cols);
      emscripten::val memoryView{emscripten::typed_memory_view(l, m->data())};
      memoryView.call<void>("set", v);
      return m;
    }), allow_raw_pointers())
    .function("get", optional_override([](const Matrix& self, int i, int j) {
      return self(i, j);
    }))
    .function("set", optional_override([](Matrix& self, int i, int j, double value) {
      self(i, j) = value;
    }))
    ;

  class_<Vector>("EigenVector")
    .constructor<int>()
    .class_function("fromArray", optional_override([](const emscripten::val &v) {
      // This function uses the technique from this issue but without first passing through a std::vector.
      //    https://github.com/emscripten-core/emscripten/pull/5655#issuecomment-520378179
      const auto l = v["length"].as<unsigned>();
      // assert(l == shape.rows * shape.cols);
      Vector *m = new Vector(l);
      emscripten::val memoryView{emscripten::typed_memory_view(l, m->data())};
      memoryView.call<void>("set", v);
      return m;
    }), allow_raw_pointers())
    .function("get", optional_override([](const Vector& self, int i) {
      return self(i);
    }))
    .function("set", optional_override([](Vector& self, int i, double value) {
      self(i) = value;
    }))
    ;

  class_<IVector>("EigenIVector")
    .constructor<int>()
    .class_function("fromArray", optional_override([](const emscripten::val &v) {
      // This function uses the technique from this issue but without first passing through a std::vector.
      //    https://github.com/emscripten-core/emscripten/pull/5655#issuecomment-520378179
      const auto l = v["length"].as<unsigned>();
      // assert(l == shape.rows * shape.cols);
      IVector *m = new IVector(l);
      emscripten::val memoryView{emscripten::typed_memory_view(l, m->data())};
      memoryView.call<void>("set", v);
      return m;
    }), allow_raw_pointers())
    .function("get", optional_override([](const IVector& self, int i) {
      return self(i);
    }))
    .function("set", optional_override([](IVector& self, int i, int value) {
      self(i) = value;
    }))
    ;

  class_<KNNClassifier>("KNNClassifier")
    .constructor<unsigned, bool>()
    .function("fit", optional_override([](KNNClassifier &self, Matrix X, IVector y) {
      self.fit(X, y);
    }))
    .function("predict", optional_override([](KNNClassifier &self, Matrix X) {
      return self.predict(X);
    }))
    ;

  class_<PCA>("PCA")
    .constructor<unsigned>()
    .function("fit", optional_override([](PCA &self, Matrix X) {
      self.fit(X);
    }))
    .class_function("from_proj", optional_override([](int alpha, Matrix X) {
      return PCA::from_proj(alpha, X);
    }))
    .function("transform", optional_override([](PCA &self, Matrix X) {
      return self.transform(X);
    }))
    ;
}
