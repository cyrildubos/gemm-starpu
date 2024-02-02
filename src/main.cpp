#include <starpu.h>

#include "matrix.hpp"

int main(int argc, char **argv) {
  starpu_init(NULL);

  {
    auto m = 4u;
    auto n = 4u;
    auto k = 4u;

    auto u = 2u;
    auto v = 2u;
    auto w = 4u;

    int alpha = 2;
    int beta = 4;

    Matrix<int> a{m, k, u, w};
    Matrix<int> b{k, n, w, v};
    Matrix<int> c{m, n, u, v};

    a.fill_value(2);
    b.fill_value(3);
    c.fill_value(4);

    Matrix<int>::gemm_1d(alpha, a, b, beta, c);

    starpu_task_wait_for_all();

    c.print();
  }

  {
    auto m = 4u;
    auto n = 4u;
    auto k = 4u;

    auto u = 2u;
    auto v = 2u;
    auto w = 2u;

    int alpha = 2;
    int beta = 4;

    Matrix<int> a{m, k, u, w};
    Matrix<int> b{k, n, w, v};
    Matrix<int> c{m, n, u, v};

    a.fill_value(2);
    b.fill_value(3);
    c.fill_value(4);

    Matrix<int>::gemm_2d(alpha, a, b, beta, c);

    starpu_task_wait_for_all();

    c.print();
  }

  starpu_shutdown();

  return 0;
}
