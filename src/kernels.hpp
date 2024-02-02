#pragma once

#include <starpu.h>

template <typename DataType>
void fill_value_cpu(void *buffers[], void *arguments) {
  auto a = (DataType *)STARPU_MATRIX_GET_PTR(*buffers);

  auto u = STARPU_MATRIX_GET_NY(*buffers);
  auto v = STARPU_MATRIX_GET_NX(*buffers);

  DataType value;

  starpu_codelet_unpack_args(arguments, &value);

  std::cout << "INFO\tfill_value_cpu: u = " << u << ", v = " << v
            << ", value = " << value << '\n';

  for (auto x = 0u; x < u; ++x)
    for (auto y = 0u; y < v; ++y)
      a[x + y * u] = value;
}

// /**
//  * Kernel starpu pour le produit de tiles sur cpu
//  *
//  * @brief prend 3 tiles A, B et C et calcul
//  * C * beta = A*alpha x B * alpha
//  *
//  * @param buffers contient 3 handle vers les données
//  * des tiles A, B et C
//  *
//  * @param arguments: pointeur vers un parameter<DataType>
//  * contenant:
//  * - i j et k les dimensions des tiles
//  *      a (ixj)
//  *      b (jxk)
//  *      c (ixk)
//  * - les valeurs alpha et beta de types DataType
//  */
template <typename DataType> void gemm_cpu(void *buffers[], void *arguments) {
  auto a = (DataType *)STARPU_VECTOR_GET_PTR(buffers[0]);
  auto b = (DataType *)STARPU_VECTOR_GET_PTR(buffers[1]);
  auto c = (DataType *)STARPU_VECTOR_GET_PTR(buffers[2]);

  auto u = STARPU_MATRIX_GET_NY(buffers[0]);
  auto v = STARPU_MATRIX_GET_NX(buffers[1]);
  auto w = STARPU_MATRIX_GET_NX(buffers[0]);

  DataType alpha;
  DataType beta;

  starpu_codelet_unpack_args(arguments, &alpha, &beta);

  std::cout << "INFO\tgemm_cpu: u = " << u << ", v = " << v << ", w = " << w
            << ", alpha = " << alpha << ", beta = " << beta << '\n';

  for (auto x = 0u; x < u; ++x) {
    for (auto y = 0u; y < v; ++y) {
      auto value = beta * c[x + y * u];

      for (auto z = 0u; z < w; ++z)
        value += alpha * a[x + z * u] * b[z + y * w];

      c[x + y * u] = value;
    }
  }
}

// /**
//  * fonction de reduction a utilisé avec gemm_2d_cpu
//  * pour le fonctionnement des STARPU_REDUX handles
//  */
template <typename DataType>
void gemm_reduction_cpu(void *buffers[], void *arguments) {
  auto a = (DataType *)STARPU_MATRIX_GET_PTR(buffers[0]);
  auto b = (DataType *)STARPU_MATRIX_GET_PTR(buffers[1]);

  auto a_u = STARPU_MATRIX_GET_NY(buffers[0]);
  auto a_v = STARPU_MATRIX_GET_NX(buffers[0]);
  auto b_u = STARPU_MATRIX_GET_NY(buffers[1]);
  auto b_v = STARPU_MATRIX_GET_NX(buffers[1]);

  if (a_u != b_u || a_v != b_v)
    throw(std::runtime_error(
        "REDUCTION IMPOSSIBLE: les buffers ne font pas la même tailles"));

  std::cout << "INFO\tgemm_reduction_cpu\n";

  for (auto x = 0u; x < a_u; ++x)
    for (auto y = 0u; y < a_v; ++y)
      a[x + y * a_u] += b[x + y * a_u];
}

// TODO: use fill_value_cpu instead?
template <typename DataType>
void gemm_initialization_cpu(void *buffers[], void *arguments) {
  auto a = (DataType *)STARPU_MATRIX_GET_PTR(*buffers);

  auto u = STARPU_MATRIX_GET_NY(*buffers);
  auto v = STARPU_MATRIX_GET_NX(*buffers);

  std::cout << "INFO\tgemm_initialization_cpu\n";

  for (auto x = 0u; x < u; ++x)
    for (auto y = 0u; y < v; ++y)
      a[x + y * u] = static_cast<DataType>(0);
}