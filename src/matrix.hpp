#pragma once

#include <iostream>
#include <vector>

#include "tile.hpp"

// -----

static int copy_handle(starpu_data_handle_t dst, starpu_data_handle_t src,
                       unsigned nb) {
  unsigned block;

  for (block = 0; block < nb; block++)
    starpu_data_cpy(starpu_data_get_sub_data(dst, 1, block),
                    starpu_data_get_sub_data(src, 1, block), 1, NULL, NULL);
  return 0;
}

// -----

/**
 * The matrix contains the tiles and operates on them
 * (TER PART 2) In the distributed case the distribution of tiles among nodes
 * should be block-cyclic (https://netlib.org/scalapack/slug/node75.html)
 */
template <typename DataType> struct Matrix {
  unsigned m;
  unsigned n;

  unsigned u;
  unsigned v;

  std::vector<Tile<DataType>> tiles;

  Matrix(unsigned m, unsigned n, unsigned u, unsigned v)
      : m{m}, n{n}, u{u}, v{v} {
    for (auto i = 0; i < m; i += u)
      for (auto j = 0; j < n; j += v)
        tiles.push_back(Tile<DataType>(u, v));

    std::cout << "INFO\t" << m << 'x' << n << " matrix with " << u << 'x' << v
              << " tile(s): " << tiles.size() << " tile(s)\n";
  }

  ~Matrix() {
    for (auto &tile : tiles)
      starpu_data_unregister(tile.handle);
  }

  /**
   * This function can be used for debugging.
   * It should print the contents of a tiled matrix
   */
  void print() {
    for (auto &tile : tiles)
      starpu_data_acquire(tile.handle, STARPU_R);

    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j)
        std::cout << tiles[(i / u) + (j / v) * (m / u)].data[i % u, j % v]
                  << ' ';

      std::cout << '\n';
    }

    for (auto &tile : tiles)
      starpu_data_release(tile.handle);
  }

  void fill_value(const DataType value) {
    auto codelet = fill_value_codelet<DataType>();

    for (auto &tile : tiles)
      starpu_task_insert(&codelet,              //
                         STARPU_W, tile.handle, //
                         STARPU_VALUE, &value,  //
                         sizeof(DataType),      //
                         0);                    //

    starpu_task_wait_for_all(); // TODO
  }

  void fill_random();

  /**
   * This function can be used to check the correctness of operations.
   * It should compare two tiled matrices and throw an exception if they are
   * not equal
   */
  void assert_equals(const Matrix &other) {
    if (m != other.m || n != other.n)
      throw(std::runtime_error("different matrix size"));

    if (u != other.v || v != other.v)
      throw(std::runtime_error("different tile size"));

    auto iterator = other.tiles.cbegin();

    for (auto &tile : tiles)
      tile.assert_equals(*iterator++);
  }

  /**
   * The gemm function does a generalised matrix multiplication on tiled
   * matrices. It should compute C <- alpha * op(A) * op(B) + beta * C. alpha
   * and beta are scalars A, B, and C are tiled matrices Each op is 'T' if the
   * matrx is transposed, 'N' otherwise
   */
  static void gemm_1d(const DataType alpha, const Matrix &a, const Matrix &b,
                      const DataType beta, Matrix &c) {
    for (auto x = 0; x < (a.m / a.u); ++x)
      for (auto y = 0; y < (b.n / b.v); ++y)
        Tile<DataType>::gemm_1d(alpha, a.tiles[x], b.tiles[y], beta,
                                c.tiles[x + y * (a.m / a.u)]);
  }

  static void gemm_2d(const DataType alpha, const Matrix &a, const Matrix &b,
                      const DataType beta, Matrix &c) {
    auto codelet = gemm_2d_codelet<DataType>();

    auto reduction = gemm_2d_reduction_codelet<DataType>();
    auto initialization = gemm_2d_initialization_codelet<DataType>();

    for (auto x = 0; x < (a.m / a.u); x++)
      for (auto y = 0; y < (b.n / b.v); y++)

        for (auto z = 0; z < (a.n / a.v); z++) {
          std::cout << "INFO\tgemm_2d: x = " << x << ", y = " << y
                    << ", z = " << z << '\n';

          auto a_index = x + z * (a.m / a.u);
          auto b_index = z + y * (b.m / b.u);
          auto c_index = x + y * (c.m / c.u);

          starpu_data_set_reduction_methods(c.tiles[c_index].handle, &reduction,
                                            &initialization);

          auto zero = static_cast<DataType>(0);

          starpu_task_insert(&codelet,                              //
                             STARPU_R, a.tiles[a_index].handle,     //
                             STARPU_R, b.tiles[b_index].handle,     //
                             STARPU_REDUX, c.tiles[c_index].handle, //
                             STARPU_VALUE, &alpha,                  //
                             sizeof(DataType),                      //
                             STARPU_VALUE, z == 0 ? &beta : &zero,  //
                             sizeof(DataType),                      //
                             0);                                    //
        }

    starpu_task_wait_for_all();
  }
};
