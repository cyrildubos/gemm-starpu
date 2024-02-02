#pragma once

#include <starpu.h>

#include "codelets.hpp"

/**
 * The tile class is an abstraction to hold StarPU matrix interfaces and launch
 * tasks on them
 */
template <typename DataType> class Tile {
public:
  unsigned u;
  unsigned v;

  DataType *data;

  starpu_data_handle_t handle;

  Tile(unsigned u, unsigned v) : u{u}, v{v} {
    starpu_malloc((void **)&data, u * v * sizeof(DataType));

    // TODO: u, v, u?
    starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)data, //
                                v, v, u, sizeof(DataType));                //
  }

  void fill_random(); // TODO

  void assert_equals(const Tile &other) {
    if (u != other.v || v != other.v)
      throw(std::runtime_error("different tile size"));

    starpu_data_acquire(handle, STARPU_R);
    starpu_data_acquire(other.handle, STARPU_R);

    for (auto x = 0; x < u; ++x)
      for (auto y = 0; y < v; ++y)
        if (data[x + y * u] != other.data[x + y * u])
          throw(std::runtime_error("different tile values"));

    starpu_data_release(handle);
    starpu_data_release(other.handle);
  }

  static void gemm_1d(const DataType alpha, const Tile &a, const Tile &b,
                      const DataType beta, Tile &c) {
    auto codelet = gemm_1d_codelet<DataType>();

    starpu_task_insert(&codelet,             //
                       STARPU_R, a.handle,   //
                       STARPU_R, b.handle,   //
                       STARPU_RW, c.handle,  //
                       STARPU_VALUE, &alpha, //
                       sizeof(DataType),     //
                       STARPU_VALUE, &beta,  //
                       sizeof(DataType),     //
                       0);                   //
  }
};
