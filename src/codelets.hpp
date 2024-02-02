#pragma once

#include <starpu.h>

#include "kernels.hpp"

template <typename DataType> starpu_codelet fill_value_codelet() {
  return {.where = STARPU_CPU,
          .cpu_func = {fill_value_cpu<DataType>},
          .nbuffers = 1,
          .modes = {STARPU_W}};
}

template <typename DataType> starpu_codelet gemm_1d_codelet() {
  return {.where = STARPU_CPU,
          .cpu_funcs = {gemm_cpu<DataType>},
          .nbuffers = 3,
          .modes = {STARPU_R, STARPU_R, STARPU_RW}};
}

template <typename DataType> starpu_codelet gemm_2d_codelet() {
  return {.where = STARPU_CPU,
          .cpu_funcs = {gemm_cpu<DataType>},
          .nbuffers = 3,
          .modes = {STARPU_R, STARPU_R, STARPU_REDUX}};
}

template <typename DataType> starpu_codelet gemm_2d_reduction_codelet() {
  return {.where = STARPU_CPU,
          .cpu_funcs = {gemm_2d_reduction_cpu<DataType>},
          .nbuffers = 2,
          .modes = {
              static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
              STARPU_R}};
}

template <typename DataType> starpu_codelet gemm_2d_initialization_codelet() {
  return {.where = STARPU_CPU,
          .cpu_funcs = {gemm_2d_initialization_cpu<DataType>},
          .nbuffers = 1,
          .modes = {STARPU_W}};
}