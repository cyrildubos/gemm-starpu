#pragma once

#include <starpu.h>

#include "kernels.hpp"

#define FILL_VALUE_CODELET(TYPE)                                               \
  (starpu_codelet{.where = STARPU_CPU,                                         \
                  .cpu_func = {fill_value_cpu<TYPE>},                          \
                  .nbuffers = 1,                                               \
                  .modes = {starpu_data_access_mode(STARPU_W)}})

#define GEMM_1D_CODELET(TYPE)                                                  \
  (starpu_codelet{.where = STARPU_CPU,                                         \
                  .cpu_funcs = {gemm_cpu<TYPE>},                               \
                  .nbuffers = 3,                                               \
                  .modes = {starpu_data_access_mode(STARPU_R),                 \
                            starpu_data_access_mode(STARPU_R),                 \
                            starpu_data_access_mode(STARPU_RW)}})

#define GEMM_2D_CODELET(TYPE)                                                  \
  (starpu_codelet{.where = STARPU_CPU,                                         \
                  .cpu_funcs = {gemm_cpu<TYPE>},                               \
                  .nbuffers = 3,                                               \
                  .modes = {starpu_data_access_mode(STARPU_R),                 \
                            starpu_data_access_mode(STARPU_R),                 \
                            starpu_data_access_mode(STARPU_REDUX)}})

#define GEMM_REDUCTION_CODELET(TYPE)                                           \
  (starpu_codelet{                                                             \
      .where = STARPU_CPU,                                                     \
      .cpu_funcs = {gemm_reduction_cpu<TYPE>},                                 \
      .nbuffers = 2,                                                           \
      .modes = {starpu_data_access_mode(STARPU_RW | STARPU_COMMUTE),           \
                starpu_data_access_mode(STARPU_R)}})

#define GEMM_INITIALIZATION_CODELET(TYPE)                                      \
  (starpu_codelet{.where = STARPU_CPU,                                         \
                  .cpu_funcs = {gemm_initialization_cpu<TYPE>},                \
                  .nbuffers = 1,                                               \
                  .modes = {starpu_data_access_mode(STARPU_W)}})