cmake_minimum_required(VERSION 3.22)

project(cuda_exercises LANGUAGES CUDA)

set(BUILD_TYPE Release)

set(EXERCISE_MEMCPY_SOURCE "src/memcpy.cu")
set(EXERCISE_MATRIX_ADDITION_SOURCE "src/matrix_addition.cu")
set(EXERCISE_ARRAY_SUM_PRE_SOURCE "src/array_sum_pre.cu")
set(EXERCISE_MATRIX_TRANSPOSE_SOURCE "src/matrix_transpose.cu")

add_executable(exercise_memcpy ${EXERCISE_MEMCPY_SOURCE})
target_link_libraries(exercise_memcpy cudart cuda)

add_executable(exercise_matrix_addition ${EXERCISE_MATRIX_ADDITION_SOURCE})
target_link_libraries(exercise_matrix_addition cudart cuda)

add_executable(exercise_array_sum_pre ${EXERCISE_ARRAY_SUM_PRE_SOURCE})
target_link_libraries(exercise_array_sum_pre cudart cuda)

add_executable(exercise_matrix_transpose ${EXERCISE_MATRIX_TRANSPOSE_SOURCE})
target_link_libraries(exercise_matrix_transpose cudart cuda)
