#include <cstdint>
#include <memory>
#include "chime/core/platform/logging.hpp"
#include <random>
#include <system_error>

namespace test_helper {

class Tensor {
public:
  enum InitialType {
    ALL_ZERO,
    RANDOM_ZERO_TO_ONE,
  };

  Tensor(int64_t num_col, int64_t num_row) : cols(num_col), rows(num_row) {
    data.reset(new float[num_col * num_row]);
  }

  Tensor() : cols(0), rows(0) {
    data.reset(nullptr);
    grad.reset(nullptr);
  }

  Tensor(int64_t num_rows, int64_t num_cols, InitialType type)
      : rows(num_rows), cols(num_cols) {
    data.reset(new float[num_cols * num_rows]);
    grad.reset(new float[num_cols * num_rows]);
    for (int64_t i = 0; i < rows * cols; i++) {
       grad.get()[i] = 0;
    }
    if (type == 0) {
      for (int64_t i = 0; i < rows * cols; i++) {
        data.get()[i] = 0;
      }
    } else {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0, 1);

      for (int64_t i = 0; i < rows * cols; i++) {
        data.get()[i] = dis(gen);
      }
    }
  }

  void Set(int64_t row, int64_t col, float value) {
    DCHECK_LT(row, rows);
    DCHECK_LT(col, cols);
    data.get()[row * cols + col] = value;
  }

  int64_t cols;
  int64_t rows;
  std::unique_ptr<float> data;
  std::unique_ptr<float> grad;
};




} // namespace test_helper