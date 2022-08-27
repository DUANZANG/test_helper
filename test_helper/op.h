#include "chime/core/platform/logging.hpp"

#include "tensor.h"

#include <atomic>
#include <cstdint>
#include <random>
#include <vector>

namespace test_helper {
class Operator {
public:
  virtual ~Operator() {}
  virtual void Compute() = 0;
  virtual void BackwardCompute() = 0;
  virtual void InitGrad() = 0;
  virtual std::vector<Tensor *> &OutputsWithAsync() { return _outputs; }
  virtual std::vector<Tensor *> &Outputs() {
    while (!_computed)
      ;
    return _outputs;
  }
  virtual bool IsComputed() { return _computed; }
  virtual bool IsBackwardComputed() { return _backward_computed; }

protected:
  std::vector<Tensor *> _inputs;
  std::vector<Tensor *> _outputs;
  std::atomic_bool _computed = {false};
  std::atomic_bool _backward_computed = {false};
};

class MulOp : public Operator {
public:
  MulOp(Tensor *input1, Tensor *input2, Tensor *output) {
    _inputs.push_back(input1);
    _inputs.push_back(input2);
    _outputs.push_back(output);
  }

  void Compute() override {
    CHECK(_inputs[0]->cols == _inputs[1]->rows) << "size not match";
    CHECK(_outputs[0]->rows == _inputs[0]->rows) << "size not match";
    CHECK(_outputs[0]->cols == _inputs[1]->cols) << "size not match";
    if (!_computed) {
      float *data1_ptr = _inputs[0]->data.get();
      float *data2_ptr = _inputs[1]->data.get();
      float *output_ptr = _outputs[0]->data.get();

      for (int64_t i = 0; i < _inputs[0]->rows; i++) {
        for (int64_t j = 0; j < _inputs[1]->cols; j++) {
          float sum = 0;
          for (int64_t k = 0; k < _inputs[0]->cols; k++) {
            sum += data1_ptr[i * _inputs[0]->cols + k] *
                   data2_ptr[k * _inputs[1]->cols + j];
          }
          output_ptr[i * _inputs[1]->cols + j] = sum;
        }
      }
    }
    _computed = true;
  }
  void InitGrad() override {
    for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
      _outputs[0]->grad.get()[i] = 1;
    }
  }

  void BackwardCompute() override {
    if (!_backward_computed) {
      float *data1_ptr = _inputs[0]->data.get();
      float *data2_ptr = _inputs[1]->data.get();
      float *grad1_ptr = _inputs[0]->grad.get();
      float *grad2_ptr = _inputs[1]->grad.get();
      float *grad_output_ptr = _outputs[0]->grad.get();

      for (int64_t i = 0; i < _inputs[1]->rows; i++) {
        for (int64_t j = 0; j < _inputs[1]->cols; j++) {
          float sum = 0;
          for (int64_t k = 0; k < _inputs[0]->rows; k++) {
            sum += data1_ptr[k * _inputs[0]->cols + i] *
                   grad_output_ptr[k * _inputs[1]->cols + j];
          }
          grad2_ptr[i * _inputs[1]->cols + j] = sum;
        }
      }

      for (int64_t i = 0; i < _inputs[0]->rows; i++) {
        for (int64_t j = 0; j < _inputs[0]->cols; j++) {
          float sum = 0;
          for (int64_t k = 0; k < _inputs[1]->cols; k++) {
            sum += data2_ptr[j * _inputs[1]->cols + k] *
                   grad_output_ptr[i * _outputs[0]->cols + k];
          }
          grad1_ptr[i * _inputs[0]->cols + j] = sum;
        }
      }



    }
    _backward_computed = true;
  }
};

class AddOp : public Operator {
public:
  AddOp(Tensor *input1, Tensor *input2, Tensor *output) {
    _inputs.push_back(input1);
    _inputs.push_back(input2);
    _outputs.push_back(output);
  }

  void Compute() override {
    CHECK(_inputs[0]->rows == _inputs[1]->rows) << "rows not match";
    CHECK(_inputs[0]->cols == _inputs[1]->cols) << "cols not match";
    if (!_computed) {
      float *data1_ptr = _inputs[0]->data.get();
      float *data2_ptr = _inputs[1]->data.get();
      float *output_ptr = _outputs[0]->data.get();

      for (int64_t i = 0; i < _inputs[0]->rows * _inputs[0]->cols; ++i) {
        output_ptr[i] = data1_ptr[i] + data2_ptr[i];
      }
      _computed = true;
    }
    // LOG(INFO) << "Called AddOp::Compute() once";
  }
  void BackwardCompute() override {
    if (!_backward_computed) {
      for (int64_t i = 0; i < _inputs[0]->rows * _inputs[0]->cols; ++i) {

        _inputs[0]->grad.get()[i] = _outputs[0]->grad.get()[i];
        _inputs[1]->grad.get()[i] = _outputs[0]->grad.get()[i];
      }
      _backward_computed = true;
    }
  }

  void InitGrad() override {
    for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
      _outputs[0]->grad.get()[i] = 1;
    }
  }
};

class RandomInitOp : public Operator {
public:
  RandomInitOp(Tensor *tensor) {
    _inputs.push_back(tensor);
    _outputs.push_back(tensor);
  }

  void Compute() override {
    float *data = _outputs[0]->data.get();

    if (!_computed) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(0.0, 1.0);
      for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
        data[i] = dis(gen);
      }
      _computed = true;
    }
    // LOG(INFO) << "Called RandomInitOp::Compute() once";
  }
  void InitGrad() override {
    for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
      _outputs[0]->grad.get()[i] = 1;
    }
  }

  void BackwardCompute() override { _backward_computed = true; }
};

class ReluOp : public Operator {
public:
  ReluOp(Tensor *input, Tensor *output) {
    _inputs.push_back(input);
    _outputs.push_back(output);
  }

  void Compute() override {
    if (!_computed) {
      for (int64_t i = 0; i < _inputs[0]->rows * _inputs[0]->cols; ++i) {
        if (_inputs[0]->data.get()[i] >= 0)
          _outputs[0]->data.get()[i] = _inputs[0]->data.get()[i];
        else
          _outputs[0]->data.get()[i] = 0;
      }
      _computed = true;
    }
  }
  void InitGrad() override {
    for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
      _outputs[0]->grad.get()[i] = 1;
    }
  }
  void BackwardCompute() override { _backward_computed = true; }
};

class SoftMaxOp : public Operator {
public:
  SoftMaxOp(Tensor *input, Tensor *output) {
    _inputs.push_back(input);
    _outputs.push_back(output);
  }
  void Compute() override {
    if (!_computed) {
      for (int64_t i = 0; i < _inputs[0]->cols; i++) {
        float sum = 0;
        for (int64_t j = 0; j < _inputs[0]->rows; j++) {
          sum += exp(_inputs[0]->data.get()[i * _inputs[0]->rows + j]);
        }
        for (int64_t j = 0; j < _inputs[0]->rows; j++) {
          _outputs[0]->data.get()[i * _inputs[0]->rows + j] =
              exp(_inputs[0]->data.get()[i * _inputs[0]->rows + j]) / sum;
        }
      }
    }
    _computed = true;
  }
  void InitGrad() override {
    for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
      _outputs[0]->grad.get()[i] = 1;
    }
  }
  void BackwardCompute() override { _backward_computed = true; }
};

class SumOp : public Operator {
public:
  SumOp(Tensor *input, Tensor *output) {
    _inputs.push_back(input);
    _outputs.push_back(output);
  }

  void Compute() override {
    if (!_computed) {
      float sum = 0;
      for (int64_t i = 0; i < _inputs[0]->rows * _inputs[0]->cols; ++i) {
        sum +=
            _inputs[0]->data.get()[i];
      }
      _outputs[0]->data.get()[0] = sum;  
        }
    _computed = true;
  }

  void BackwardCompute() override {
    if (!_backward_computed) {
      for (int64_t i = 0; i < _inputs[0]->rows * _inputs[0]->cols; ++i) {
        _inputs[0]->grad.get()[i] = 1.0;
      }
    }
    _backward_computed = true;
  }

  void InitGrad() override {
    for (int64_t i = 0; i < _outputs[0]->rows * _outputs[0]->cols; ++i) {
      _outputs[0]->grad.get()[i] = 1;
    }
  }
};

} // namespace test_helper