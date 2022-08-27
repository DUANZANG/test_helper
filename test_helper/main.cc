#include "graph.h"

#include "chime/core/platform/threadpool.h"

#include <iostream>

namespace ah = test_helper;

static const int TEST_COUNT = 300;

bool CheckAdd(const ah::Tensor &matrix1, const ah::Tensor &matrix2,
              const ah::Tensor &matrix3) {
  const float error = 1e-5;
  for (int64_t i = 0; i < matrix1.rows * matrix1.cols; i++) {
    if (std::fabs(matrix1.data.get()[i] + matrix2.data.get()[i] -
                  matrix3.data.get()[i]) > error) {
      std::cout << "i = " << i << ", " << matrix1.data.get()[i] << " + "
                << matrix2.data.get()[i] << " - " << matrix3.data.get()[i]
                << " = "
                << matrix1.data.get()[i] + matrix2.data.get()[i] -
                       matrix3.data.get()[i]
                << std::endl;
      return false;
    }
  }
  return true;
}

class Tensor;
class Op;

// 1. API to contruct graph
// 2. Schedule alogrithm to run graph

int main() {

    auto pool = new chime::platform::ThreadPool(chime::platform::Env::Default(),
                                                "async_helper", 12);

    ah::Tensor m1(2,3);
    ah::Tensor m2(3,2); 
    ah::Tensor m3(3,3);
    ah::Tensor m4(1,1);       
    // Op: (launch -> scheduling(fathers done) -> finished)
    ah::RandomInitOp random_op1(&m1); // launch -> scheduling
    ah::RandomInitOp random_op2(&m2); // launch -> scheduling


    ah::MulOp mul_op(random_op1.OutputsWithAsync()[0],
                     random_op2.OutputsWithAsync()[0],
                     &m3); // launch -> scheduling

    ah::SumOp sum_op(&m3,
                    &m4);
    ah::StaticGraph graph(pool);


    graph.AddEdge(&random_op2, &mul_op);
    graph.AddEdge(&random_op1, &mul_op);

    graph.AddEdge(&mul_op, &sum_op);

    std::cout << "Graph forward... ";
 
    graph.ForwardSynced();

    std::cout << "Graph forward done\n";

    // ah::ShowTensor(t3);
    graph.Backward();


  // Op: (launch(must go on) -> scheduling(fathers done) -> finished)
  // chime::SetGraph(true); // static graph

  // Op op0            launch -> scheduling -> finished
  // print(op0.Synced());  // stop next op's launch
  // Op op1(op0)       launch -> waiting 
  // Op op2(op0)       launch -> waiting
  // Op op3(op2, op1)  launch -> waiting 
  // Op op4(op3)       launch
  // Op op5(op4)       launch
  // Op op6(op5)       launch

  // op1->op2->op3->op4->..->op11->op12->...->op100+

  std::cout << "Mul success!\n";
  ah::ShowTensor(m1);
  ah::ShowTensor(m2);
  ah::ShowTensor(m3);
  ah::ShowTensor(m4);
  ah::ShowGrad(m1);
  ah::ShowGrad(m2);
  ah::ShowGrad(m3);
  ah::ShowGrad(m4);


}