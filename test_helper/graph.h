#ifndef ASYNC_HELPER_GRAPH_H_
#define ASYNC_HELPER_GRAPH_H_

#include "op.h"

#include "chime/core/platform/threadpool.h"

#include <algorithm>

namespace test_helper {

using chime::platform::ThreadPool;

class StaticGraph {
public:
  using Edge = std::pair<Operator *, Operator *>;
  using EdgeList = std::vector<Edge>;
  using OperatorList = std::vector<Operator *>;

  StaticGraph(ThreadPool *pool) { _pool = pool; }

  ~StaticGraph() {
    _pool->Wait();
  }






  void AddEdge(Operator *from, Operator *to) {
    _edges.push_back(std::make_pair(from, to));
    if (std::find(_operators.begin(), _operators.end(), from) ==
        _operators.end()) {
      _operators.push_back(from);
    }
    if (std::find(_operators.begin(), _operators.end(), to) ==
        _operators.end()) {
      _operators.push_back(to);
    }
  }

  void Forward() {
    for (auto &op : _operators) {
      _pool->Schedule([op, this]() {
        while (!IsReadyToBeComputed(op))
          ;
        op->Compute();
      });
    }
  }


  void Backward(){
        for (auto &op : _operators) {
      _pool->Schedule([op, this]() {
        while (!IsReadyToBeBackwardComputed(op))
          ;
        op->BackwardCompute();
      });
    }
       _pool->Wait();
  }

  void ForwardSynced() {
    Forward();
    _pool->Wait();
  }

  bool IsReadyToBeComputed(Operator *op) {
    OperatorList fathers(GetFathers(op));
    for (auto &father : fathers) {
      if (!father->IsComputed())
        return false;
    }
    return true;
  }

  bool IsReadyToBeBackwardComputed(Operator *op) {
    OperatorList sons(Getsons(op));
    for (auto &son : sons) {
      if (!son->IsBackwardComputed())
        return false;
      else if (son->IsBackwardComputed())
        return true;
    }
    op->InitGrad();
    return true;
  }

private:
  OperatorList GetFathers(Operator *op) {
    OperatorList fathers;
    for (auto &edge : _edges) {
      if (edge.second == op) {
        fathers.push_back(edge.first);
      }
    }
    return std::move(fathers);
  }

  OperatorList Getsons(Operator *op) {
    OperatorList sons;
    for (auto &edge : _edges) {
      if (edge.first == op) {
        sons.push_back(edge.second);
      }
    }
    return std::move(sons);
  }

  ThreadPool *_pool; // not owned
  EdgeList _edges;
  OperatorList _operators;
};

} // namespace async_helper

#endif // ASYNC_HELPER_GRAPH_H_