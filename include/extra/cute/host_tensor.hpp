#pragma once

#include <cstdint>
#include <cstdlib>
#include <cute/config.hpp>

#include <cute/util/type_traits.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/integer_sequence.hpp>

#include <cute/container/tuple.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/container/array_subbyte.hpp>

#include <cute/pointer.hpp>
#include <cute/layout.hpp>

#include <cute/tensor.hpp>

namespace cute {


template <typename T, ssize_t Alignment = 64>
struct HostMallocEngine
{
  using iterator = T*;
  using reference    = typename cute::iterator_traits<iterator>::reference;
  using element_type = typename cute::iterator_traits<iterator>::element_type;
  using value_type   = typename cute::iterator_traits<iterator>::value_type;
  
  HostMallocEngine() {}

  // constructor
  HostMallocEngine(ssize_t size){
    storage_ = reinterpret_cast<iterator>(std::aligned_alloc(Alignment, size * sizeof(T)));
  }

  ~HostMallocEngine(){
    if(storage_){
      std::free(storage_);
    }
  }
  
  iterator storage_{nullptr};
  
  CUTE_HOST constexpr auto begin() const {return storage_;}
  CUTE_HOST constexpr auto begin()       {return storage_;}
};

template <typename T, class Layout, ssize_t Alignment = 64>
struct HostTensor : public cute::Tensor<HostMallocEngine<T, Alignment>, Layout>
{
  using Engine = HostMallocEngine<T, Alignment>;
  using iterator     = typename Engine::iterator;
  using value_type   = typename Engine::value_type;
  using element_type = typename Engine::element_type;
  using reference    = typename Engine::reference;

  using engine_type  = Engine;
  using layout_type  = Layout;

  CUTE_HOST
  HostTensor() {}

  CUTE_HOST
  HostTensor(Layout const& layout): cute::Tensor<HostMallocEngine<T, Alignment>, Layout>(layout){}

  // additonal move constructor
  CUTE_HOST
  HostTensor(HostTensor&& other) noexcept
    : cute::Tensor<HostMallocEngine<T, Alignment>, Layout>(std::move(other)){}

};

template <typename T, class Layout, ssize_t Alignment = 64>
auto make_host_tensor(Layout const& layout){
  return HostTensor<T, Layout, Alignment>(layout);
}


}