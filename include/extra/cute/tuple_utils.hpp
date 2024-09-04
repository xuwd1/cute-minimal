#pragma once
#include <cute/config.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>

namespace cute {

/*
  CUTE recently introduced a very breaking change in their tuple implementation.
  The changed implementation would not allow for the following code to compile:
  
  ```cpp

  auto dyn = cute::make_shape(0,0,0);
  auto sta = cute::make_shape(cute::_1{}, cute::_2{}, cute::_3{});
  dyb = sta;

  ```
  What happened is basically they removed a once-existed copy constructor

  Thus we make the below helper function to explicitly convert
  the given tuple (source) elements to the desired (ref) tuple type.
  
  Note that the source tuple and the ref tuple should be congruent.
*/

template <class TpSource, class TpRef>
CUTE_HOST_DEVICE constexpr
auto
convert_tuple_elemtypes(TpSource const& source, TpRef const& ref)
{
  return cute::transform(source, ref, [] (auto const& s, auto const& r) { return static_cast<decltype(r)>(s); });
}

  
}