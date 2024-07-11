#include <cute/tensor.hpp>

using namespace cute;

int main(){
  auto l1 = make_layout(make_shape(_5{},_7{}));
  print(l1);

  auto owning_tensor = make_tensor<float>(l1);
  print(owning_tensor);


  float* data = new float[cosize_v<decltype(l1)>];
  auto nonowning_tensor = make_tensor(make_gmem_ptr(data), l1);
  print(nonowning_tensor);
  delete[] data;
  
}