/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

// #include <vector_types.h>
#include <cute/numeric/int.hpp>
#include <cute/numeric/real.hpp>

namespace cute {

template <typename T>
struct sizeof_bits {
  static constexpr int value = int(sizeof(T) * 8);
};

template <typename T>
struct sizeof_bits<T const>: sizeof_bits<T> {};

template <>
struct sizeof_bits<void> {
  static constexpr int value = 0;
};

template <class T>
static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the number of bytes required to hold a specified number of bits
CUTE_HOST_DEVICE
constexpr
int
bits_to_bytes(int bits) {
  return (bits + 7) / 8;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
struct is_subbyte {
  static constexpr bool value = sizeof_bits<T>::value < 8;
};

template <class T>
struct is_subbyte<T const> : is_subbyte<T> {};

template <class T>
static constexpr auto is_subbyte_v = is_subbyte<T>::value;

// struct uint1b_t;
// struct int2b_t;
// struct uint2b_t;
// struct int4b_t;
// struct uint4b_t;
// struct bin1_t;

} // end namespace cute
