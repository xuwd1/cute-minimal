# CUTE-MINIMAL

- What is this

  A minimal CUTE library that could be compiled with ordinary compilers (e.g. clang, gcc), targeting CPUs, with it's major functionalities retained.

- How this is made

  By removing most of the CUDA specific headers and applying some very simple patches. It's worth noticing that this minimal CUTE library is completely detached from cutlass.

## Current Limitations:

### 1. Datatype support

- CUTE relies heavily on datatype structs (or templates) defined in the cutlass library to work. In this initial version, I detached CUTE from cutlass in a rather naive way, and thus:
  - Complex datatype support is dropped completely. (Very low priority to fix)
  - Sub-byte datatypes are replaced by placeholders. (Will be fixed very soon)
  - `uint128_t` datatypes are plainly aliased to `__int128`, which may be improper. (Will be fixed soon)
  - Software `tfloat32_t`, `half_t`, `bfloat16_t` datatypes are missing. (Mid priority to fix)

### 2. The `debug.hpp` and `config.hpp` headers

- The `config.hpp` header relies on proper compiler definitions (-D) to define some helper macros for the whole library. Currently this header is not thoroghly checked, but in general it should be fine.
- The `debug.hpp` header relies on CUDA runtime API to call functions like `cudaDeviceSynchronize` Thus, this header should also be patched in theory. But luckily these function calls only exists inside macro definitions so as long as we don't use the macros, we are generally safe.