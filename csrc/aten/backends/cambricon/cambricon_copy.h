// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/ATen.h>

#if defined(USE_CAMBRICON)

namespace at::native::flagos::cambricon {

// Copies `src` into the contiguous `dst` on the device, honouring `src`'s
// strides.
//
// contiguous_ops.cc sends a strided PrivateUse1 source through a host
// round-trip when this returns false, and that round-trip is the single most
// expensive thing a training step does: `x.contiguous()` on the
// (2,64,16,512) stride (524288,1,32768,64) view RoPE hands it measured 16228 us
// against 18 us for the same gather through cnnlCopy_v2. The multiply inside
// every binary kernel gathers its operands this way, so the cost was paid
// several times a layer.
//
// Returns false for anything this path declines -- a non-contiguous or
// mismatched destination, a source out of the dtype range -- in which case the
// caller keeps its existing fallback.
bool StridedCopy(const at::Tensor& dst, const at::Tensor& src);

// Casts `src` to `dtype` on the device through cnnlCastDataType, returning a new
// contiguous tensor.
//
// `_to_copy` sends a dtype change on a PrivateUse1 tensor through a CPU
// round-trip when this returns an undefined tensor, and that round-trip is
// measured at 1714 us for float32 -> float16 on a (2048, 1024) tensor against
// 15.89 us for the vendor's own cast. The Qwen3 step issues 38 `.to()` calls, so
// the round-trip is worth milliseconds a step. Ascend, GCU, Kunlunxin and
// Iluvatar all cast on device here for the same reason; Cambricon was the
// remaining platform without one.
//
// Returns an undefined tensor for anything this declines -- a non-PrivateUse1
// source, an unchanged dtype, or a pair cnnlCastDataType does not define -- in
// which case the caller keeps its CPU round-trip.
at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype);

} // namespace at::native::flagos::cambricon

#else

namespace at::native::flagos::cambricon {

// Same shape every other vendor header takes in a build without its kernels:
// an inline no-op so contiguous_ops.cc can name this unconditionally, and the
// caller keeps its host round-trip because the answer is "not copied".
inline bool StridedCopy(const at::Tensor&, const at::Tensor&) {
  return false;
}

inline at::Tensor DtypeCast(const at::Tensor&, at::ScalarType) {
  return at::Tensor();
}

} // namespace at::native::flagos::cambricon

#endif
