// Copyright (c) 2026, BAAI. All rights reserved.

#include "cambricon_copy.h"

#if defined(USE_CAMBRICON)

#include "cambricon_common.h"

namespace at::native::flagos::cambricon {

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (!dst.is_contiguous()) {
    return false;
  }
  if (dst.sizes() != src.sizes() || dst.scalar_type() != src.scalar_type()) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;
  }
  // The dtype set the generated kernels serve, spelled here rather than shared:
  // those helpers are internal to cambricon_kernels.cc.
  const auto dt = src.scalar_type();
  if (dt != at::kFloat && dt != at::kHalf && dt != at::kBFloat16) {
    return false;
  }

  CambriconDeviceGuard guard(src);
  const auto handle = HandleForDevice(static_cast<int>(src.device().index()));
  TensorDesc in_desc, out_desc;
  // The source keeps its strides; that is the whole point of the call.
  // cnnlSetTensorDescriptorEx requires the innermost stride to be 1, which a
  // view of a dense tensor always has.
  in_desc.SetStrided(src);
  out_desc.Set(dst);
  CheckCnnl(
      cnnlCopy_v2(
          handle, in_desc.get(), src.const_data_ptr(), out_desc.get(),
          dst.mutable_data_ptr(), nullptr, 0),
      "cnnlCopy_v2 (strided copy)");
  return true;
}

namespace {

// cnnlCastDataType names each conversion as a (source, destination) pair rather
// than taking two dtypes, so the pair is spelled out. Only the three float types
// the generated kernels serve are mapped; anything else declines and leaves the
// caller its CPU round-trip.
bool CastTypeFor(at::ScalarType src, at::ScalarType dst, cnnlCastDataType_t* out) {
  if (src == at::kFloat && dst == at::kHalf) {
    *out = CNNL_CAST_FLOAT_TO_HALF;
  } else if (src == at::kHalf && dst == at::kFloat) {
    *out = CNNL_CAST_HALF_TO_FLOAT;
  } else if (src == at::kFloat && dst == at::kBFloat16) {
    *out = CNNL_CAST_FLOAT_TO_BFLOAT16;
  } else if (src == at::kBFloat16 && dst == at::kFloat) {
    *out = CNNL_CAST_BFLOAT16_TO_FLOAT;
  } else if (src == at::kHalf && dst == at::kBFloat16) {
    *out = CNNL_CAST_HALF_TO_BFLOAT16;
  } else if (src == at::kBFloat16 && dst == at::kHalf) {
    *out = CNNL_CAST_BFLOAT16_TO_HALF;
  } else {
    return false;
  }
  return true;
}

} // namespace

at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype) {
  if (!src.defined() || !src.is_privateuseone()) {
    return at::Tensor();
  }
  const auto from = src.scalar_type();
  if (from == dtype) {
    return at::Tensor();
  }
  cnnlCastDataType_t cast_type;
  if (!CastTypeFor(from, dtype, &cast_type)) {
    return at::Tensor();
  }

  // The device is bound before the destination exists: at::empty resolves the
  // device from the current one, so allocating first would put the result on
  // whichever card happened to be selected.
  CambriconDeviceGuard guard(src);
  const auto handle = HandleForDevice(static_cast<int>(src.device().index()));
  const auto src_contig = src.is_contiguous() ? src : src.contiguous();
  auto out = at::empty(
      src_contig.sizes(), src_contig.options().dtype(dtype));
  if (src_contig.numel() == 0) {
    return out;
  }

  TensorDesc in_desc, out_desc;
  in_desc.Set(src_contig);
  out_desc.Set(out);
  CheckCnnl(
      cnnlCastDataType(
          handle, in_desc.get(), src_contig.const_data_ptr(), cast_type,
          out_desc.get(), out.mutable_data_ptr()),
      "cnnlCastDataType (DtypeCast)");
  return out;
}

} // namespace at::native::flagos::cambricon

#endif
