#!/usr/bin/env bash
# rocm-env.sh — make a ds4 ROCm build runnable regardless of how ROCm is installed.
#
# Source this before running ds4 / ds4-bench / ds4-server on AMD:
#
#     source scripts/rocm-env.sh
#     ./ds4 -m gguf/<MODEL>.gguf -p "..."
#
# Why this exists
# ---------------
# The classic /opt/rocm install ships the rocBLAS and hipBLASLt "Tensile"
# kernel libraries next to the .so files, so the runtime finds them on its
# own. The newer pip "ROCm SDK" wheels (rocm-sdk-*, layout
# .../site-packages/_rocm_sdk_devel) split those kernel libraries into a
# SEPARATE wheel (_rocm_sdk_libraries) that the runtime does NOT look in by
# default. The result is a hard abort on the first GEMM:
#
#     rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat
#     rocblaslt error: Cannot read .../hipblaslt/library/TensileLibrary_lazy_gfx1151.dat
#
# This script locates the kernel-library directories (whatever the layout)
# and exports ROCBLAS_TENSILE_LIBPATH / HIPBLASLT_TENSILE_LIBPATH so both
# the hipBLAS GemmEx path and the default hipBLASLt path load correctly.
# It is a no-op on a classic /opt/rocm install where the libraries are
# already found.

# Architecture to look up hipBLASLt kernels for (hipBLASLt uses a per-arch
# subdir in the wheel layout). Override with ROCM_ARCH=... if needed.
_ds4_arch="${ROCM_ARCH:-gfx1151}"

# Build a list of candidate ROCm roots to probe.
_ds4_roots=()
[ -n "$ROCM_PATH" ] && _ds4_roots+=("$ROCM_PATH")
[ -n "$HIP_PATH" ]  && _ds4_roots+=("$HIP_PATH")
# The wheel splits libraries into a sibling _rocm_sdk_libraries dir; derive it
# from any _rocm_sdk_* root we were given.
for _r in "${_ds4_roots[@]}"; do
    case "$_r" in
        */_rocm_sdk_*) _ds4_roots+=("${_r%/_rocm_sdk_*}/_rocm_sdk_libraries") ;;
    esac
done
# hipconfig / on-PATH hipcc as fallbacks.
_ds4_hipcfg="$(hipconfig --rocmpath 2>/dev/null)"
[ -n "$_ds4_hipcfg" ] && _ds4_roots+=("$_ds4_hipcfg")
_ds4_hipcc="$(command -v hipcc 2>/dev/null)"
[ -n "$_ds4_hipcc" ] && _ds4_roots+=("${_ds4_hipcc%/bin/hipcc}")
_ds4_roots+=(/opt/rocm)
# Probe any python ROCm SDK libraries wheel as a last resort.
for _sp in /opt/python/lib/python*/site-packages /usr/lib/python*/site-packages; do
    [ -d "$_sp/_rocm_sdk_libraries" ] && _ds4_roots+=("$_sp/_rocm_sdk_libraries")
done

# Find rocBLAS Tensile dir: <root>/lib/rocblas/library containing *gfx*.dat
if [ -z "$ROCBLAS_TENSILE_LIBPATH" ]; then
    for _r in "${_ds4_roots[@]}"; do
        _d="$_r/lib/rocblas/library"
        if ls "$_d"/*gfx*.dat >/dev/null 2>&1; then
            export ROCBLAS_TENSILE_LIBPATH="$_d"
            break
        fi
    done
fi

# Find hipBLASLt Tensile dir. Wheel layout nests per-arch:
# <root>/lib/hipblaslt/library/<arch>/TensileLibrary_lazy_<arch>.dat
# Classic layout is flat: <root>/lib/hipblaslt/library/TensileLibrary_lazy_<arch>.dat
if [ -z "$HIPBLASLT_TENSILE_LIBPATH" ]; then
    for _r in "${_ds4_roots[@]}"; do
        if [ -f "$_r/lib/hipblaslt/library/$_ds4_arch/TensileLibrary_lazy_${_ds4_arch}.dat" ]; then
            export HIPBLASLT_TENSILE_LIBPATH="$_r/lib/hipblaslt/library/$_ds4_arch"
            break
        elif [ -f "$_r/lib/hipblaslt/library/TensileLibrary_lazy_${_ds4_arch}.dat" ]; then
            export HIPBLASLT_TENSILE_LIBPATH="$_r/lib/hipblaslt/library"
            break
        fi
    done
fi

if [ -n "$ROCBLAS_TENSILE_LIBPATH" ]; then
    echo "rocm-env: ROCBLAS_TENSILE_LIBPATH=$ROCBLAS_TENSILE_LIBPATH"
else
    echo "rocm-env: warning: rocBLAS Tensile library not found (classic /opt/rocm install? then this is fine)" >&2
fi
if [ -n "$HIPBLASLT_TENSILE_LIBPATH" ]; then
    echo "rocm-env: HIPBLASLT_TENSILE_LIBPATH=$HIPBLASLT_TENSILE_LIBPATH"
else
    echo "rocm-env: note: hipBLASLt kernels for $_ds4_arch not found; set DS4_HIP_NO_BLASLT=1 to use the hipBLAS path" >&2
fi

unset _ds4_arch _ds4_roots _ds4_hipcfg _ds4_hipcc _ds4_root _r _d _sp
