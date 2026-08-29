import os
import sys
import subprocess
import warnings

from setuptools import setup, find_packages

SKIP_BUILD = (
    os.getenv("SAGEATTN_SKIP_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}


def append_env_flags(flags, env_name):
    """Append extra compiler flags from environment variable."""
    extra = os.getenv(env_name, "").strip()
    if extra:
        flags += extra.split()


def unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def rocm_sdk_path(which):
    try:
        return subprocess.check_output(
            ["rocm-sdk", "path", f"--{which}"], text=True
        ).strip()
    except Exception:
        return None


def configure_rocm(default_rocm_home):
    sdk_root = rocm_sdk_path("root")
    sdk_bin = rocm_sdk_path("bin")
    rocm_home = sdk_root or default_rocm_home or os.getenv("ROCM_HOME")
    if not rocm_home:
        raise RuntimeError(
            "Cannot find ROCm. Activate a ROCm-enabled PyTorch environment."
        )

    os.environ["ROCM_HOME"] = rocm_home
    if os.name == "nt":
        os.environ.setdefault("CC", "clang-cl")
        os.environ.setdefault("CXX", "clang-cl")
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    path_parts = [
        os.path.join(rocm_home, "lib", "llvm", "bin"),
        os.path.join(rocm_home, "bin"),
        sdk_bin,
    ]

    # On Windows, ensure MSVC linker (link.exe) and SDK tools (rc.exe) are findable
    if os.name == "nt":
        msvc_link_dir = _find_msvc_bin_dir()
        if msvc_link_dir:
            path_parts.append(msvc_link_dir)
        sdk_bin_dir = _find_windows_sdk_bin()
        if sdk_bin_dir:
            path_parts.append(sdk_bin_dir)

    os.environ["PATH"] = os.pathsep.join(
        unique_paths(path_parts) + [os.environ.get("PATH", "")]
    )
    return rocm_home


def _find_msvc_bin_dir():
    """Find the MSVC Hostx64/x64 bin directory containing link.exe."""
    import glob
    patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern), reverse=True)
        for m in matches:
            if os.path.isfile(os.path.join(m, "link.exe")):
                return m
    return None


def _find_windows_sdk_bin():
    """Find the Windows SDK x64 bin directory containing rc.exe."""
    import glob
    matches = sorted(
        glob.glob(r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64"),
        reverse=True,
    )
    for m in matches:
        if os.path.isfile(os.path.join(m, "rc.exe")):
            return m
    return None


def _get_msvc_lib_dirs():
    """Get MSVC and Windows SDK library directories for linking."""
    import glob
    dirs = []
    # MSVC lib
    msvc_patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\lib\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\lib\x64",
    ]
    for p in msvc_patterns:
        matches = sorted(glob.glob(p), reverse=True)
        if matches:
            dirs.append(matches[0])
            break
    # Windows SDK libs
    sdk_patterns = [
        r"C:\Program Files (x86)\Windows Kits\10\Lib\*\ucrt\x64",
        r"C:\Program Files (x86)\Windows Kits\10\Lib\*\um\x64",
    ]
    for sp in sdk_patterns:
        matches = sorted(glob.glob(sp), reverse=True)
        if matches:
            dirs.append(matches[0])
    return [d for d in dirs if os.path.isdir(d)]


def get_target_archs():
    """Get target AMD GPU architectures.

    Supports gfx1103 (RDNA3, WMMA) and gfx1035 (RDNA2, MFMA/V_DOT4) so a single
    built wheel can run on both. Select via env GPU_ARCHS / PYTORCH_ROCM_ARCH
    (e.g. "gfx1035;gfx1103"), else defaults to both installed device wheels.
    """
    env_archs = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if env_archs:
        out = []
        for arch in env_archs.replace(";", " ").replace(",", " ").split():
            arch = arch.strip().split(":", 1)[0]
            if arch:
                out.append(arch)
        if out:
            return out
    return ["gfx1103", "gfx1035"]


if not SKIP_BUILD:
    import torch
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

    LIMITED_API_FLAGS = ["-DPy_LIMITED_API=0x03090000", "-DTORCH_STABLE_ONLY"]
    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

    if torch.version.hip is not None:
        rocm_home = configure_rocm(ROCM_HOME)
        cpp_extension.ROCM_HOME = rocm_home
        target_archs = get_target_archs()
        print(f"Target AMD GPU architectures: {target_archs}")

        if not any(a.startswith("gfx11") for a in target_archs):
            warnings.warn(
                f"Target architectures {target_archs} contain no gfx11xx. "
                "The extension also supports gfx10xx (RDNA2, MFMA/V_DOT4)."
            )
            CXX_FLAGS = [
                "/O2",
                "/std:c++17",
                "/permissive-",
                f"/D_GLIBCXX_USE_CXX11_ABI={ABI}",
            ]
        else:
            CXX_FLAGS = [
                "-O3",
                "-std=c++17",
                f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
            ]
        CXX_FLAGS += LIMITED_API_FLAGS

        HIP_FLAGS = [
            "-O3",
            "-std=c++17",
            "-ffast-math",
            "-fgpu-flush-denormals-to-zero",
            "-fno-offload-uniform-block",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
            "-mllvm",
            "--lsr-drop-solution=1",
            "-mllvm",
            "-enable-post-misched=1",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
            "-mllvm",
            "-amdgpu-max-memory-clause=32",
            "-mllvm",
            "-amdgpu-vgpr-index-mode=1",
            f"--rocm-path={rocm_home}",
            # V 全局转置 PV 方案 (out = P @ V, V_T [B,H,D,N] 行读 B operand):
            # 消除 v_frag 的 128 次 u16 LDS 列读, 大幅提升 PV 效率
            # (配合 core.py 的 V 转置; 若需旧路径, 改为 -DSAGEATTN_VT_GLOBAL=0 并关 core 转置)
            "-DSAGEATTN_VT_GLOBAL=1",
        ] + LIMITED_API_FLAGS

        # Emit device code for each target arch. gfx1035 (RDNA2) uses V_DOT4_I32_I8
        # for the int8 QK and V_DOT2_F32_F16 for the fp16 PV — both are SIMD dot
        # instructions that need NO extra target feature (gfx1035 enables them by
        # default; verified by probe). Do NOT pass "+mai-insts" here: it is only
        # needed for MFMA (unavailable on RDNA2 consumer gfx1035 per gfx1035.md
        # §0.12) and it is a global clang flag that, when combined in the same
        # multi-arch hipcc invocation, also reaches gfx1103 and crashes the LLVM
        # AMDGPU backend during WMMA codegen (Branch relaxation pass segfault).
        for arch in target_archs:
            HIP_FLAGS.append(f"--offload-arch={arch}")

        rocm_device_lib_path = os.path.join(
            rocm_home, "lib", "llvm", "amdgcn", "bitcode"
        )
        if os.path.isdir(rocm_device_lib_path):
            HIP_FLAGS.append(f"--rocm-device-lib-path={rocm_device_lib_path}")

        append_env_flags(CXX_FLAGS, "CXX_APPEND_FLAGS")
        append_env_flags(HIP_FLAGS, "NVCC_APPEND_FLAGS")
        append_env_flags(HIP_FLAGS, "HIPCC_APPEND_FLAGS")

        include_dirs = unique_paths([os.path.join(rocm_home, "include")])

        # On Windows, find MSVC/SDK library paths for linking
        extra_link_args = []
        if os.name == "nt":
            link_lib_dirs = _get_msvc_lib_dirs()
            for d in link_lib_dirs:
                extra_link_args.append(f"/LIBPATH:{d}")

        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_gfx11",
                sources=[
                    "csrc/pybind_gfx11.cpp",
                    "csrc/attn_gfx11.cu",
                ],
                include_dirs=include_dirs,
                extra_compile_args={
                    "cxx": CXX_FLAGS,
                    "nvcc": HIP_FLAGS,
                },
                extra_link_args=extra_link_args,
                py_limited_api=True,
            )
        )
    else:
        warnings.warn(
            "ROCm/HIP not detected (torch.version.hip is None). "
            "Skipping the gfx11 native attention extension. "
            "This package requires a ROCm-enabled PyTorch build with an AMD GPU."
        )

    cmdclass["build_ext"] = BuildExtension

setup(
    name="sageattention",
    version="0.1.0",
    description="SageAttention HIP native implementation for RDNA3 (gfx11xx)",
    author="SageAttention RDNA3 Contributors",
    license="Apache-2.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=["torch>=2.4.0"],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
