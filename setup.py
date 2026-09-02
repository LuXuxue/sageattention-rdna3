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
    import glob
    dirs = []
    msvc_patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\lib\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\lib\x64",
    ]
    for p in msvc_patterns:
        matches = sorted(glob.glob(p), reverse=True)
        if matches:
            dirs.append(matches[0])
            break
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

    Supports gfx110x (RDNA3, WMMA) and gfx103x (RDNA2, V_DOT4/V_DOT2) so a single
    built tree can produce either or both extension modules. Select via
    GPU_ARCHS / PYTORCH_ROCM_ARCH (e.g. "gfx1035;gfx1103"), else defaults to
    both installed device wheels.
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


def split_archs(archs):
    """Split arch list into RDNA3 (gfx11xx) and RDNA2 (gfx103x) groups.

    Returns (archs_11, archs_103). RDNA1 (gfx10xx except gfx103x) is not
    supported and triggers a warning.
    """
    a11, a103, other = [], [], []
    for a in archs:
        if a.startswith("gfx11"):
            a11.append(a)
        elif a.startswith("gfx103"):
            a103.append(a)
        else:
            other.append(a)
    if other:
        warnings.warn(
            f"Unsupported AMD arch(s) {other} ignored. "
            "Supported: gfx110x (RDNA3), gfx103x (RDNA2)."
        )
    return a11, a103


def base_hip_flags(rocm_home, abi, limited_api_flags):
    return [
        "-O3",
        "-std=c++17",
        "-ffast-math",
        "-fgpu-flush-denormals-to-zero",
        "-fno-offload-uniform-block",
        "-D__HIP_PLATFORM_AMD__=1",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-mllvm", "--lsr-drop-solution=1",
        "-mllvm", "-enable-post-misched=1",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        "-mllvm", "-amdgpu-max-memory-clause=32",
        "-mllvm", "-amdgpu-vgpr-index-mode=1",
        f"--rocm-path={rocm_home}",
        # V 全局转置 PV 方案 (out = P @ V, V_T [B,H,D,N] 行读 B operand):
        # 消除 v_frag 的 128 次 u16 LDS 列读, 大幅提升 PV 效率
        "-DSAGEATTN_VT_GLOBAL=1",
    ] + limited_api_flags


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

        archs_11, archs_103 = split_archs(target_archs)
        if not archs_11 and not archs_103:
            raise RuntimeError(
                f"No supported AMD archs in {target_archs}. "
                "Need gfx110x (RDNA3) or gfx103x (RDNA2)."
            )

        CXX_FLAGS_BASE = [
            "-O3",
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
        ] + LIMITED_API_FLAGS

        rocm_device_lib_path = os.path.join(
            rocm_home, "lib", "llvm", "amdgcn", "bitcode"
        )

        include_dirs = unique_paths([os.path.join(rocm_home, "include")])

        extra_link_args = []
        if os.name == "nt":
            link_lib_dirs = _get_msvc_lib_dirs()
            for d in link_lib_dirs:
                extra_link_args.append(f"/LIBPATH:{d}")

        if archs_11:
            print(f"  Building RDNA3 (gfx110x) extension for: {archs_11}")
            hip_flags_11 = base_hip_flags(rocm_home, ABI, LIMITED_API_FLAGS)
            for a in archs_11:
                hip_flags_11.append(f"--offload-arch={a}")
            if os.path.isdir(rocm_device_lib_path):
                hip_flags_11.append(f"--rocm-device-lib-path={rocm_device_lib_path}")
            append_env_flags(CXX_FLAGS_BASE, "CXX_APPEND_FLAGS")
            append_env_flags(hip_flags_11, "NVCC_APPEND_FLAGS")
            append_env_flags(hip_flags_11, "HIPCC_APPEND_FLAGS")
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_gfx110x",
                    sources=[
                        "csrc/pybind_gfx110x.cpp",
                        "csrc/attn_gfx110x.cu",
                    ],
                    include_dirs=include_dirs,
                    extra_compile_args={"cxx": CXX_FLAGS_BASE, "nvcc": hip_flags_11},
                    extra_link_args=extra_link_args,
                    py_limited_api=True,
                )
            )

        if archs_103:
            print(f"  Building RDNA2 (gfx103x) extension for: {archs_103}")
            hip_flags_103 = base_hip_flags(rocm_home, ABI, LIMITED_API_FLAGS)
            for a in archs_103:
                hip_flags_103.append(f"--offload-arch={a}")
            if os.path.isdir(rocm_device_lib_path):
                hip_flags_103.append(f"--rocm-device-lib-path={rocm_device_lib_path}")
            append_env_flags(CXX_FLAGS_BASE, "CXX_APPEND_FLAGS_103")
            append_env_flags(hip_flags_103, "NVCC_APPEND_FLAGS_103")
            append_env_flags(hip_flags_103, "HIPCC_APPEND_FLAGS_103")
            ext_modules.append(
                CUDAExtension(
                    name="sageattention._qattn_gfx103x",
                    sources=[
                        "csrc/pybind_gfx103x.cpp",
                        "csrc/attn_gfx103x.cu",
                    ],
                    include_dirs=include_dirs,
                    extra_compile_args={"cxx": CXX_FLAGS_BASE, "nvcc": hip_flags_103},
                    extra_link_args=extra_link_args,
                    py_limited_api=True,
                )
            )
    else:
        warnings.warn(
            "ROCm/HIP not detected (torch.version.hip is None). "
            "Skipping the native attention extension. "
            "This package requires a ROCm-enabled PyTorch build with an AMD GPU."
        )

    cmdclass["build_ext"] = BuildExtension

setup(
    name="sageattention",
    version="0.1.0",
    description="SageAttention HIP native implementation for AMD RDNA GPUs (gfx110x RDNA3, gfx103x RDNA2)",
    author="SageAttention RDNA Contributors",
    license="Apache-2.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=["torch>=2.4.0"],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
