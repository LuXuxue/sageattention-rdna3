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


def get_target_arch():
    """Get target gfx11 architecture."""
    arch_env = os.getenv("GPU_ARCHS") or os.getenv("PYTORCH_ROCM_ARCH")
    if arch_env:
        for arch in arch_env.replace(";", " ").replace(",", " ").split():
            arch = arch.strip().split(":", 1)[0]
            if arch.startswith("gfx11"):
                return arch
    return "gfx1103"


if not SKIP_BUILD:
    import torch
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

    LIMITED_API_FLAGS = ["-DPy_LIMITED_API=0x03090000", "-DTORCH_STABLE_ONLY"]
    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

    if torch.version.hip is not None:
        rocm_home = configure_rocm(ROCM_HOME)
        cpp_extension.ROCM_HOME = rocm_home
        target_arch = get_target_arch()
        print(f"Target AMD GPU architecture: {target_arch}")

        if not target_arch.startswith("gfx11"):
            warnings.warn(
                f"Target architecture {target_arch} is not gfx11xx. "
                "This extension is designed for RDNA3 (gfx11xx)."
            )

        if os.name == "nt":
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
            f"--offload-arch={target_arch}",
            f"--rocm-path={rocm_home}",
        ] + LIMITED_API_FLAGS

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
