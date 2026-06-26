import codecs
import os
import platform
import re
import shutil
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

use_clang = False
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r", encoding="utf-8") as fp:
        return fp.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

        # Windows: copy produced DLLs next to the python package if needed
        if platform.system() == "Windows":
            mesh_renderer_dir = os.path.join(here, "igibson", "render", "mesh_renderer")
            release_dir = os.path.join(mesh_renderer_dir, "Release")

            if os.path.isdir(release_dir):
                for f in os.listdir(release_dir):
                    shutil.copy(os.path.join(release_dir, f), mesh_renderer_dir)
                shutil.rmtree(release_dir)

            vr_dll = os.path.join(here, "igibson", "render", "openvr", "bin", "win64", "openvr_api.dll")
            sr_ani_dir = os.path.join(here, "igibson", "render", "sranipal", "bin")

            if os.path.isfile(vr_dll):
                shutil.copy(vr_dll, mesh_renderer_dir)

            if os.path.isdir(sr_ani_dir):
                for f in os.listdir(sr_ani_dir):
                    if f.lower().endswith(".dll"):
                        shutil.copy(os.path.join(sr_ani_dir, f), mesh_renderer_dir)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY="
            + os.path.join(extdir, "igibson", "render", "mesh_renderer"),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY="
            + os.path.join(extdir, "igibson", "render", "mesh_renderer", "build"),
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        if use_clang:
            cmake_args += ["-DCMAKE_C_COMPILER=/usr/bin/clang", "-DCMAKE_CXX_COMPILER=/usr/bin/clang++"]

        cmake_args += ["-DMAC_PLATFORM=" + ("TRUE" if platform.system() == "Darwin" else "FALSE")]

        if os.getenv("USE_VR"):
            cmake_args += ["-DUSE_VR=TRUE"]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


long_description = read("README.md") if os.path.isfile(os.path.join(here, "README.md")) else ""

setup(
    # If you keep PEP621 metadata in pyproject.toml (recommended), you can omit name/version here.
    author="Soualhi Takieddine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TakieddineSOUALHI/MARLPointNAV/iGibson",
    zip_safe=False,

    # This will include igibson AND igibson.onpolicy automatically
    packages=find_packages(),

    ext_modules=[CMakeExtension("MeshRendererContext", sourcedir="igibson/render")],
    cmdclass={"build_ext": CMakeBuild},

    include_package_data=True,
    package_data={
        "igibson": [
            "global_config.yaml",
            "render/mesh_renderer/shaders/*",
        ]
    },
)
