import os
import platform
import subprocess
import sysconfig
import numpy as np

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"


cereal_dir = Dir('./cereal')
messaging_dir = Dir('./cereal/messaging')
visionipc_dir = Dir('./cereal/visionipc')
common = ''

cpppath = [
  '/usr/lib/include',
  sysconfig.get_paths()['include'],
  '#',
  '#cereal'
]

libpath = [
  "/usr/local/lib",
  "/usr/lib",
  "/usr/lib/aarch64-linux-gnu",
  "#cereal"
]

AddOption('--test',
          action='store_true',
          help='build test files')

ldflags = ["-pthread", "-Wl,--as-needed", "-Wl,--no-undefined"]

env = Environment(
  ENV=os.environ,
  CC='clang',
  CXX='clang++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow",
    "-Wno-unknown-warning-option",
    "-Wno-deprecated-register",
    "-Wno-register",
    "-Wno-inconsistent-missing-override",
    "-Wno-c99-designator",
    "-Wno-reorder-init-list",
    "-Wno-error=unused-but-set-variable",
  ],
  CFLAGS="-std=gnu11",
  CXXFLAGS="-std=c++1z",
  LINKFLAGS=ldflags,
  CPPPATH=cpppath,
  LIBPATH=libpath,
  CYTHONCFILESUFFIX=".cpp",
  tools=["default", "cython"]
)

Export('env', 'arch', 'common')

# Base cython environment
envCython = env.Clone(LIBS=[])
envCython["CPPPATH"] += [np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('envCython')

# build and export shared cereal libs
SConscript(['cereal/SConscript'])

cereal = [File('#cereal/libcereal.a')]
messaging = [File('#cereal/libmessaging.a')]
visionipc = [File('#cereal/libvisionipc.a')]

Export('cereal', 'messaging', 'visionipc')


SConscript(["src/SConscript"])