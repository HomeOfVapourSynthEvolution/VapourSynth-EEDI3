project('EEDI3', 'cpp',
  default_options: ['buildtype=release', 'optimization=2', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'GPL-2.0-or-later',
  meson_version: '>=0.51.0',
  version: '5',
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_project_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : '/GS-', language: 'cpp')
endif

sources = files('EEDI3/EEDI3.cpp')

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = []
  install_dir = get_option('libdir') / 'vapoursynth'
endif

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DEEDI3_X86', language: 'cpp')

  libs += static_library('sse2',
    files(
      'EEDI3/EEDI3_SSE2.cpp',
      'EEDI3/vectorclass/instrset_detect.cpp',
    ),
    cpp_args: gcc_syntax ? ['-msse2'] : '/arch:SSE2',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('avx2',
    files('EEDI3/EEDI3_AVX2.cpp'),
    cpp_args: gcc_syntax ? ['-mavx2', '-mfma'] : '/arch:AVX2',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('avx512',
    files('EEDI3/EEDI3_AVX512.cpp'),
    cpp_args: gcc_syntax ? ['-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq', '-mfma'] : '/arch:AVX512',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )
endif

shared_module('eedi3m',
  sources,
  dependencies: vapoursynth_dep,
  link_with: libs,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden',
)
