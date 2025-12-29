project('EEDI3', 'cpp',
  default_options: ['buildtype=release', 'optimization=2', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'GPL-2.0-or-later',
  meson_version: '>=0.58.0',
  version: '8',
)

gcc_syntax = meson.get_compiler('cpp').get_argument_syntax() == 'gcc'

if get_option('buildtype') == 'release'
  add_project_arguments(gcc_syntax ? ['-fno-math-errno', '-fno-trapping-math'] : '/GS-', language: 'cpp')
endif

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable('libdir') / 'vapoursynth'
else
  vapoursynth_dep = []
  install_dir = get_option('libdir') / 'vapoursynth'
endif

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DEEDI3_X86', language: 'cpp')

  libs += static_library('instrset_detect',
    files('EEDI3/vectorclass/instrset_detect.cpp'),
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('sse4',
    files('EEDI3/EEDI3_SSE4.cpp'),
    cpp_args: gcc_syntax ? '-march=x86-64-v2' : '/arch:SSE2',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )

  libs += static_library('avx2',
    files('EEDI3/EEDI3_AVX2.cpp'),
    cpp_args: gcc_syntax ? '-march=x86-64-v3' : '/arch:AVX2',
    dependencies: vapoursynth_dep,
    gnu_symbol_visibility: 'hidden',
  )
endif

shared_module('eedi3m',
  files('EEDI3/EEDI3.cpp'),
  dependencies: vapoursynth_dep,
  gnu_symbol_visibility: 'hidden',
  install: true,
  install_dir: install_dir,
  link_with: libs,
)
