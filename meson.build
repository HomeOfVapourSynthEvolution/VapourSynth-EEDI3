project('EEDI3', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.48.0',
  version : '3'
)

add_project_arguments('-Wno-unused-variable', '-ffast-math', language : 'cpp')

sources = [
  'EEDI3/EEDI3.cpp',
  'EEDI3/EEDI3.hpp',
  'EEDI3/shared.hpp',
  'EEDI3/vectorclass/instrset.h',
  'EEDI3/vectorclass/instrset_detect.cpp'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

deps = [vapoursynth_dep]

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DVS_TARGET_CPU_X86', '-mfpmath=sse', '-msse2', language : 'cpp')

  sources += [
    'EEDI3/EEDI3_SSE2.cpp',
    'EEDI3/vectorclass/vectorclass.h',
    'EEDI3/vectorclass/vectorf128.h',
    'EEDI3/vectorclass/vectorf256.h',
    'EEDI3/vectorclass/vectorf256e.h',
    'EEDI3/vectorclass/vectorf512.h',
    'EEDI3/vectorclass/vectorf512e.h',
    'EEDI3/vectorclass/vectori128.h',
    'EEDI3/vectorclass/vectori256.h',
    'EEDI3/vectorclass/vectori256e.h',
    'EEDI3/vectorclass/vectori512.h',
    'EEDI3/vectorclass/vectori512e.h'
  ]

  if get_option('opencl')
    add_project_arguments('-DHAVE_OPENCL', language : 'cpp')

    sources += [
      'EEDI3/EEDI3CL.cpp',
      'EEDI3/EEDI3CL.hpp',
      'EEDI3/EEDI3CL_SSE2.cpp'
    ]

    opencl_dep = dependency('OpenCL', required : false)
    if not opencl_dep.found()
      opencl_dep = meson.get_compiler('cpp').find_library('OpenCL')
    endif

    boost_dep = dependency('boost', modules : ['filesystem', 'system'])

    deps += [opencl_dep, boost_dep]
  endif

  libs += static_library('sse4', 'EEDI3/EEDI3_SSE4.cpp',
    dependencies : [vapoursynth_dep],
    cpp_args : ['-msse4.1'],
    gnu_symbol_visibility : 'hidden'
  )

  libs += static_library('avx', 'EEDI3/EEDI3_AVX.cpp',
    dependencies : [vapoursynth_dep],
    cpp_args : ['-mavx'],
    gnu_symbol_visibility : 'hidden'
  )

  libs += static_library('avx512', 'EEDI3/EEDI3_AVX512.cpp',
    dependencies : [vapoursynth_dep],
    cpp_args : ['-mavx512f', '-mavx512cd', '-mfma'],
    gnu_symbol_visibility : 'hidden'
  )
endif

shared_module('eedi3m', sources,
  dependencies : deps,
  link_with : libs,
  install : true,
  install_dir : join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility : 'hidden'
)
