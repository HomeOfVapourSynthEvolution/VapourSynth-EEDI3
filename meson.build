project('EEDI3', 'cpp',
    default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
    license: 'GPL-2.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '9.0',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

libs = []

add_project_arguments('-DVS_USE_LATEST_API', language: 'cpp')

if host_machine.cpu_family().startswith('x86')
    add_project_arguments('-DEEDI3_X86', language: 'cpp')

    libs += static_library('instrset_detect',
        files('EEDI3/vectorclass/instrset_detect.cpp'),
        gnu_symbol_visibility: 'hidden',
    )

    libs += static_library('sse4',
        files('EEDI3/EEDI3_SSE4.cpp'),
        cpp_args: '-march=x86-64-v2',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )

    libs += static_library('avx2',
        files('EEDI3/EEDI3_AVX2.cpp'),
        cpp_args: '-march=x86-64-v3',
        gnu_symbol_visibility: 'hidden',
        include_directories: incdir,
    )
endif

shared_module('eedi3m',
    files('EEDI3/EEDI3.cpp'),
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    link_with: libs,
    name_prefix: '',
)
