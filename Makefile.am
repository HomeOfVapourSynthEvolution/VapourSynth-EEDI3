warning_flags = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-attributes -Wno-ignored-attributes
common_cflags = -O3 -ffast-math $(warning_flags) $(MFLAGS)
AM_CXXFLAGS = -std=c++14 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libeedi3m.la

libeedi3m_la_SOURCES = EEDI3/EEDI3.cpp \
                       EEDI3/EEDI3.hpp \
                       EEDI3/shared.hpp \
                       EEDI3/vectorclass/instrset.h \
                       EEDI3/vectorclass/instrset_detect.cpp

if VS_TARGET_CPU_X86
libeedi3m_la_SOURCES += EEDI3/EEDI3_SSE2.cpp \
                        EEDI3/vectorclass/vectorclass.h \
                        EEDI3/vectorclass/vectorf128.h \
                        EEDI3/vectorclass/vectorf256.h \
                        EEDI3/vectorclass/vectorf256e.h \
                        EEDI3/vectorclass/vectori128.h \
                        EEDI3/vectorclass/vectori256.h \
                        EEDI3/vectorclass/vectori256e.h

noinst_LTLIBRARIES = libsse4.la libavx.la libavx2.la

libsse4_la_SOURCES = EEDI3/EEDI3_SSE4.cpp
libsse4_la_CXXFLAGS = $(AM_CXXFLAGS) -msse4.1

libavx_la_SOURCES = EEDI3/EEDI3_AVX.cpp
libavx_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx

libavx2_la_SOURCES = EEDI3/EEDI3_AVX2.cpp
libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mfma

libeedi3m_la_LIBADD = libsse4.la libavx.la libavx2.la
endif

if OPENCL
libeedi3m_la_SOURCES += EEDI3/EEDI3CL.cpp \
                        EEDI3/EEDI3CL.hpp \
                        EEDI3/EEDI3CL_SSE2.cpp
endif

libeedi3m_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS) $(OPENCLLDFLAGS)
