SUBDIRS = . MPIPCL pt2pt comm_patterns

AM_CFLAGS = -Wall -Wextra -Wpedantic -Werror -fopenmp -lm -lstdc++

noinst_LIBRARIES = libnormal_distribution.a \
									 libutil.a

libnormal_distribution_a_SOURCES = normal_distribution.cpp
libnormal_distribution_a_CFLAGS = -fPIC $(AM_CFLAGS)

libutil_a_SOURCES = util.c
libutil_a_CFLAGS = $(AM_CFLAGS)
