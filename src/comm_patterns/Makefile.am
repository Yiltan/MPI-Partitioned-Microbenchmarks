bin_PROGRAMS = \
			   sweep3d \
 			   sweep3d_thread_multiple \
 			   sweep3d_thread_partitioned \
				 halo3d \
				 halo3d_thread_multiple


# TODO: Is there a better way to add these flags
# -lm is for math.h
AM_CFLAGS = -Wall -Wextra -Wpedantic -Werror -fopenmp -lm -lstdc++

LDADD =../libutil.a ../libnormal_distribution.a

halo3d_SOURCES = halo3d.c
halo3d_CFLAGS = $(AM_CFLAGS)
halo3d_LDADD = $(LDADD)

halo3d_thread_multiple_SOURCES = halo3d_thread_multiple.c
halo3d_thread_multiple_CFLAGS = $(AM_CFLAGS)
halo3d_thread_multiple_LDADD = $(LDADD)

sweep3d_SOURCES = sweep3d.c
sweep3d_CFLAGS = $(AM_CFLAGS)
sweep3d_LDADD = $(LDADD)

sweep3d_thread_multiple_SOURCES = sweep3d_thread_multiple.c
sweep3d_thread_multiple_CFLAGS = $(AM_CFLAGS)
sweep3d_thread_multiple_LDADD = $(LDADD)

sweep3d_thread_partitioned_SOURCES = sweep3d_thread_partitioned.c
sweep3d_thread_partitioned_CFLAGS = $(AM_CFLAGS)
sweep3d_thread_partitioned_LDADD = $(LDADD)
