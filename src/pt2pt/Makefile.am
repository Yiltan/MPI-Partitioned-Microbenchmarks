bin_PROGRAMS = overhead              \
							 perceived_bandwidth   \
			   		   availability 				 \
  						 early_bird

# TODO: Is there a better way to add these flags
# -lm is for math.h
AM_CFLAGS = -Wall -Wextra -Wpedantic -Werror -fopenmp -lm -lstdc++

LDADD =../libutil.a ../libnormal_distribution.a

overhead_SOURCES = overhead.c
overhead_CFLAGS = $(AM_CFLAGS)
overhead_LDADD = $(LDADD)

perceived_bandwidth_SOURCES = perceived_bandwidth.c
perceived_bandwidth_CFLAGS = $(AM_CFLAGS)
perceived_bandwidth_LDADD = $(LDADD)

availability_SOURCES = availability.c
availability_CFLAGS = $(AM_CFLAGS)
availability_LDADD = $(LDADD)

early_bird_SOURCES = early_bird.c
early_bird_CFLAGS = $(AM_CFLAGS)
early_bird_LDADD = $(LDADD)
