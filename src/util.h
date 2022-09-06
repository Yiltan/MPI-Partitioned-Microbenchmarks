#ifndef UTIL_H
#define UTIL_H

#include <config.h>
#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>    /* for getopt_long */
#include <string.h>    /* for strchr */
#include "mpi.h"

#define CACHE_BUFF_SIZE (8 * 1024 * 1024)
#define WIDTH 20
#define MAX_THREADS 64

#pragma GCC diagnostic ignored "-Wunused-variable"
static char *cache_buf = NULL;

enum NoiseType {Single, Uniform, Gaussian};

typedef struct bm_config_t {
  int threads;
  int use_hot_cache;
  int min_message_size; //TODO: change to size_t
  int max_message_size; //TODO: change to size_t
  int compute_time;
  int percent_noise;
  enum NoiseType noise_type;
  int iterations;
  int warmup_iterations;
} bm_config_t;

typedef struct interval_t {
  double start;
  double end;
} interval_t;

// TODO: Remove and do something nicer
typedef struct times_t {   // Data Struct for measurements of each iteration
  double t_0[MAX_THREADS]; // Exiting Compute
  double t_1[MAX_THREADS]; // Data Recived
  double t_2[MAX_THREADS]; //
  double t_1_single_send[MAX_THREADS];
  double t_2_single_send;
} times_t;

/* TODO:
 * Return int to state status value
 * Error Check args
 * Exit on incorrect args
 */
void parse_args(int argc, char **argv, bm_config_t *config);
void print_args(bm_config_t *config, char *name);
void compute_time(double time_ms, int noise, enum NoiseType type);
void invalidate_cache(int yes_no);

int compareDoubles(const void *a, const void *b);
int compareIntervals(const void *a, const void *b);
void sort_doubles(double *list, int count);
void sort_interval_t(interval_t *list, int count);

double mean(double *list, int n);

#endif /* UTIL_H */
