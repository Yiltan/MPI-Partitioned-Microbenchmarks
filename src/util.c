#include "util.h"
#include "omp.h"
#include "normal_distribution.h"

void
parse_args(int argc, char **argv, bm_config_t *config) {
  // Default Values
  config->threads  = 1;
  config->use_hot_cache = 1;
  config->min_message_size  = 8;
  config->max_message_size  = (1 * 1024 * 1024); // 1MB
  config->compute_time = 100;
  config->percent_noise = 0;
  config->noise_type = Uniform;
  config->iterations  = 100;
  config->warmup_iterations  = 10;

  int c;
  int option_index = 0;

  // Add print csv file option

  static struct option long_options[] = {
    {"help",              no_argument,       0, 'h'},
    {"version",           no_argument,       0, 'v'},
    {"disable-hot-cache", no_argument,       0, 'd'}, // Later change to enable hot cache
    {"threads",           required_argument, 0, 't'},
    {"message-size",      required_argument, 0, 'm'},
    {"iterations",        required_argument, 0, 'i'},
    {"warmup",            required_argument, 0, 'x'},
    {"compute-time",      required_argument, 0, 'c'},
    {"percent-noise",     required_argument, 0, 'p'},
    {"noise-type",        required_argument, 0, 'n'}
  };

  char const * optstring = "hvcs:r:m:i:x:";

  while (1) {
      c = getopt_long(argc, argv, optstring, long_options, &option_index);

      if (c == -1)
        break;

      switch (c) {
        case 'v':
        {
          printf("Version: " PACKAGE_STRING "\n");
          exit(EXIT_SUCCESS);
        }

        case 'd':
        {
          config->use_hot_cache = 0;
          config->warmup_iterations = 0;
          break;
        }

        case 't':
        {
          config->threads = atoi(optarg); // Validate this for correctness
          break;
        }

        case 'm':
        {
          const char delimiter = ':';
          char *min_str = strtok(optarg, &delimiter);
          char *max_str = strtok(NULL,   &delimiter);

          // Make sure to validate this somehow
          config->min_message_size = atoi(min_str);
          config->max_message_size = atoi(max_str);
          break;
        }

        case 'i':
        {
          config->iterations = atoi(optarg);
          break;
        }

        case 'x':
        {
          config->warmup_iterations = atoi(optarg);
          break;
        }

        case 'c':
        {
          config->compute_time= atoi(optarg);
          break;
        }

        case 'p':
        {
          config->percent_noise = atoi(optarg);
          break;
        }

        case 'n':
        {
          if (!strcmp("Single", optarg))
          {
            config->noise_type = Single;
          }
          else if (!strcmp("Uniform", optarg))
          {
            config->noise_type = Uniform;
          }
          else if (!strcmp("Gaussian", optarg))
          {
            config->noise_type = Gaussian;
          }
          else
          {
            config->noise_type = Single;
            printf("Noise Type %s is not valid\n", optarg);
            printf("Using Noise Type %s\n", "Single");
          }
          break;
        }

        case 'h':
        {
          printf("Print Help\n");
          break;
        }
      }
  }

  init_normal_distribution((double) config->compute_time,
                           (double) config->percent_noise);
}

void print_args(bm_config_t *config, char *name)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
  {
    printf("#-----------------------------------------------#\n");
    printf("#            Benchmark Configuration            #\n");
    printf("#-----------------------------------------------#\n");
    printf("# %-*s  :  %-*s#\n", WIDTH, "Name",                 WIDTH, name);
    printf("# %-*s  :  %*d #\n", WIDTH, "Threads",              WIDTH, config->threads);
    printf("# %-*s  :  %*d #\n", WIDTH, "Use Hot Cache",        WIDTH, config->use_hot_cache);
    printf("# %-*s  :  %*d #\n", WIDTH, "Min Message Size (B)", WIDTH, config->min_message_size);
    printf("# %-*s  :  %*d #\n", WIDTH, "Max Message Size (B)", WIDTH, config->max_message_size);
    printf("# %-*s  :  %*d #\n", WIDTH, "Compute Time (ms)",    WIDTH, config->compute_time);
    printf("# %-*s  :  %*d #\n", WIDTH, "% Noise",              WIDTH, config->percent_noise);
    printf("# %-*s  :  %*d #\n", WIDTH, "Noise Type",           WIDTH, config->noise_type); // TODO: enum to string please
    printf("# %-*s  :  %*d #\n", WIDTH, "Iterations",           WIDTH, config->iterations);
    printf("# %-*s  :  %*d #\n", WIDTH, "Warm Up Iterations",   WIDTH, config->warmup_iterations);
    printf("#-----------------------------------------------#\n");
    printf("\n"); // Some additional space before printing other things
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void
compute_time(double time_ms, int noise, enum NoiseType type)
{
  if (time_ms <= 0)
  {
    return;
  }

  // Add random noise based on % noise
  if (noise > 0)
  {
    double rand_multiplier, time_percent;
    int tid = omp_get_thread_num(); // assume this is called in a parallel region

    switch (type) {
      case Single:
        rand_multiplier = (tid == 0);
        time_percent = (time_ms * (double) noise) / 100.0;
        time_ms = time_ms + (time_percent * rand_multiplier);
        break;
      case Gaussian:
        time_ms = get_random_normal_number(time_ms, noise);
        break;
      case Uniform:
      default:
        rand_multiplier = ((double) rand()) / ((double) RAND_MAX);
        time_percent = (time_ms * (double) noise) / 100.0;
        time_ms = time_ms + (time_percent * rand_multiplier);
    }
  }

  double t_0, t_1, diff_ms;
  long a = 0, b = 0, c = 0;

  t_0 = MPI_Wtime();

  do
  {
    // Do random work
    a++;
    b++;
    c = a * b + c;

    t_1 = MPI_Wtime();
    diff_ms = (t_1 - t_0) * 1000.0;
  } while(diff_ms <= time_ms);
}

void
invalidate_cache(int yes_no)
{
  if (yes_no)
  {
    if (cache_buf == NULL)
    {
      cache_buf = (char*) malloc(CACHE_BUFF_SIZE);
    }

    char *tmp = cache_buf;
    tmp[0] = 1;
    for (int i = 1 ; i < CACHE_BUFF_SIZE; ++i) {
      tmp[i] = tmp[i - 1];
    }
  }
}

int
compareDoubles(const void * a, const void * b)
{
  if ( *(double*)a <  *(double*)b ) return -1;
  if ( *(double*)a == *(double*)b ) return 0;
  //if ( *(double*)a >  *(double*)b ) return 1;
  //else
  return 1;
}

void
sort_doubles(double* list, int count)
{
    qsort((void*) list, count, sizeof(double), compareDoubles);
}

int
compareIntervals(const void *a, const void *b)
{
  interval_t *interval_a = (interval_t *) a;
  interval_t *interval_b = (interval_t *) b;

  if (interval_a->start < interval_b->start) return -1;
  if (interval_a->start == interval_b->start) return 0;
  //else
  return 1;
}

void
sort_interval_t(interval_t *list, int count)
{
    qsort((void*) list, count, sizeof(interval_t), compareIntervals);
}

double mean(double *list, int n)
{
  double total_time = 0.0;
  for (int i=0; i<n; i++)
  {
    total_time = total_time + list[i];
  }

  return total_time / (double) n;
}
