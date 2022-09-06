#include <config.h>
#include "../util.h"
#include "../MPIPCL/mpipcl.h"
#include "mpi.h"
#include "omp.h"

int
main(int argc, char **argv)
{
  bm_config_t config;
  double *message;
  double *total_communication_time;
  int provided, myrank, flag, count;
  int min, max, partitions;
  int comp_ms, percent_noise;
  enum NoiseType noise_type;
  int num_iters;
  times_t time_stamp;
  MPI_Request request;

  MPI_Info info = MPI_INFO_NULL;
  const int dsize = (int) sizeof(double);
  int source = 0,
      dest = 1,
      tag = 0;

  // Initalise MPI
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE)
    MPI_Abort(MPI_COMM_WORLD, 1);

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // Parse Parameters
  parse_args(argc, argv, &config);
  print_args(&config, __FILE__);

  comp_ms = config.compute_time;
  percent_noise = config.percent_noise;
  noise_type = config.noise_type;
  partitions = config.threads;
  min = config.min_message_size;
  max = config.max_message_size;
  num_iters = config.iterations + config.warmup_iterations;
  omp_set_num_threads(config.threads);

  message = (double*) malloc(config.max_message_size);
  total_communication_time = (double*) calloc(num_iters, dsize);

  if (!config.use_hot_cache) {
    cache_buf = (char*) calloc(CACHE_BUFF_SIZE, sizeof(char));
  }

  if (myrank  == source)
  {
    printf("%-*s%-*s\n", WIDTH, "Message Size (B)", WIDTH, "Perceived Bandwidth (MB/s)");
  }

  MPI_Barrier(MPI_COMM_WORLD); // Init MPI_Wtime()

  for (int message_size = min; message_size<=max; message_size *= 2) {
    count = (message_size / dsize) / (partitions);

    // Init Buffers
    if (myrank == source)
    {
      MPI_Psend_init(message, partitions, count, MPI_DOUBLE, dest, tag,
                      MPI_COMM_WORLD, info, &request);
    }
    else if (myrank == dest)
    {
      MPI_Precv_init(message, partitions, count, MPI_DOUBLE, source, tag,
                      MPI_COMM_WORLD, info, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Starting Benchmark
    for (int itr=0; itr<num_iters; itr++) {
      flag = 0;
      MPI_Start(&request);

      invalidate_cache(!config.use_hot_cache);

      if (myrank == source) {
        #pragma omp parallel for shared(request)
        for (int i=0; i<config.threads; i++) {
          int tid = omp_get_thread_num();
          compute_time(comp_ms, percent_noise, noise_type);
          MPI_Pready(tid, &request);
          time_stamp.t_0[tid] = MPI_Wtime();
        }

        while(!flag){
          MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        }

        // Notifiy Completion
        MPI_Recv(message, 1, MPI_CHAR, dest, 420, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        time_stamp.t_1[0] = MPI_Wtime();

        sort_doubles(time_stamp.t_0, config.threads);

        total_communication_time[itr] = time_stamp.t_1[0]
                                      - time_stamp.t_0[config.threads - 1];
      } else if (myrank == dest) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        // Notifiy Completion
        MPI_Send(message, 1, MPI_CHAR, source, 420, MPI_COMM_WORLD);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Do Final Calculate Average bandwidth for message size
    // TODO: Double check these calculations
    if (myrank  == source)
    {
      // Calculate Bandwidth for each itr
      double ave_communication_time;
      ave_communication_time = mean(&total_communication_time[config.warmup_iterations],
                                    config.iterations);

      double percived_bandwidth = ((double) message_size)
                                / (ave_communication_time * 1000.0 * 1000.0);

      printf("%-*d%*lf\n",
             WIDTH, message_size,
             (int) strlen("Perceived Bandwidth (MB/s)"), percived_bandwidth);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request_free(&request);
  }

  free(message);
  free(total_communication_time);
  free(cache_buf);

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
