#include "../util.h"
#include "mpi.h"
#include "omp.h"

int
main(int argc, char **argv)
{
  bm_config_t config;
  double *message;
  double *total_communication_time;
  double *total_overlap_communication_time;
  int provided, myrank, flag, count;
  int min, max, partitions;
  int comp_ms, percent_noise;
  enum NoiseType noise_type;
  int num_iters;
  double mean_pt2pt_latency;
  times_t time_stamp;
  MPI_Request request;
  MPI_Request notification_request;
  char notification_flags[MAX_THREADS];
  interval_t intervals[MAX_THREADS];

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
  total_overlap_communication_time = (double*) calloc(num_iters, dsize);

  if (!config.use_hot_cache) {
    cache_buf = (char*) calloc(CACHE_BUFF_SIZE, sizeof(char));
  }

  if (myrank  == source)
  {
    printf("%-*s%-*s\n", WIDTH, "Message Size (B)", WIDTH, "Overhead");
  }

  MPI_Barrier(MPI_COMM_WORLD); // Init MPI_Wtime()

  for (int message_size = min; message_size<=max; message_size *= 2) {
    // Starting Benchmark For Single Send
    count = (message_size / dsize);
    for (int itr=0; itr<num_iters; itr++) {
      invalidate_cache(!config.use_hot_cache);

      MPI_Barrier(MPI_COMM_WORLD);

      if (myrank == source) {
        #pragma omp parallel for
        for (int i=0; i<config.threads; i++) {
          compute_time(comp_ms, percent_noise, noise_type);
        }
        time_stamp.t_0[0] = MPI_Wtime();

        MPI_Send(message, count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        MPI_Recv(message, 1, MPI_CHAR, dest, 420, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        time_stamp.t_1[0] = MPI_Wtime();

        total_communication_time[itr] = time_stamp.t_1[0] - time_stamp.t_0[0];

      } else if (myrank == dest) {
        MPI_Recv(message, count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(message, 1, MPI_CHAR, source, 420, MPI_COMM_WORLD);
      }
    }

    if (myrank == source) {
      mean_pt2pt_latency = mean(&total_communication_time[config.warmup_iterations],
                                config.iterations);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Initialising Buffers for MPI Partitioned
    count = (message_size / dsize) / (partitions);
    if (myrank == source)
    {
      MPI_Psend_init(message, partitions, count, MPI_DOUBLE, dest, tag,
                      MPI_COMM_WORLD, info, &request);
      MPI_Precv_init(notification_flags, partitions, 1, MPI_CHAR, dest, 420,
                      MPI_COMM_WORLD, info, &notification_request);
    }
    else if (myrank == dest)
    {
      MPI_Precv_init(message, partitions, count, MPI_DOUBLE, source, tag,
                      MPI_COMM_WORLD, info, &request);
      MPI_Psend_init(notification_flags, partitions, 1, MPI_CHAR, source, 420,
                      MPI_COMM_WORLD, info, &notification_request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Starting Benchmark For MPI Partitioned
    for (int itr=0; itr<num_iters; itr++) {
      invalidate_cache(!config.use_hot_cache);

      MPI_Barrier(MPI_COMM_WORLD);

      flag = 0;
      MPI_Start(&request);
      MPI_Start(&notification_request);

      if (myrank == source) {
        #pragma omp parallel for shared(request)
        for (int i=0; i<config.threads; i++) {
          int tid = omp_get_thread_num();
          compute_time(comp_ms, percent_noise, noise_type);
          MPI_Pready(tid, request);
          time_stamp.t_0[tid] = MPI_Wtime();

          int arr_flag = 0;
          while(!arr_flag) {
            MPI_Parrived(notification_request, tid, &arr_flag);
          }
          time_stamp.t_1[tid] = MPI_Wtime();
        }

        int flag_2 = 0;
        while(!(flag & flag_2)){
          MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
          MPI_Test(&notification_request, &flag_2, MPI_STATUS_IGNORE);
        }

        // Calculate Total Communication Time
        // Populate intervals
        for(int tid=0; tid<config.threads; tid++)
        {
          intervals[tid].start = time_stamp.t_0[tid];
          intervals[tid].end = time_stamp.t_1[tid];
        }

        qsort((void*) intervals, config.threads, sizeof(interval_t), compareIntervals);

        // Merge overlapping intervals
        interval_t non_overlapped_intervals[MAX_THREADS];
        int num_non_overlapped_intervals = 0;
        total_overlap_communication_time[itr] = 0.0;

        for(int tid=0; tid<config.threads; tid++)
        {
          // If interval i does not overlaps with i+1 or the last i
          if ((intervals[tid].end < intervals[tid+1].start) ||
              (tid == (config.threads - 1)))
          {
            // then append to list
            non_overlapped_intervals[num_non_overlapped_intervals].start = intervals[tid].start;
            non_overlapped_intervals[num_non_overlapped_intervals].end = intervals[tid].end;
            num_non_overlapped_intervals++;

          }
          else // If they do overlap
          {
            // Then merge i into i+1
            intervals[tid+1].start = intervals[tid].start;

            total_overlap_communication_time[itr] += intervals[tid].end - intervals[tid+1].start;
          }
        }
        // Calculate total time in intervals
        total_communication_time[itr] = 0.0;
        for (int tid=0; tid<num_non_overlapped_intervals; tid++) {
          total_communication_time[itr] += non_overlapped_intervals[tid].end - non_overlapped_intervals[tid].start;
        }
      } else if (myrank == dest) {
        #pragma omp parallel for shared(request)
        for (int i=0; i<config.threads; i++) {
          int arr_flag = 0;
          int tid = omp_get_thread_num();

          while(!arr_flag) {
            MPI_Parrived(request, tid, &arr_flag);
          }

          MPI_Pready(tid, notification_request);
        }

        int flag_2 = 0;
        while(!(flag & flag_2)){
          MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
          MPI_Test(&notification_request, &flag_2, MPI_STATUS_IGNORE);
        }
      }
    }

    if (myrank  == source)
    {
      double mean_part_latency;
      mean_part_latency = mean(&total_communication_time[config.warmup_iterations],
                               config.iterations);

      #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
      double overhead = mean_part_latency
                      / mean_pt2pt_latency;

      printf("%-*d%*lf\n", WIDTH, message_size, WIDTH, overhead);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request_free(&request);
    MPI_Request_free(&notification_request);
  }

  free(message);
  free(total_communication_time);

  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
