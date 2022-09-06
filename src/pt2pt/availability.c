#include "../util.h"
#include "mpi.h"
#include "omp.h"

int
main(int argc, char **argv)
{
  bm_config_t config;
  double *message;
  double *fork_join_time;
  double *total_communication_time;
  double *total_communication_time_after_join;
  int provided, myrank, flag, count;
  int min, max, partitions;
  int comp_ms, percent_noise;
  enum NoiseType noise_type;
  int num_iters;
  double mean_fork_join_latency;
  double mean_communication_time;
  times_t time_stamp;
  MPI_Request request;
  MPI_Request notification_request;
  char notification_flags[MAX_THREADS];

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

  // add these to comandlineflags
  comp_ms = config.compute_time;
  percent_noise = config.percent_noise;
  noise_type = config.noise_type;
  partitions = config.threads;
  min = config.min_message_size;
  max = config.max_message_size;
  num_iters = config.iterations + config.warmup_iterations;
  omp_set_num_threads(config.threads);

  message = (double*) malloc(config.max_message_size);
  fork_join_time = (double*) calloc(num_iters, dsize);
  total_communication_time = (double*) calloc(num_iters, dsize);
  total_communication_time_after_join = (double*) calloc(num_iters, dsize);

  if (!config.use_hot_cache) {
    cache_buf = (char*) calloc(CACHE_BUFF_SIZE, sizeof(char));
  }

  if (myrank  == source)
  {
    omp_set_num_threads(config.threads);
    printf("%-*s%-*s\n",
           WIDTH, "Message Size (B)",
           WIDTH, "Availability %");
  }

  // Init MPI_Wtime()
  MPI_Barrier(MPI_COMM_WORLD);

  for (int message_size = min; message_size<=max; message_size *= 2) {
    count = (message_size / dsize);
    for (int itr=0; itr<num_iters; itr++) {
      invalidate_cache(!config.use_hot_cache);

      MPI_Barrier(MPI_COMM_WORLD);

      if (myrank == source) {
        time_stamp.t_0[0] = MPI_Wtime();
        #pragma omp parallel for
        for (int i=0; i<config.threads; i++) {
          compute_time(comp_ms, percent_noise, noise_type);
        }
        time_stamp.t_1[0] = MPI_Wtime();

        MPI_Send(message, count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        MPI_Recv(message, 1, MPI_CHAR, dest, 420, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        time_stamp.t_2[0] = MPI_Wtime();

        fork_join_time[itr] = time_stamp.t_1[0] - time_stamp.t_0[0];

        // Assume Send/Recv after join
        total_communication_time[itr] = time_stamp.t_2[0] - time_stamp.t_1[0];
      } else if (myrank == dest) {
        MPI_Recv(message, count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(message, 1, MPI_CHAR, source, 420, MPI_COMM_WORLD);
      }
    }

    if (myrank == source) {
      mean_fork_join_latency = mean(&fork_join_time[config.warmup_iterations],
                                    config.iterations);
      mean_communication_time = mean(&total_communication_time[config.warmup_iterations],
                                     config.iterations);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    count = (message_size / dsize) / (partitions);

    // Init Buffers
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

    // Starting Partitioned Benchmark
    for (int itr=0; itr<num_iters; itr++) {
       invalidate_cache(!config.use_hot_cache);

      flag = 0;
      MPI_Start(&request);
      MPI_Start(&notification_request);

      if (myrank == source) {
        time_stamp.t_0[0] = MPI_Wtime();
        #pragma omp parallel for shared(request)
        for (int i=0; i<config.threads; i++) {
          int tid = omp_get_thread_num();
          compute_time(comp_ms, percent_noise, noise_type);
          MPI_Pready(tid, &request);
          time_stamp.t_1[tid] = MPI_Wtime();

          int arr_flag = 0;
          while(!arr_flag) {
            MPI_Parrived(&notification_request, tid, &arr_flag);
          }
          time_stamp.t_2[tid] = MPI_Wtime();
        }

        int flag_2 = 0;
        while(!(flag & flag_2)){
          MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
          MPI_Test(&notification_request, &flag_2, MPI_STATUS_IGNORE);
        }

        // Populate intervals
        interval_t intervals[MAX_THREADS];
        for(int tid=0; tid<config.threads; tid++)
        {
          // These should be relative to t_0
          intervals[tid].start = time_stamp.t_1[tid] - time_stamp.t_0[0];
          intervals[tid].end = time_stamp.t_2[tid] - time_stamp.t_0[0];
        }

        qsort((void*) intervals, config.threads, sizeof(interval_t), compareIntervals);

        // Merge overlapping intervals
        interval_t non_overlapped_intervals[MAX_THREADS];
        int num_non_overlapped_intervals = 0;

        for(int i=0; i<config.threads; i++)
        {
          // If interval i does not overlaps with i+1 or the last i
          if ((intervals[i].end < intervals[i+1].start) ||
              (i == (config.threads - 1)))
          {
            // then append to list
            non_overlapped_intervals[num_non_overlapped_intervals].start = intervals[i].start;
            non_overlapped_intervals[num_non_overlapped_intervals].end = intervals[i].end;
            num_non_overlapped_intervals++;

          }
          else // If they do overlap
          {
            // Then merge i into i+1
            intervals[i+1].start = intervals[i].start;
          }
        }

        // Calculate total time in intervals
        total_communication_time[itr] = 0.0;
        total_communication_time_after_join[itr] = 0.0;
        for (int i=0; i<num_non_overlapped_intervals; i++) {
          if (non_overlapped_intervals[i].end > mean_fork_join_latency) {
            if (non_overlapped_intervals[i].start < mean_fork_join_latency) {
              #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
              total_communication_time_after_join[itr] = total_communication_time_after_join[itr]
                                                       + non_overlapped_intervals[i].end
                                                       - mean_fork_join_latency;
            } else {
              total_communication_time_after_join[itr] = total_communication_time_after_join[itr]
                                                       + non_overlapped_intervals[i].end
                                                       - non_overlapped_intervals[i].start;
            }
          }
        }
      } else if (myrank == dest) {
        #pragma omp parallel for shared(request)
        for (int i=0; i<config.threads; i++) {
          int arr_flag = 0;
          int tid = omp_get_thread_num();

          while(!arr_flag) {
            MPI_Parrived(&request, tid, &arr_flag);
          }

          MPI_Pready(tid, &notification_request);
        }

        int flag_2 = 0;
        while(!(flag & flag_2)){
          MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
          MPI_Test(&notification_request, &flag_2, MPI_STATUS_IGNORE);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Do Final Calculate Average bandwidth for message size
    // TODO: Double check these calculations
    if (myrank  == source)
    {
      // Calculate Bandwidth for each itr
      double ave_communication_time_after_join
        = mean(&total_communication_time_after_join[config.warmup_iterations],
               config.iterations);

      double availability = (mean_communication_time - ave_communication_time_after_join)
                          / mean_communication_time;
      availability = availability * 100.0;

      printf("%-*d%*lf\n",
             WIDTH, message_size,
             (int) strlen("Availability %"), availability);
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
