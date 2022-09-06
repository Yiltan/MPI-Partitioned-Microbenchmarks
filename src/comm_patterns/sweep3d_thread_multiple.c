//
// This code has been exteneded from https://github.com/sstsimulator/ember/
//
// Copyright 2009-2018 Sandia Corporation. Under the terms
// of Contract DE-NA0003525 with Sandia Corporation, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2018, Sandia Corporation
// Copyright (c) 2021-2022, PPRL, Queen's University
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "../util.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
void get_position(const int rank, const int pex, const int pey, int* myX,
                  int* myY) {
  *myX = rank % pex;
  *myY = rank / pex;
}

void validate_grid(int rank, int world_size,
                   int pex, int pey,
                   int kba, int nz) {
  if (kba == 0) {
    if (rank == 0) {
      fprintf(stderr,
              "K-Blocking Factor must not be zero. Please specify -kba <value "
              "> 0>\n");
    }

    exit(-1);
  }

  if (nz % kba != 0) {
    if (rank == 0) {
      fprintf(stderr,
              "KBA must evenly divide NZ, KBA=%d, NZ=%d, remainder=%d (must be "
              "zero)\n",
              kba, nz, (nz % kba));
    }

    exit(-1);
  }

  if ((pex * pey) != world_size) {
    if (0 == rank) {
      fprintf(
          stderr,
          "Error: processor decomposition (%d x %d) != number of ranks (%d)\n",
          pex, pey, world_size);
    }

    exit(-1);
  }
}

int main(int argc, char* argv[]) {

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE)
    MPI_Abort(MPI_COMM_WORLD, 1);

  int me = -1;
  int world = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  bm_config_t config;
  // Parse Generic Parameters
  parse_args(argc, argv, &config);
  print_args(&config, __FILE__);

  int comp_ms = config.compute_time;
  int percent_noise = config.percent_noise;
  int noise_type = config.noise_type;
  int min = config.min_message_size;
  int max = config.max_message_size;
  int num_iters = config.iterations + config.warmup_iterations;
  omp_set_num_threads(config.threads);

  MPI_Status status[MAX_THREADS];
  double* xRecvBuffer = (double*)malloc(max);
  double* xSendBuffer = (double*)malloc(max);
  double* yRecvBuffer = (double*)malloc(max);
  double* ySendBuffer = (double*)malloc(max);


  int pex = sqrt(world);
  int pey = pex;
  int kba = MAX_THREADS;
  int vars = 4;
  int nz = 16 * MAX_THREADS;

  for (int message_size = min; message_size<=max; message_size *= 2) {

    int msg_count = message_size / sizeof(double);
    int nx  = msg_count / (kba * vars);
    int ny = nx;
    int myX = -1;
    int myY = -1;

    validate_grid(me, world, pex, pey, kba, nz);
    get_position(me, pex, pey, &myX, &myY);

    for (int i = 0; i < nx; ++i) {
      xRecvBuffer[i] = 0;
      xSendBuffer[i] = i;
    }

    for (int i = 0; i < ny; ++i) {
      yRecvBuffer[i] = 0;
      ySendBuffer[i] = i;
    }

    const int xUp = (myX != (pex - 1)) ? me + 1 : -1;
    const int xDown = (myX != 0) ? me - 1 : -1;

    const int yUp = (myY != (pey - 1)) ? me + pex : -1;
    const int yDown = (myY != 0) ? me - pex : -1;


    // We repeat this sequence twice because there are really 8 vertices in the 3D
    // data domain and we sweep from each of them, processing the top four first
    // and then the bottom four vertices next.
    double start, end;
    for (int i = 0; i < (num_iters * 2); ++i) {
      if (i == (config.warmup_iterations * 2))
      {
        start = MPI_Wtime();
      }

      if (0 == (i % 2))
      {
        invalidate_cache(!config.use_hot_cache);
      }

      // Recreate communication pattern of sweep from (0,0) towards (Px,Py)
      #pragma omp parallel for
      for (int k = 0; k < nz; k += kba) {
        int tid = omp_get_thread_num();

        if (xDown > -1) {
          MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 1000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        if (yDown > -1) {
          MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yDown, 1000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        compute_time(comp_ms, percent_noise, noise_type);

        if (xUp > -1) {
          MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 1000,
                   MPI_COMM_WORLD);
        }

        if (yUp > -1) {
          MPI_Send(ySendBuffer, (ny * kba * vars), MPI_DOUBLE, yUp, 1000,
                   MPI_COMM_WORLD);
        }
      }

      // Recreate communication pattern of sweep from (Px,0) towards (0,Py)
      #pragma omp parallel for
      for (int k = 0; k < nz; k += kba) {
        int tid = omp_get_thread_num();

        if (xUp > -1) {
          MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 2000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        if (yDown > -1) {
          MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yDown, 2000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        compute_time(comp_ms, percent_noise, noise_type);

        if (xDown > -1) {
          MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 2000,
                   MPI_COMM_WORLD);
        }

        if (yUp > -1) {
          MPI_Send(ySendBuffer, (ny * kba * vars), MPI_DOUBLE, yUp, 2000,
                   MPI_COMM_WORLD);
        }
      }

      // Recreate communication pattern of sweep from (Px,Py) towards (0,0)
      #pragma omp parallel for
      for (int k = 0; k < nz; k += kba) {
        int tid = omp_get_thread_num();

        if (xUp > -1) {
          MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 3000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        if (yUp > -1) {
          MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yUp, 3000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        compute_time(comp_ms, percent_noise, noise_type);

        if (xDown > -1) {
          MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 3000,
                   MPI_COMM_WORLD);
        }

        if (yDown > -1) {
          MPI_Send(ySendBuffer, (ny * kba * vars), MPI_DOUBLE, yDown, 3000,
                   MPI_COMM_WORLD);
        }
      }

      // Recreate communication pattern of sweep from (0,Py) towards (Px,0)
      #pragma omp parallel for
      for (int k = 0; k < nz; k += kba) {
        int tid = omp_get_thread_num();

        if (xDown > -1) {
          MPI_Recv(xRecvBuffer, (nx * kba * vars), MPI_DOUBLE, xDown, 4000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        if (yUp > -1) {
          MPI_Recv(yRecvBuffer, (ny * kba * vars), MPI_DOUBLE, yUp, 4000,
                   MPI_COMM_WORLD, &status[tid]);
        }

        compute_time(comp_ms, percent_noise, noise_type);

        if (xUp > -1) {
          MPI_Send(xSendBuffer, (nx * kba * vars), MPI_DOUBLE, xUp, 4000,
                   MPI_COMM_WORLD);
        }

        if (yDown > -1) {
          MPI_Send(ySendBuffer, (ny * kba * vars), MPI_DOUBLE, yDown, 4000,
                   MPI_COMM_WORLD);
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    double timeTaken = end - start;
    double bytesXchng =
        ((double) config.iterations) *
        (((double)(xUp > -1   ? sizeof(double) * nx * kba * vars * 2 : 0)) +
         ((double)(xDown > -1 ? sizeof(double) * nx * kba * vars * 2 : 0)) +
         ((double)(yUp > -1   ? sizeof(double) * ny * kba * vars * 2 : 0)) +
         ((double)(yDown > -1 ? sizeof(double) * ny * kba * vars * 2 : 0)));

    double mean_timeTaken = 0.0;
    double mean_bytesXchng = 0.0;

    MPI_Reduce(&timeTaken,  &mean_timeTaken,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bytesXchng, &mean_bytesXchng, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    mean_timeTaken  = mean_timeTaken  / ((double) world);
    mean_bytesXchng = mean_bytesXchng / ((double) world);

    if (me == 0) {
      if (message_size == min) // if first data point, then print headers
      {
        printf("# Results from rank: %d\n", me);
        printf("# %20s %20s %20s %20s\n", "Size", "Time", "KBytesXchng/Rank", "KB/S/Rank");
      }

      printf("  %20d %20.6f %20.4f %20.4f\n",
             message_size,
             mean_timeTaken,
             (mean_bytesXchng / 1024.0),
             ((mean_bytesXchng / 1024.0) / mean_timeTaken));
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
