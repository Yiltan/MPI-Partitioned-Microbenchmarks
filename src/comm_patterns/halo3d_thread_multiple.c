//
// This code has been exteneded from https://github.com/sstsimulator/ember/
//
// Copyright 2009-2018 Sandia Corporation. Under the terms
// of Contract DE-NA0003525 with Sandia Corporation, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2018, Sandia Corporation
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
void get_position(const int rank, const int pex, const int pey, const int pez,
                  int* myX, int* myY, int* myZ) {
  const int plane = rank % (pex * pey);
  *myY = plane / pex;
  *myX = (plane % pex) != 0 ? (plane % pex) : 0;
  *myZ = rank / (pex * pey);
}

int convert_position_to_rank(const int pX, const int pY, const int pZ,
                             const int myX, const int myY, const int myZ) {
  // Check if we are out of bounds on the grid
  if ((myX < 0) || (myY < 0) || (myZ < 0) || (myX >= pX) || (myY >= pY) ||
      (myZ >= pZ)) {
    return -1;
  } else {
    return (myZ * (pX * pY)) + (myY * pX) + myX;
  }
}

void validate_halo_inputs_mt(int rank, int world_size, int pex, int pey, int pez, int threads)
{
  if ((pex * pey * pez) != world_size) {
    if (0 == rank) {
      fprintf(stderr, "Error: rank grid does not equal number of ranks.\n");
      fprintf(stderr, "%7d x %7d x %7d != %7d\n", pex, pey, pez, world_size);
    }
    exit(-1);
  }

  double cbrt_double = cbrt(world_size);
  int cbrt_int = (int) cbrt(world_size);
  int is_cbrt_thread_count = (cbrt_double == (double) cbrt_int);

  if (!is_cbrt_thread_count) {
    if (0 == rank) {
      fprintf(stderr, "Error: thread count must be a cubed number.\n");
      fprintf(stderr, "cbrt(%7d) = %0.3lf\n", threads, cbrt_double);
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
  int thread_dim = (int) cbrt(config.threads);

  double* xUpSendBuffer   = (double*)malloc(max);
  double* xUpRecvBuffer   = (double*)malloc(max);
  double* xDownSendBuffer = (double*)malloc(max);
  double* xDownRecvBuffer = (double*)malloc(max);
  double* yUpSendBuffer   = (double*)malloc(max);
  double* yUpRecvBuffer   = (double*)malloc(max);
  double* yDownSendBuffer = (double*)malloc(max);
  double* yDownRecvBuffer = (double*)malloc(max);
  double* zUpSendBuffer   = (double*)malloc(max);
  double* zUpRecvBuffer   = (double*)malloc(max);
  double* zDownSendBuffer = (double*)malloc(max);
  double* zDownRecvBuffer = (double*)malloc(max);

  MPI_Request* requests;
  requests = (MPI_Request*)malloc(sizeof(MPI_Request) * 3 * 4 * config.threads);

  int pex = (int) cbrt(world);
  int pey = pex;
  int pez = pex;

  validate_halo_inputs_mt(me, world, pex, pey, pez, config.threads);

  int posX, posY, posZ;
  get_position(me, pex, pey, pez, &posX, &posY, &posZ);

  int xUp   = convert_position_to_rank(pex, pey, pez, posX + 1, posY, posZ);
  int xDown = convert_position_to_rank(pex, pey, pez, posX - 1, posY, posZ);
  int yUp   = convert_position_to_rank(pex, pey, pez, posX, posY + 1, posZ);
  int yDown = convert_position_to_rank(pex, pey, pez, posX, posY - 1, posZ);
  int zUp   = convert_position_to_rank(pex, pey, pez, posX, posY, posZ + 1);
  int zDown = convert_position_to_rank(pex, pey, pez, posX, posY, posZ - 1);

  /******************************************************************************
   * We choose message_size *= 4, and vars = 2
   * so that dim = sqrt(ny_nz) gives us a whole number
   ******************************************************************************/

  for (size_t message_size = min; message_size<=((size_t)max); message_size *= 4) {
    int vars = 2;

    int ny_nz = ((int) message_size) / (sizeof(double) * vars);
    int dim = sqrt(ny_nz);
    int nx = dim;
    int ny = dim;
    int nz = dim;


    MPI_Barrier(MPI_COMM_WORLD);

    double start, end;
    for (int i = 0; i < num_iters; i++) {
      if (i == config.warmup_iterations)
      {
        start = MPI_Wtime();
      }

      invalidate_cache(!config.use_hot_cache);

      #pragma omp parallel
      {
        int requestcount = 0;
        int tid = omp_get_thread_num();
        int t_posX, t_posY, t_posZ;
        get_position(tid, thread_dim, thread_dim, thread_dim, &t_posX, &t_posY, &t_posZ);

        int t_xUp   = t_posX == (thread_dim - 1);
        int t_xDown = t_posX == 0;
        int t_yUp   = t_posY == (thread_dim - 1);
        int t_yDown = t_posY == 0;
        int t_zUp   = t_posZ == (thread_dim - 1);
        int t_zDown = t_posZ == 0;

        compute_time(comp_ms, percent_noise, noise_type);

        if (xUp > -1 && t_xUp) {
          MPI_Irecv(xUpRecvBuffer, (ny * nz * vars) / config.threads, MPI_DOUBLE, xUp, 1000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
          MPI_Isend(xUpSendBuffer, (ny * nz * vars) / config.threads, MPI_DOUBLE, xUp, 1000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
        }

        if (xDown > -1 && t_xDown) {
          MPI_Irecv(xDownRecvBuffer, (ny * nz * vars) / config.threads, MPI_DOUBLE, xDown, 1000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
          MPI_Isend(xDownSendBuffer, (ny * nz * vars) / config.threads, MPI_DOUBLE, xDown, 1000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
        }

        if (yUp > -1 && t_yUp) {
          MPI_Irecv(yUpRecvBuffer, (nx * nz * vars) / config.threads, MPI_DOUBLE, yUp, 2000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
          MPI_Isend(yUpSendBuffer, (nx * nz * vars) / config.threads, MPI_DOUBLE, yUp, 2000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
        }

        if (yDown > -1 && t_yDown) {
          MPI_Irecv(yDownRecvBuffer, (nx * nz * vars) / config.threads, MPI_DOUBLE, yDown, 2000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
          MPI_Isend(yDownSendBuffer, (nx * nz * vars) / config.threads, MPI_DOUBLE, yDown, 2000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
        }

        if (zUp > -1 && t_zUp) {
          MPI_Irecv(zUpRecvBuffer, (nx * ny * vars) / config.threads, MPI_DOUBLE, zUp, 4000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
          MPI_Isend(zUpSendBuffer, (nx * ny * vars) / config.threads, MPI_DOUBLE, zUp, 4000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
        }

        if (zDown > -1 && t_zDown) {
          MPI_Irecv(zDownRecvBuffer, (nx * ny * vars) / config.threads, MPI_DOUBLE, zDown, 4000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
          MPI_Isend(zDownSendBuffer, (nx * ny * vars) / config.threads, MPI_DOUBLE, zDown, 4000,
                    MPI_COMM_WORLD, &requests[(tid * config.threads) + (requestcount++)]);
        }

        MPI_Waitall(requestcount, &requests[(tid * config.threads)], MPI_STATUSES_IGNORE);
        requestcount = 0;
      }
    }

    end = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    const double timeTaken = end - start;
    const double bytesXchng =
        ((double)(xUp > -1 ? sizeof(double) * ny * nz * 2 * vars : 0)) +
        ((double)(xDown > -1 ? sizeof(double) * ny * nz * 2 * vars : 0)) +
        ((double)(yUp > -1 ? sizeof(double) * nx * nz * 2 * vars : 0)) +
        ((double)(yDown > -1 ? sizeof(double) * nx * nz * 2 * vars : 0)) +
        ((double)(zUp > -1 ? sizeof(double) * nx * ny * 2 * vars : 0)) +
        ((double)(zDown > -1 ? sizeof(double) * nx * ny * 2 * vars : 0));

    double mean_timeTaken = 0.0;
    double mean_bytesXchng = 0.0;

    MPI_Reduce(&timeTaken,  &mean_timeTaken,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bytesXchng, &mean_bytesXchng, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    mean_timeTaken  = mean_timeTaken  / ((double) world);
    mean_bytesXchng = mean_bytesXchng / ((double) world);

    if (me == 0) {
      if ((int) message_size == min) // if first data point, then print headers
      {
        printf("# %20s %20s %20s %20s\n", "Size", "Time", "KBytesXchng/Rank", "KB/S/Rank");
      }

      printf("  %20d %20.6f %20.4f %20.4f\n",
             (int) message_size,
             mean_timeTaken,
             (mean_bytesXchng / 1024.0),
             ((mean_bytesXchng / 1024.0) / mean_timeTaken));
    }

    MPI_Barrier(MPI_COMM_WORLD);

  }

  free(xUpRecvBuffer);
  free(xDownRecvBuffer);
  free(yUpRecvBuffer);
  free(yDownRecvBuffer);
  free(zUpRecvBuffer);
  free(zDownRecvBuffer);

  MPI_Finalize();
}
