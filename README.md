# MPI-Partitioned-Microbenchmarks

These are the MPI Partitioned Microbenchmarks used in the ICCP confrence paper
titled "Micro-Benchmarking MPI Partitioned Point-to-Point Communication".
This contains a benchmark suite to measure the following point-to-point
benchmarks:
- availability.c
- early_bird.c
- overhead.c
- perceived_bandwidth.c

And the following communication patterns:
- Halo3D
- Sweep3D

## Build Instructions

The results in the ICPP paper were collected using MPIPCL.
As were are not the authors of that library it is not included in this
repository, to recreate the results you can add your copy of that library to
`src/MPIPCL`.

This benchmark has not been tested with MPI native implemenations
of MPI Partitioned as they were not sufficiantly mature at the time of writing of this paper.
There is currently a branch `mpi-native` where ongoing work to port these
benchmarks to MPI native libraries is currently being conducted,
but they have not been fully tested.

The benchmarks can be built like so:

```
./autogen.sh
./configure CC=<mpicc_path> --prefix=<prefix_path>
make
make install
```

## Run Instructions

We have the following runtime parameters to use our benchmarks
```
--disable-hot-cache        # Invalidate the CPU cache with each iteration .
-threads <n>               # The number of threads to use, currently this is
                           # equal to the number of partitions.
-mmessage-size <min>:<max> # The message range for the benchmark to use.
-iterations <n>            # The number of iterations
-x <n>                     # The number of warmup iterations for hot-cache
-compute-time <n>          # The amount of compute time
-percent-noise <n>         # The percent noise [0, 100]
-noise-type <noise>        # The noise type [Single, Uniform, Gaussian]
```

You can run the tests like so:
```
mpirun -np 2 ./build/bin/perceived_bandwidth --iterations 100
                                             --threads 16
                                             --message-size 1024:4096
                                             --compute-time 100
                                             --percent-noise 4
                                             --noise-type Uniform
```

## Reference

```
@inproceedings{2022_TEMUCIN_ICPP,
  author = {Yıltan Hassan Temuçin, and Ryan E. Grant, and Ahmad Afsahi}
  title = {{Micro-Benchmarking MPI Partitioned Point-to-Point Communication}},
  year = {2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {},
  doi = {},
  booktitle = {{51st International Conference on Parallel Processing}},
  articleno = {},
  numpages = {},
  location = {},
  series = {ICPP '22}
}
```
