#include "mpipcl.h"

int MPIX_Psend_init(void *buf, int partitions, MPI_Count count,
		                MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request)
{
  return MPI_SUCCESS;
}

int MPIX_Precv_init(void *buf, int partitions, MPI_Count count,
		                MPI_Datatype datatype, int src, int tag, MPI_Comm comm,
                    MPI_Info info, MPIX_Request *request)
{
  return MPI_SUCCESS;
}

int MPIX_Pready(int partition, MPIX_Request *request)
{
  return MPI_SUCCESS;
}

int MPIX_Pready_range(int partition_low, int partition_high, MPIX_Request *request)
{
  return MPI_SUCCESS;
}

int MPIX_Pready_list(int length, int array_of_partitions[], MPIX_Request *request)
{
  return MPI_SUCCESS;
}

int MPIX_Parrived(MPIX_Request *request, int partition, int *flag)
{
  return MPI_SUCCESS;
}

int MPIX_Start(MPIX_Request *request)
{
  return MPI_SUCCESS;
}

int MPIX_Startall(int count, MPIX_Request array_of_requests[])
{
  return MPI_SUCCESS;
}

int MPIX_Wait(MPIX_Request *request, MPI_Status *status)
{
  return MPI_SUCCESS;
}

int MPIX_Waitall(int count, MPIX_Request array_of_requests[],
                 MPI_Status array_of_statuses[])
{
  return MPI_SUCCESS;
}

int MPIX_Waitany(int count, MPIX_Request array_of_requests[],
                 int *index, MPI_Status *status)
{
  return MPI_SUCCESS;
}


//local attempt
int MPIX_Waitsome(int incount, MPIX_Request array_of_requests[],
                  int *outcount, int array_of_indices[],
                  MPI_Status array_of_statuses[])
{
  return MPI_SUCCESS;
}

int MPIX_Test(MPIX_Request *request, int *flag, MPI_Status *status)
{
  return MPI_SUCCESS;
}

int MPIX_Testall(int count, MPIX_Request array_of_requests[],
                 int *flag, MPI_Status array_of_statuses[])
{
  return MPI_SUCCESS;
}

int MPIX_Testany(int count, MPIX_Request array_of_requests[], int *index, int *flag, MPI_Status *status)
{
  return MPI_SUCCESS;
}


int MPIX_Testsome(int incount, MPIX_Request array_of_requests[],
                  int *outcount, int array_of_indices[],
                  MPI_Status array_of_statuses[])
{
  return MPI_SUCCESS;
}

int MPIX_Request_free(MPIX_Request *request)
{
  return MPI_SUCCESS;
}

