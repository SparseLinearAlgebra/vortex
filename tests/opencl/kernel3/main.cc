#include <CL/opencl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "common.h"

int M = 256, N = 256, K = 256;

static void show_usage() {
  printf("Usage: [-M number of rows in first matrix] [-N number of columns in first matrix] [-K number of columns in first matrix and rows in second matrix] [-h: help]\n");
}

#define CL_CHECK(_expr)                                                        \
  do {                                                                         \
    cl_int _err = _expr;                                                       \
    if (_err == CL_SUCCESS)                                                    \
      break;                                                                   \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);            \
    cleanup();                                                                 \
    exit(-1);                                                                  \
  } while (0)

#define CL_CHECK2(_expr)                                                       \
  ({                                                                           \
    cl_int _err = CL_INVALID_VALUE;                                            \
    decltype(_expr) _ret = _expr;                                              \
    if (_err != CL_SUCCESS) {                                                  \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);          \
      cleanup();                                                               \
      exit(-1);                                                                \
    }                                                                          \
    _ret;                                                                      \
  })

static int read_kernel_file(const char *filename, uint8_t **data,
                            size_t *size) {
  if (NULL == filename || NULL == data || 0 == size)
    return -1;

  FILE *fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }

  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t *)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

static void sgemm_cpu(float *C, const float *A, const float *B, int M, int N,
                      int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += A[k * M + m] * B[n * K + k];
      }
      C[n * M + m] = acc;
    }
  }
}

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem c_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (command_queue)
    clReleaseCommandQueue(command_queue);
  if (kernel)
    clReleaseKernel(kernel);
  if (program)
    clReleaseProgram(program);
  if (a_memobj)
    clReleaseMemObject(a_memobj);
  if (b_memobj)
    clReleaseMemObject(b_memobj);
  if (c_memobj)
    clReleaseMemObject(c_memobj);
  if (context)
    clReleaseContext(context);
  if (device_id)
    clReleaseDevice(device_id);
  if (kernel_bin)
    free(kernel_bin);
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:M:N:K:h?")) != -1) {
    switch (c) {
    case 'n':
      M = N = K = atoi(optarg);
      break;
    case 'M':
      M = atoi(optarg);
      break;
    case 'N':
      N = atoi(optarg);
      break;
    case 'K':
      K = atoi(optarg);
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }

  if (M < 2 || N < 2 || K < 2) {
    printf("Error: invalid size!\n");
    exit(-1);
  }
}

int main(int argc, char **argv) {
  // parse command arguments
  parse_args(argc, argv);

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  // create context
  cl_context_properties context_properties[]{
      CL_CONTEXT_PLATFORM, cl_context_properties(platform_id), 0};
  cl_device_id devices[]{device_id};
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);
  
  // create command queue
  command_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // generate input data
  float *A, *B, *C;
  A = (float *)(malloc(M * K * sizeof(float)));
  B = (float *)(malloc(N * K * sizeof(float)));
  C = (float *)(malloc(M * N * sizeof(float)));
  if (A == NULL || B == NULL || C == NULL) {
    printf("Not enough memory");
    cleanup();
    return -1;
  }
  srand(time(NULL));
  for (int i = 0; i < M * K; i++)
    A[i] = (int)((float)rand() / (float)RAND_MAX);
  for (int i = 0; i < N * K; i++)
    B[i] = (int)((float)rand() / (float)RAND_MAX);

  // create buffers
  a_memobj =
      CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     M * K * sizeof(float), A, &_err));
  b_memobj =
      CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     N * K * sizeof(float), B, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   M * N * sizeof(float), NULL, &_err));

  // load kernel text
  size_t kernel_size;
  if (read_kernel_file("kernel.cl", &kernel_bin, &kernel_size) != 0) {
    cleanup();
    return -1;
  }
  program = CL_CHECK2(clCreateProgramWithSource(context, 1, (const char **)&kernel_bin,
                                      &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }

  // build program
  cl_int build_status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  // create kernel
  kernel = CL_CHECK2(clCreateKernel(program, "myGEMM3", &_err));

  // check building info
  size_t log_size = 0;
  CL_CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0,
                                 NULL, &log_size));
  char *log = (char *)malloc(log_size * sizeof(char));
  if (log == NULL) {
    printf("Not enough memory");
    cleanup();
    return -1;
  }
  CL_CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                 log_size, log, NULL));
  if (log_size > 1) {
    printf("Log:\n");
    printf("%s", log);
    printf("\n");
  }
  CL_CHECK(build_status);


  // set kernel arguments
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), &M));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &N));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &K));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&c_memobj));

  // run kernel
  const size_t local[2] = {TS, TS / WPT};
  const size_t global[2] = {M, N / WPT};
  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(command_queue));
  //CL_CHECK(clWaitForEvents(1, &event));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       time_end - time_start)
                       .count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // get results from VRAM
  CL_CHECK(clEnqueueReadBuffer(command_queue, c_memobj, CL_TRUE, 0,
                               M * N * sizeof(float), C, 0, NULL, NULL));
  CL_CHECK(clFinish(command_queue));

  // verify results
  printf("Verify result\n");
  float *C_cpu = (float *)malloc(M * N * sizeof(float));
  if (C_cpu == NULL) {
    printf("Not enough memory");
    cleanup();
    return -1;
  }
  sgemm_cpu(C_cpu, A, B, M, N, K);
  int errors = 0;

  for (size_t i = 0; i < size_t(M * N); i++)
    if (C_cpu[i] != C[i])
      errors++;
  if (errors != 0)
    printf("FAILED! - %d errors\n", errors);
  else
    printf("PASSED!\n");

  // free resureses
  cleanup();
  free(A);
  free(B);
  free(C);
  free(log);
  free(C_cpu);
  return errors;
}
