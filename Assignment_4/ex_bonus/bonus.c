// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define NUM_PARTICLES 100000
#define DIM 3


typedef struct  particle {
    cl_float pos_x;
    cl_float pos_y;
    cl_float pos_z;
    cl_float v_x;
    cl_float v_y;
    cl_float v_z;
} Particle;

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);


const char *mykernel = "\
  typedef struct{\
    float pos_x;\
    float pos_y;\
    float pos_z;\
    float v_x;\
    float v_y;\
    float v_z;\
  } Particle;\
  __kernel\
  void update_particles(\
    __global Particle* particles,\
    int size\
  ) {\
    int i = get_global_id(0);\
    if(i >= size) return;\
    particles[i].v_x = particles[i].pos_x / 10;\
    particles[i].v_y = particles[i].pos_y / 10;\
    particles[i].v_z = particles[i].pos_z / 10;\
    particles[i].pos_x += particles[i].v_x;\
    particles[i].pos_y += particles[i].v_y;\
    particles[i].pos_z += particles[i].v_z;\
  }"; 

void cpu_update_particles(Particle* particles, int NUM_ITERATIONS) {
    for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int i = 0; i < NUM_PARTICLES; i++) {
            particles[i].v_x = particles[i].pos_x / 10;
            particles[i].v_y = particles[i].pos_y / 10;
            particles[i].v_z = particles[i].pos_z / 10;
            particles[i].pos_x += particles[i].v_x;
            particles[i].pos_y += particles[i].v_y;
            particles[i].pos_z += particles[i].v_z;
        }
    }
}
  
void generatePositions(Particle* particles_cpu, Particle* particles_gpu) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles_cpu[i].pos_x = (cl_float)rand()/RAND_MAX;
        particles_cpu[i].pos_y = (cl_float)rand()/RAND_MAX;
        particles_cpu[i].pos_z = (cl_float)rand()/RAND_MAX;
        particles_gpu[i].pos_x = particles_cpu[i].pos_x;
        particles_gpu[i].pos_y = particles_cpu[i].pos_y;
        particles_gpu[i].pos_z = particles_cpu[i].pos_z;
    }
}

void compare_results(Particle* particles_gpu, Particle* particles_cpu) {
  int i;
  for (i = 0; i < NUM_PARTICLES; i++) {
        printf("%f %f\n", particles_cpu[i].pos_x, particles_gpu[i].pos_x);
        if (abs(particles_cpu[i].pos_x - particles_gpu[i].pos_x) > 0.0001
            || abs(particles_cpu[i].pos_y - particles_gpu[i].pos_y) > 0.0001
            || abs(particles_cpu[i].pos_z - particles_gpu[i].pos_z) > 0.0001) break;
    }

  if (i == NUM_PARTICLES) printf("Comparing the output for each implementation... Correct!\n"); 
  else printf("Comparing the output for each implementation... Incorrect :(\n");
}

int main(int argc, char *argv[]) {
  printf("Starting the program...\n");
  int BLOCK_SIZE = 32;
  int NUM_ITERATIONS = 10;

  // read BLOCK_SIZE from terminal
  if (argc > 1) BLOCK_SIZE = atoi(argv[1]);
  if (argc > 2) NUM_ITERATIONS = atoi(argv[2]);

  cl_platform_id * platforms; cl_uint     n_platform;

  // Find OpenCL Platforms
  cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
  err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

  // Find and sort devices
  cl_device_id *device_list; cl_uint n_devices;
  err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
  err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);
  
  // Create and initialize an OpenCL context
  cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

  // Create a command queue
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err); 

  /* Insert your own code here */

  // initialise on GPU
  int array_size = NUM_PARTICLES * sizeof(Particle);


  cl_mem particles_dev = clCreateBuffer (context, CL_MEM_READ_WRITE, array_size, NULL, &err); 

  // initialize on CPU
  Particle* particles_cpu = (Particle *)malloc(array_size);
  Particle* particles_gpu = (Particle *)malloc(array_size);

  generatePositions(particles_cpu, particles_gpu);

  // compute on GPU

  err = clEnqueueWriteBuffer (cmd_queue, particles_dev, CL_TRUE, 0, array_size, particles_gpu, 0, NULL, NULL); CHK_ERROR(err);

  cl_program update_particles = clCreateProgramWithSource(context, 1, (const char **)&mykernel, NULL, &err); CHK_ERROR(err);
  err = clBuildProgram(update_particles, 1, device_list, NULL, NULL, NULL); CHK_ERROR(err);

  cl_kernel kernel = clCreateKernel(update_particles, "update_particles", &err); CHK_ERROR(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &particles_dev); CHK_ERROR(err);

  size_t n_workitem = (int)(NUM_PARTICLES / BLOCK_SIZE + 1) * BLOCK_SIZE;
  size_t workgroup_size = BLOCK_SIZE;

  // Launch the kernel!
  err = clEnqueueNDRangeKernel (cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL); CHK_ERROR(err);
  
  // Transfer the data from C back
  err = clEnqueueReadBuffer (cmd_queue, particles_dev, CL_TRUE, 0, array_size, particles_gpu, 0, NULL, NULL); CHK_ERROR(err);
  err = clFlush(cmd_queue); CHK_ERROR(err); 
  err = clFinish(cmd_queue); CHK_ERROR(err);

  printf("Computing update_particles on the GPU... Done!\n\n");

  // check the result with CPU
  cpu_update_particles(particles_cpu, NUM_ITERATIONS);
  printf("Computing update_particles on the CPU... Done!\n\n");

  compare_results(particles_gpu, particles_cpu);

  free(particles_cpu);
  
  // Finally, release all that we have allocated.
  err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
  err = clReleaseContext(context); CHK_ERROR(err);
  free(platforms);
  free(device_list);
  
  return 0;
}



// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}
