#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define THREADS_PER_BLOCK 128
#define DUMP

//-------------------------------------------------------------------------------------------------------------------------

struct PType{
  float x,y,z;
};

//-------------------------------------------------------------------------------------------------------------------------

//CUDA VERSION:
__global__ void MoveParticles_CUDA(const int nParticles, struct PType* const POS, struct PType* const VIT){
  // Particle propagation time step
  const float dt = 0.0005f;
  float Fx = 0.0, Fy = 0.0, Fz = 0.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<nParticles){
      for(int j = 0; j< nParticles; j++){
	//no self interaction:
	if(i != j){
	  //avoid singularity and interaction with self:
	  const float softening = 1e-20;
	  //Newton's law of universal gravity:
	  const float dx = POS[j].x - POS[i].x;
	  const float dy = POS[j].y - POS[i].y;
          const float dz = POS[j].z - POS[i].z;
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
	  //Calculate the net force:
	  Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
	}//fi i!=j
      }//j loop
      //Accelerate particles in response to the gravitational force:
      VIT[i].x += dt*Fx; 
      VIT[i].y += dt*Fy; 
      VIT[i].z += dt*Fz;
      }
}//fct MoveParticles_CUDA

//-------------------------------------------------------------------------------------------------------------------------

// Initialize random number generator and particles:
void init_rand(int nParticles, struct PType* POS, struct PType* VIT){
  srand48(0x2020);
  for (int i = 0; i < nParticles; i++)
  {
    POS[i].x =  2.0*drand48() - 1.0;
    POS[i].y =  2.0*drand48() - 1.0;
    POS[i].z =  2.0*drand48() - 1.0;
    VIT[i].x = 2.0*drand48() - 1.0;
    VIT[i].y = 2.0*drand48() - 1.0;
    VIT[i].z = 2.0*drand48() - 1.0;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

// Initialize (no random generator) particles
void init_norand(int nParticles, struct PType* POS, struct PType* VIT){
  const float a=127.0/nParticles;
  for (int i = 0; i < nParticles; i++)
  {
    POS[i].x =  i*a;//2.0*drand48() - 1.0;
    POS[i].y =  i*a;//2.0*drand48() - 1.0;
    POS[i].z =  1.0;//2.0*drand48() - 1.0;
    VIT[i].x = 0.5;//2.0*drand48() - 1.0;
    VIT[i].y = 0.5;//2.0*drand48() - 1.0;
    VIT[i].z = 0.5;//2.0*drand48() - 1.0;
  }
}


void init_sphere(int nParticles, struct ParticleType* particle){
  const float a = 0.0f, b = 0.0f, c = 0.0f;
  const float r = 100.0f;
  particle[0].x =  0.0f;
  particle[0].y =  0.0f;
  particle[0].z =  r;
  particle[0].vx = 0.0f;
  particle[0].vy = 0.0f;
  particle[0].vz = -r;
	
  particle[1].x =  0.0f;
  particle[1].y =  0.0f;
  particle[1].z =  -r;
  particle[1].vx = 0.0f;
  particle[1].vy = 0.0f;
  particle[1].vz = r;
  for (int i = 2; i < 3289; i++)
  { float eta = 2*3.14159265359/3287;
    particle[i].x =  r*cos(eta);
    particle[i].y =  r*sin(eta);
    particle[i].z =  0.0f;
    particle[i].vx = -r*cos(eta);
    particle[i].vy = r*sin(eta);
    particle[i].vz = 0.0f;
  }
 for (int i = 3289; i < nParticles; i++)
    float eta = 2*3.14159265359/13095.0;
    particle[i].x =  r*cos(eta);
    particle[i].y =  r*sin(eta);
    particle[i].z =  0.0f;
    particle[i].vx = -r*cos(eta);
    particle[i].vy = r*sin(eta);
    particle[i].vz = -99.0f+0.01504391f;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

void dump(int iter, int nParticles, struct PType* POS, struct PType* VIT)
{
    char filename[64];
        snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
        f = fopen(filename, "w+");

    int i;
        for (i = 0; i < nParticles; i++)
	    {
	      fprintf(f, "%e %e %e %e %e %e\n",
		POS[i].x, POS[i].y, POS[i].z, VIT[i].x, VIT[i].y, VIT[i].z);
							      }

    fclose(f);
}
    
//-------------------------------------------------------------------------------------------------------------------------

int main(const int argc, const char** argv)
{
  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float ddt = 0.0005f;
//-------------------------------------------------------------------------------------------------------------------------
  //DEFINE SIZE:
  int SIZE = nParticles * sizeof(struct PType);
//-------------------------------------------------------------------------------------------------------------------------
  //DECLARATION & ALLOC particle ON HOST:
  struct PType* POS = (struct PType*) malloc(SIZE);
  struct PType* VIT = (struct PType*) malloc(SIZE);
//-------------------------------------------------------------------------------------------------------------------------
  // Initialize random number generator and particles
  srand48(0x2020);
  // Initialize random number generator and particles
  //init_rand(nParticles, POS, VIT);
  // Initialize (no random generator) particles
  init_norand(nParticles, POS, VIT);
//-------------------------------------------------------------------------------------------------------------------------
  

  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );

  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
//-------------------------------------------------------------------------------------------------------------------------
  cudaProfilerStart();
//-------------------------------------------------------------------------------------------------------------------------
  struct PType *POS_cuda;
  cudaMalloc((void **)&POS_cuda, SIZE);
  //cudaMemcpy(POS_cuda, POS, SIZE, cudaMemcpyHostToDevice);
  //-------------------------------------------------------------------------------------------------------------------------
  struct PType *VIT_cuda;
  cudaMalloc((void **)&VIT_cuda, SIZE);
  //cudaMemcpy(VIT_cuda, VIT, SIZE, cudaMemcpyHostToDevice);
//-------------------------------------------------------------------------------------------------------------------------
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    cudaMemcpy(POS_cuda, POS, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(VIT_cuda, VIT, SIZE, cudaMemcpyHostToDevice);
    MoveParticles_CUDA<<<nParticles/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(nParticles, POS_cuda, VIT_cuda);
    cudaMemcpy(POS, POS_cuda, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(VIT, VIT_cuda, SIZE, cudaMemcpyDeviceToHost);
    const double tEnd = omp_get_wtime(); // End timing

    // Move particles according to their velocities
      // O(N) work, so using a serial loop
        for (int i = 0 ; i < nParticles; i++) {
	    POS[i].x  += VIT[i].x*ddt;
	    POS[i].y  += VIT[i].y*ddt;
	    POS[i].z  += VIT[i].z*ddt;
	}
		      

    const float HztoInts   = ((float)nParticles)*((float)(nParticles-1)) ;
    const float HztoGFLOPs = 20.0*1e-9*((float)(nParticles))*((float)(nParticles-1));

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoInts/(tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);

#ifdef DUMP
    dump(step, nParticles, POS, VIT);
#endif
  }
  cudaFree(POS_cuda);
  cudaFree(VIT_cuda);
//-------------------------------------------------------------------------------------------------------------------------
  cudaProfilerStop();
//-------------------------------------------------------------------------------------------------------------------------
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  free(POS);
  free(VIT);
  return 0;
}
