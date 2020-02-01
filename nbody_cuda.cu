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

struct ParticleType{
  float x, y, z, vx, vy, vz;
};

//-------------------------------------------------------------------------------------------------------------------------

//CUDA VERSION:
__global__ void MoveParticles_CUDA(const int nParticles, struct ParticleType* const particle){
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
	  const float dx = particle[j].x - particle[i].x;
	  const float dy = particle[j].y - particle[i].y;
          const float dz = particle[j].z - particle[i].z;
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
	  //Calculate the net force:
	  Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
	}//fi i!=j
      }//j loop
      //Accelerate particles in response to the gravitational force:
      particle[i].vx += dt*Fx; 
      particle[i].vy += dt*Fy; 
      particle[i].vz += dt*Fz;
      }
}//fct MoveParticles_CUDA

//-------------------------------------------------------------------------------------------------------------------------

// Initialize random number generator and particles:
void init_rand(int nParticles, struct ParticleType* particle){
  srand48(0x2020);
  for (int i = 0; i < nParticles; i++)
  {
    particle[i].x =  2.0*drand48() - 1.0;
    particle[i].y =  2.0*drand48() - 1.0;
    particle[i].z =  2.0*drand48() - 1.0;
    particle[i].vx = 2.0*drand48() - 1.0;
    particle[i].vy = 2.0*drand48() - 1.0;
    particle[i].vz = 2.0*drand48() - 1.0;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

// Initialize (no random generator) particles
void init_norand(int nParticles, struct ParticleType* particle){
  const float a=127.0/nParticles;
  for (int i = 0; i < nParticles; i++)
  {
    particle[i].x =  i*a;//2.0*drand48() - 1.0;
    particle[i].y =  i*a;//2.0*drand48() - 1.0;
    particle[i].z =  1.0;//2.0*drand48() - 1.0;
    particle[i].vx = 0.5;//2.0*drand48() - 1.0;
    particle[i].vy = 0.5;//2.0*drand48() - 1.0;
    particle[i].vz = 0.5;//2.0*drand48() - 1.0;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

void dump(int iter, int nParticles, struct ParticleType* particle)
{
    char filename[64];
        snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
        f = fopen(filename, "w+");

    int i;
        for (i = 0; i < nParticles; i++)
	    {
	      fprintf(f, "%e %e %e %e %e %e\n",
		particle[i].x, particle[i].y, particle[i].z, particle[i].vx, particle[i].vy, particle[i].vz);
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
  int SIZE = nParticles * sizeof(struct ParticleType);
//-------------------------------------------------------------------------------------------------------------------------
  //DECLARATION & ALLOC particle ON HOST:
  struct ParticleType* particle = (struct ParticleType*) malloc(SIZE);
//-------------------------------------------------------------------------------------------------------------------------
  // Initialize random number generator and particles
  srand48(0x2020);
  // Initialize random number generator and particles
  //init_rand(nParticles, particle);
  // Initialize (no random generator) particles
  init_norand(nParticles, particle);
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
  struct ParticleType *particle_cuda;
  cudaMalloc((void **)&particle_cuda, SIZE);
//-------------------------------------------------------------------------------------------------------------------------
  cudaMemcpy(particle_cuda, particle, SIZE, cudaMemcpyHostToDevice);
//-------------------------------------------------------------------------------------------------------------------------
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    cudaMemcpy(particle_cuda, particle, SIZE, cudaMemcpyHostToDevice);
    MoveParticles_CUDA<<<nParticles/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(nParticles, particle_cuda);
    cudaMemcpy(particle, particle_cuda, SIZE, cudaMemcpyDeviceToHost);//need to do?
    const double tEnd = omp_get_wtime(); // End timing

    // Move particles according to their velocities
      // O(N) work, so using a serial loop
        for (int i = 0 ; i < nParticles; i++) {
	    particle[i].x  += particle[i].vx*ddt;
	    particle[i].y  += particle[i].vy*ddt;
	    particle[i].z  += particle[i].vz*ddt;
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
    dump(step, nParticles, particle);
#endif
  }
  cudaFree(particle_cuda);
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
  free(particle);
  return 0;
}
