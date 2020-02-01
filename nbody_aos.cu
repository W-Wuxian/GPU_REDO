#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define nPart 16384 
#define THREADS_PER_BLOCK 256
#define DUMP

//-------------------------------------------------------------------------------------------------------------------------

//struct ParticleType{
//  float x, y, z, vx, vy, vz;
//};

struct ParticleType{
   float x[nPart],y[nPart],z[nPart];
   float vx[nPart],vy[nPart],vz[nPart]; 
};

//-------------------------------------------------------------------------------------------------------------------------

//CUDA VERSION:
__global__ void MoveParticles_CUDA(const int nParticles, struct ParticleType* const particle){
  // Particle propagation time step
  const float dt = 0.0005f;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<nParticles){
      float Fx=0.,Fy=0.,Fz=0.;	
      for(int j = 0; j< nParticles; j++){
	//no self interaction:
	if(i != j){
	  //avoid singularity and interaction with self:
	  const float softening = 1e-20;
	  //Newton's law of universal gravity:
	  const float dx = particle->x[j] - particle->x[i];
	  const float dy = particle->y[j] - particle->y[i];
          const float dz = particle->z[j] - particle->z[i];
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
	  //Calculate the net force:
	  Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
	}//fi i!=j
      }//j loop
      //Accelerate particles in response to the gravitational force:
      particle->vx[i] += dt*Fx; 
      particle->vy[i] += dt*Fy; 
      particle->vz[i] += dt*Fz;
      }
}//fct MoveParticles_CUDA

//-------------------------------------------------------------------------------------------------------------------------

// Initialize random number generator and particles:
void init_rand(int nParticles, struct ParticleType* particle){
  srand48(0x2020);
  for (int i = 0; i < nParticles; i++)
  {
    particle->x[i] =  2.0*drand48() - 1.0;
    particle->y[i] =  2.0*drand48() - 1.0;
    particle->z[i] =  2.0*drand48() - 1.0;
    particle->vx[i] = 2.0*drand48() - 1.0;
    particle->vy[i] = 2.0*drand48() - 1.0;
    particle->vz[i] = 2.0*drand48() - 1.0;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

// Initialize (no random generator) particles
void init_norand(int nParticles, struct ParticleType* const particle){
  const float a=127.0/nParticles;
  for (int i = 0; i < nParticles; i++)
  {
    particle->x[i] =  i*a;//2.0*drand48() - 1.0;
    particle->y[i] =  i*a;//2.0*drand48() - 1.0;
    particle->z[i] =  1.0;//2.0*drand48() - 1.0;
    particle->vx[i] = 0.5;//2.0*drand48() - 1.0;
    particle->vy[i] = 0.5;//2.0*drand48() - 1.0;
    particle->vz[i] = 0.5;//2.0*drand48() - 1.0;
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
		particle->x[i], particle->y[i], particle->z[i], particle->vx[i], particle->vy[i], particle->vz[i]);
							      }

    fclose(f);
}
    
//-------------------------------------------------------------------------------------------------------------------------

int main(const int argc, const char** argv)
{
  // Problem size and other parameters
  // const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  const int nParticles=nPart;
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float ddt = 0.0005f;
//-------------------------------------------------------------------------------------------------------------------------
  //DEFINE SIZE:
  int SIZE = sizeof(struct ParticleType);
//-------------------------------------------------------------------------------------------------------------------------
  //DECLARATION & ALLOC particle ON HOST:
  struct ParticleType* particle = (struct ParticleType*) malloc(SIZE);
//-------------------------------------------------------------------------------------------------------------------------
  // Initialize random number generator and particles
  srand48(0x2020);
  //Initialize random number generator and particles
  //init_rand(nParticles, particle);
  // Initialize (no random generator) particles
  init_norand(nParticles, particle);
//-------------------------------------------------------------------------------------------------------------------------
  

  // Perform benchmark
  printf("\nPropagating %d particles using %d thread...\n\n", 
	  128 ,nParticles
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
//  cudaMemcpy(particle_cuda, particle, SIZE, cudaMemcpyHostToDevice);
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
	    particle->x[i]  += particle->vx[i]*ddt;
	    particle->y[i]  += particle->vy[i]*ddt;
	    particle->z[i]  += particle->vz[i]*ddt;
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
