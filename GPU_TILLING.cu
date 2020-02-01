#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#define SOFT 1e-20f

#define THREADS_PER_BLOCK 128
#define DUMP

typedef struct { 
	float4 *POS, *VIT; 
} PType;


//CUDA VERSION:
__global__ void VitParticles_CUDA(const int nParticles, float *p, float *v)
{
  // Particle propagation time step
    const float dt = 0.0005f;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<nParticles){
	float Fx = 0.0, Fy = 0.0, Fz = 0.0;
	for(int stack=0; stack<gridDim.x; stack++){
		__shared__ float3 p3[THREADS_PER_BLOCK];
		float4 tmp_p = p[stack*blockDim.x+threadIdx.x];
		p3[threadIdx.x] = make_float3(tmp_p.x,tmp_p.y,tmp_p.z);
		__syncthreads(); 
		for(int j = 0; j< THREADS_PER_BLOCK; j++){
	  		//avoid singularity and interaction with self:
	  		//const float softening = 1e-20;
	  		//Newton's law of universal gravity:
	  		float dx = p3[j].x - p[i].x;
	  		float dy = p3[j].y - p[i].y;
          		float dz = p3[j].z - p[i].z;
          		float drSquared    = dx*dx + dy*dy + dz*dz + SOFT;
			float invdrS       = rsqrtf(drSquared);
          		float invdrPower32 = invdrS*invdrS*invdrS;
       			  //Calculate the net force:
	  		Fx += dx * invdrPower32;  
          		Fy += dy * invdrPower32;  
          		Fz += dz / invdrPower32;
      		}//j loop
		__syncthreads();
	}	
      	//Accelerate particles in response to the gravitational force:
      	VIT[i].x += dt*Fx; 
      	VIT[i].y += dt*Fy; 
      	VIT[i].z += dt*Fz;
      }
}//fct MoveParticles_CUDA

//-------------------------------------------------------------------------------------------------------------------------

// Initialize random number generator and particles:
void init_rand(int ntol, float *tab){
  srand48(0x2020);
  for (int i = 0; i < ntol; i++)
  {
    tab[i].x =  2.0*drand48() - 1.0;
    tab[i].y =  2.0*drand48() - 1.0;
    tab[i].z =  2.0*drand48() - 1.0;
    tab[i].x =  2.0*drand48() - 1.0;
    tab[i].y =  2.0*drand48() - 1.0;
    tab[i].z =  2.0*drand48() - 1.0;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

// Initialize (no random generator) particles
void init_norand(int ntol, float *tab){
  const float a=127.0/nParticles;
  for (int i = 0; i < ntol; i++)
  {
    tab[i].x =  i*a;//2.0*drand48() - 1.0;
    tab[i].y =  i*a;//2.0*drand48() - 1.0;
    tab[i].z =  1.0;//2.0*drand48() - 1.0;
    tab[i].x =  0.5;//2.0*drand48() - 1.0;
    tab[i].y =  0.5;//2.0*drand48() - 1.0;
    tab[i].z =  0.5;//2.0*drand48() - 1.0;
  }
}


void init_sphere(int ntol, float *tab){
  const float a = 0.0f, b = 0.0f, c = 0.0f;
  const float r = 100.0f;
  tab[0].x =  0.0f;
  tab[0].y =  0.0f;
  tab[0].z =  r;
  tab[0].vx = 0.0f;
  tab[0].vy = 0.0f;
  tab[0].vz = -r;
	
  tab[1].x =  0.0f;
  tab[1].y =  0.0f;
  tab[1].z =  -r;
  tab[1].vx = 0.0f;
  tab[1].vy = 0.0f;
  tab[1].vz = r;
  for (int i = 2; i < 3289; i++)
  { float eta = 2*3.14159265359/3287;
    tab[i].x =  r*cos(eta);
    tab[i].y =  r*sin(eta);
    tab[i].z =  0.0f;
    tab[i].vx = -r*cos(eta);
    tab[i].vy = r*sin(eta);
    tab[i].vz = 0.0f;
  }
 for (int i = 3289; i < ntol; i++)
    float eta = 2*3.14159265359/13095.0;
    tab[i].x =  r*cos(eta);
    tab[i].y =  r*sin(eta);
    tab[i].z =  0.0f;
    tab[i].vx = -r*cos(eta);
    tab[i].vy = r*sin(eta);
    tab[i].vz = -99.0f+0.01504391f;
  }
}

//-------------------------------------------------------------------------------------------------------------------------

void dump(int iter, int nParticles, float *p, float *v)
{
    char filename[64];
        snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
        f = fopen(filename, "w+");

    int i;
        for (i = 0; i < nParticles; i++)
	    {
	      fprintf(f, "%e %e %e %e %e %e\n",
		p[i].x, p[i].y, p[i].z, v[i].x, v[i].y, v[i].z);
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
  int SIZE = 2*nParticles * sizeof(float4);
//-------------------------------------------------------------------------------------------------------------------------
  //DECLARATION & ALLOC particle ON HOST:
  float *evo = (float*) malloc(SIZE );
  struct PType pevo = {(float4*)evo,((float4*)evo)+nParticles};
//-------------------------------------------------------------------------------------------------------------------------
  // Initialize random number generator and particles
  //srand48(0x2020);
  // Initialize random number generator and particles
  //init_rand(nParticles, POS, VIT);
  // Initialize (no random generator) particles
  init_norand(8*nParticles, evo);
//-------------------------------------------------------------------------------------------------------------------------
  

  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );

  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
 int NBR_BLOCKS = (nParticles+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK:
//-------------------------------------------------------------------------------------------------------------------------
  cudaProfilerStart();
//-------------------------------------------------------------------------------------------------------------------------
  float *cuda_evo;
  cudaMalloc(&cuda_evo, SIZE);
  struct PType cuda_pevo = {(float4*)cuda_evo,((float4*)cuda_evo)+nParticles};
//-------------------------------------------------------------------------------------------------------------------------
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    cudaMemcpy(cuda_pevo, pevo, SIZE, cudaMemcpyHostToDevice);
    MoveParticles_CUDA<<<NBR_BLOCKS,THREADS_PER_BLOCK>>>(nParticles, cuda_pevo.POS, cuda_pevo.VIT);
    cudaMemcpy(pevo, cuda_pevo, SIZE, cudaMemcpyDeviceToHost);
    const double tEnd = omp_get_wtime(); // End timing

    // Move particles according to their velocities
      // O(N) work, so using a serial loop
        for (int i = 0 ; i < nParticles; i++) {
	    pevo.POS[i].x  += pevo.VIT[i].x*ddt;
	    pevo.POS[i].y  += pevo.VIT[i].y*ddt;
	    pevo.POS[i].z  += pevo.VIT[i].z*ddt;
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
    dump(step, nParticles, pevo.POS, pevo.VIT);
#endif
  }
  cudaFree(cuda_pevo);
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
  free(pevo);
  return 0;
}
