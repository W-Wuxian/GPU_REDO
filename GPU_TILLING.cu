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
#define BLOCK_SIZE 256


typedef struct { float4 *pos, *vit; } PType;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void dump(int iter, int nParticles, PType pv)
{
    char filename[64];
        snprintf(filename, 64, "output_%d.txt", iter);
    FILE *f;
        f = fopen(filename, "w+");
    int i;
        for (i = 0; i < nParticles; i++)
	    {
	      fprintf(f, "%e %e %e %e %e %e\n",
		pv.POS[i].x, pv.POS[i].y, pv.POS[i].z, pv.VIT[i].x, pv.VIT[i].y, pv.VIT[i].z);
							      }
    fclose(f);
}


__global__
void bodyForce(float4 *p, float4 *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float3 p3[BLOCK_SIZE];
      float4 tmp_pos = p[tile * blockDim.x + threadIdx.x];
      p3[threadIdx.x] = make_float3(tmp_pos.x, tmp_pos.y, tmp_pos.z);
      __syncthreads();

      for (int j = 0; j < BLOCK_SIZE; j++) {
        float dx = p3[j].x - p[i].x;
        float dy = p3[j].y - p[i].y;
        float dz = p3[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
      __syncthreads();
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  int nBodies = 16384;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.00005f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = 2*nBodies*sizeof(float4);
  float *buf = (float*)malloc(bytes);
  PType p = { (float4*)buf, ((float4*)buf) + nBodies };

  randomizeBodies(buf, 8*nBodies); // Init pos / vit data

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  PType d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0;
	// Perform benchmark
	  printf("\nPropagating %d particles using 1 thread...\n\n",
		 nParticles
	);
	double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
	printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
	cudaProfilerStart();
  for (int iter = 1; iter <= nIters; iter++) {
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vit, dt, nBodies);
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vit[i].x*dt;
      p.pos[i].y += p.vit[i].y*dt;
      p.pos[i].z += p.vit[i].z*dt;
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
		dump(iter, nParticles, p);
	}
  cudaProfilerStop();
  free(buf);
  cudaFree(d_buf);
	//-------------------------------------------------------------------------------------------------------------------------
  rate/=(double)(nSteps-skipSteps);
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
}
