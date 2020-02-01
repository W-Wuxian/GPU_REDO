#C
nbody: nbody.c
	gcc -pg -g -fopenmp nbody.c -o nbody -lm


#nbody_omp: nbody_omp.c
#	gcc -pg -g -fopenmp nbody_omp.c -o nbody_omp -lm

#nbody_cuda: nbody_cuda.cu
#	nvcc nbody_cuda.cu -Xcompiler -fopenmp -o nbody_cuda -lm -O3

#nbody_cudaV2: nbody_cudaV2.cu
#	nvcc nbody_cudaV2.cu -Xcompiler -fopenmp -o nbody_cudaV2 -lm -O3

#nbody_aos.c: nbody_aos.c
#	nvcc nbody_aos.cu -Xcompiler -fopenmp -o nbody_aos -lm -O3

#GPU_TILLING.cu: GPU_TILLING.cu
#	nvcc GPU_TILLING.cu -Xcompiler -fopenmp -o GPU_TILLING -lm -O3

#acc
#openacc: openacc.c
#	pgcc -pg openacc.c -o openacc -acc #-ta=tesla:cc70 -acc #-minfo=all

#cc70 car RTX 2070 <=> cc70  (7.0X) cuda doc



clean:
	rm -f nbody cuda openacc *.txt *~ *.out nbody_cuda nbody_cudaV2 nbody_aos nbody_omp GPU_TILLING
