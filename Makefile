#C
#nbody: nbody.c
#	gcc -pg -g -fopenmp nbody.c -o nbody -lm


#nbody_cuda: nbody_cuda.cu
#	nvcc nbody_cuda.cu -Xcompiler -fopenmp -o nbody_cuda -lm -O3

nbody_cudaV2: nbody_cudaV2.cu
	nvcc nbody_cudaV2.cu -Xcompiler -fopenmp -o nbody_cudaV2 -lm -O3


#acc
#openacc: openacc.c
#	pgcc -pg openacc.c -o openacc -acc #-ta=tesla:cc70 -acc #-minfo=all



clean:
	rm -f nbody cuda openacc *.txt *~ *.out nbody_cuda nbody_cudaV2
