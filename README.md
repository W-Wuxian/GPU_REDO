# CUDA NBODY SIMULATION
* Dans le Makefile modifier (commenter/decommenter) selon votre choix.
* Liste des codes
- nbody.c
  + sequentiel
- nbody_omp.c
  + openmp version
- openacc
  + openacc version
- nbody_cuda.cu 
  + version naive cuda
- nbody_cudaV2.cu 
  + split vitesse et position cuda
- nbody_aos.cu 
  + devrait s'appeler nbody_soa.cu car version SoA cuda
- GPU_TILLING 
  + Tilling (inspiration prod MAT/MAT) cuda + SoA + Coallescing


