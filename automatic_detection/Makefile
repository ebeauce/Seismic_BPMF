all: libc.so libcu.so
python_cpu: libc.so
python_gpu: libcu.so 
.SUFFIXES: .c .cu

NVCC=nvcc
CC=gcc

# GPU FLAGS
COPTIMFLAGS_GPU=-O3
CFLAGS_GPU=-D_FORCE_INLINE -Xcompiler "-fopenmp -fPIC -march=native -ftree-vectorize" -Xlinker -lgomp
CARDDEPENDENTFLAG=-arch=sm_35
LDFLAGS_GPU=--shared

# CPU FLAGS
COPTIMFLAGS_CPU=-O3
CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native
LDFLAGS_CPU=-shared

# MEX FLAGS
COPTIMFLAGS_MEX=-O3
CFLAGS_MEX=-fopenmp -fPIC -march=native
 # who knows why mex needs fopenmp again
LDFLAGS_MEX=-fopenmp -shared

# build for python
libcu.so: libcu.cu
	$(NVCC) $(COPTIMFLAGS_GPU) $(CFLAGS_GPU) $(CARDDEPENDENTFLAG) $(LDFLAGS_GPU) $< -o $@

libc.so: libc.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

clean:
	rm ./*.so

