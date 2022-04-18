# DIRECTORIES
maindir=BPMF
libdir=$(maindir)/lib

# define compilers
NVCC=nvcc
CC=gcc

# define commands
#all: $(libdir)/libc.so $(libdir)/libcu.so
all: $(libdir)/libc.so
python_cpu: $(libdir)/libc.so
#python_gpu: $(libdir)/libcu.so
.SUFFIXES: .c .cu

# GPU FLAGS
COPTIMFLAGS_GPU=-O3
CFLAGS_GPU=-D_FORCE_INLINE -Xcompiler "-fopenmp -fPIC -march=native -ftree-vectorize" -Xlinker -lgomp
CARDDEPENDENTFLAG=-arch=sm_35
LDFLAGS_GPU=--shared

# CPU FLAGS
COPTIMFLAGS_CPU=-O3
CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native
LDFLAGS_CPU=-shared

# build for python
#$(libdir)/libcu.so: $(maindir)/libcu.cu
#	$(NVCC) $(COPTIMFLAGS_GPU) $(CFLAGS_GPU) $(CARDDEPENDENTFLAG) $(LDFLAGS_GPU) $< -o $@

$(libdir)/libc.so: $(maindir)/libc.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

clean:
	rm $(libdir)/*.so

