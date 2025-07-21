# DIRECTORIES
maindir=BPMF
libdir=$(maindir)/lib

# define compilers
NVCC=nvcc

# -------------------------------------------------
# Unix system
#CC=gcc
# -------------------------------------------------
# Apple Silicon chip:
CC=clang
# -------------------------------------------------

# define commands
all: $(libdir)/libc.so
python_cpu: $(libdir)/libc.so
.SUFFIXES: .c .cu

# CPU FLAGS
COPTIMFLAGS_CPU=-O3

# -------------------------------------------------
# Unix system
#CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native -std=c99
#LDFLAGS_CPU=-shared
# -------------------------------------------------


# -------------------------------------------------
# Apple Silicon chip
CFLAGS_CPU=-fopenmp=libomp -L$(CONDA_PREFIX)/lib -I$(CONDA_PREFIX)/include -fPIC -ftree-vectorize -march=native -std=c99
LDFLAGS_CPU=-shared -fuse-ld=lld
# -------------------------------------------------

# build for python
$(libdir)/libc.so: $(maindir)/libc.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

clean:
	rm $(libdir)/*.so

