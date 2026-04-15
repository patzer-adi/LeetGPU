# ─── Makefile ────────────────────────────────────────────────────────────────
# Target: NVIDIA GTX 750 Ti  →  Compute Capability 5.0  (sm_50)
# Host compiler: g++-10
# CUDA toolkit: 11.x recommended (last with full sm_50 support)
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX         := g++-10

ARCH        := -gencode arch=compute_50,code=sm_50

NVCCFLAGS   := -std=c++14 $(ARCH) -O2 --compiler-bindir=$(CXX) -Xcompiler -Wall
CXXFLAGS    := -std=c++14 -O2 -Wall

TARGET      := matrix_addition_test
CUDA_OBJ    := matrix_addition.o
CPP_OBJ     := main.o

all: $(TARGET)

$(CUDA_OBJ): matrix_addition.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(CPP_OBJ): main.cpp
	$(CXX) $(CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@

$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(CUDA_OBJ) $(CPP_OBJ) $(TARGET)

.PHONY: all run clean
