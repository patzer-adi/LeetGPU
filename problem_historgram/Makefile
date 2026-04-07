# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Histogram CUDA Solution
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CUDA_ARCH   := sm_75          # RTX 2080 — change if needed:
                               #   RTX 3080/3090 → sm_86
                               #   RTX 4090      → sm_89
                               #   A100          → sm_80
NVCCFLAGS   := -arch=$(CUDA_ARCH) -O2 -std=c++17

SRC         := solution.cu
LIB_TARGET  := histogram.so
TEST_TARGET := histogram_test

.PHONY: all lib test clean

all: lib

lib: $(LIB_TARGET)
$(LIB_TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) --compiler-options '-fPIC' -shared -o $@ $<

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DTEST_MAIN -o $@ $<

clean:
	rm -f $(LIB_TARGET) $(TEST_TARGET)
