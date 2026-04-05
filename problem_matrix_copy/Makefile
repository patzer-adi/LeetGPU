# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Matrix Copy CUDA Solution
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CUDA_ARCH   := sm_75          # RTX 2080 (SM 7.5) — change for your GPU:
                               #   RTX 3080/3090 → sm_86
                               #   RTX 4090      → sm_89
                               #   A100          → sm_80
NVCCFLAGS   := -arch=$(CUDA_ARCH) -O2 -std=c++17

SRC         := solution.cu
LIB_TARGET  := matrix_copy.so
TEST_TARGET := matrix_copy_test

.PHONY: all lib test clean

all: lib

# ── Shared library (for the judge) ───────────────────────────────────────────
lib: $(LIB_TARGET)

$(LIB_TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) --compiler-options '-fPIC' -shared -o $@ $<

# ── Test binary ───────────────────────────────────────────────────────────────
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DTEST_MAIN -o $@ $<

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(LIB_TARGET) $(TEST_TARGET)
