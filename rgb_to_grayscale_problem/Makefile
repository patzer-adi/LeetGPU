# ─────────────────────────────────────────────────────────────────────────────
# Makefile — RGB to Grayscale CUDA Solution
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CUDA_ARCH   := sm_75          # RTX 2080 (SM 7.5) — change for your GPU:
                               #   RTX 3080/3090 → sm_86
                               #   RTX 4090      → sm_89
                               #   A100          → sm_80
NVCCFLAGS   := -arch=$(CUDA_ARCH) -O2 -std=c++17 --use_fast_math

SRC         := solution.cu
LIB_TARGET  := grayscale.so   # shared library (for the judge)
TEST_TARGET := grayscale_test  # standalone test binary

.PHONY: all lib test clean

all: lib

# ── Shared library (what the LeetGPU judge loads) ────────────────────────────
lib: $(LIB_TARGET)

$(LIB_TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) --compiler-options '-fPIC' -shared -o $@ $<

# ── Test binary (validates both examples + 2048×2048 stress) ─────────────────
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DTEST_MAIN -o $@ $<

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(LIB_TARGET) $(TEST_TARGET)
