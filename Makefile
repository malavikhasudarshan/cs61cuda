NVCC = nvcc
CFLAGS = -O2 -std=c++11

SRC = src/main.cu src/utils.cu src/cpu_matmul.cu src/cuda_naive.cu src/cuda_simd.cu
OBJ = $(SRC:.cu=.o)
TARGET = cs61cuda

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)