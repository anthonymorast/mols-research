CC=g++
NVCC=nvcc
MPICC=mpic++
CUDAFLAGS=-O3 --std=c++11 -I.
CFLAGS= -O3 --std=c++11 -I. -Wall
DEPS= LatinSquare.h
UTIL=utilities
GENSER=generate_squares_serial
GENMPI=generate_squares_mpi
GENCUDA=generate_squares_cuda
EXES=$(UTIL) $(GENSER) $(GENMPI) $(GENCUDA)

all: $(UTIL) $(GENSER) $(GENMPI) $(GENCUDA)

obj/ls/%.o: ../LatinSquare/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

obj/utils/%.o: utils/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

obj/serial/%.o: serial/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

obj/mpi/%.o: mpi/%.cpp
	$(MPICC) -c -o $@ $< $(CFLAGS)

obj/cuda/%.o: cuda/%.cu
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

$(UTIL): obj/utils/Utils.o
	$(CC) -o $(UTIL) obj/utils/Utils.o

$(GENSER): obj/ls/LatinSquare.o obj/serial/GenerateSquares.o
	$(CC) -o $(GENSER) obj/ls/LatinSquare.o obj/serial/GenerateSquares.o

$(GENMPI): obj/ls/LatinSquare.o obj/mpi/GenerateSquares.o
	$(MPICC) -o $(GENMPI) obj/ls/LatinSquare.o obj/mpi/GenerateSquares.o

$(GENCUDA): obj/ls/LatinSquare.o obj/cuda/GenerateSquares.o
	$(NVCC) -o $(GENCUDA) obj/ls/LatinSquare.o obj/cuda/GenerateSquares.o

#generate_squares_mpi: obj/LatinSquare.o obj/GenerateSquaresMPI.o
#	$(MPICC) -o $(GENSQPTH) obj/LatinSquare.o obj/GenerateSquaresMPI.o

#obj/%.o: %.cu
#	$(NVCC) -c -o $@ $< $(CFLAGS)

#generate_squares_cuda: obj/LatinSquare.o obj/GenerateSquaresCuda.o
#	$(NVCC) -o $(GENCUDA) obj/LatinSquare.o obj obj/GenerateSquaresCuda.o

.PHONY: clean
clean:
	rm obj/ls/*.o obj/utils/*.o obj/serial/*.o obj/mpi/*.o obj/cuda/*.o $(EXES)
