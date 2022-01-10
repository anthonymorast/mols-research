CC=g++
NVCC=nvcc
MPICC=mpic++
CUDAFLAGS=-O3 --std=c++11 -I. -lpthread
CFLAGS= -O3 --std=c++11 -I. -Wall -lpthread
DEPS= LatinSquare.h
GENSER=generate_squares_serial
GENMPI=generate_squares_mpi
GENCUDA=generate_squares_cuda
UTILS=utilities
CHECKSER=check_squares_serial
DIR_GUARD=@mkdir -p $(@D)
EXES=$(GENSER) $(GENMPI) $(GENCUDA) $(CHECKSER) $(UTILS)

all: $(GENSER) $(GENMPI) $(GENCUDA) $(CHECKSER) $(UTILS)

obj/ls/%.o: ../LatinSquare/%.cpp
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -c -o $@ $< 

obj/utils/%.o: utils/%.cpp
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -c -o $@ $<

obj/serial/%.o: serial/%.cpp
	$(DIR_GUARD)
	$(CC) -c -o $@ $< $(CFLAGS)

obj/mpi/%.o: mpi/%.cpp
	$(DIR_GUARD)
	$(MPICC) -c -o $@ $< $(CFLAGS)

obj/cuda/%.o: cuda/%.cu
	$(DIR_GUARD)
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

$(UTIL): obj/utils/Utils.o
	$(CC) -o $(UTIL) obj/utils/Utils.o

$(GENSER): obj/ls/LatinSquare.o obj/serial/GenerateSquares.o obj/utils/FileHandler.o
	$(CC) -o $(GENSER) obj/ls/LatinSquare.o obj/serial/GenerateSquares.o

$(GENMPI): obj/ls/LatinSquare.o obj/mpi/GenerateSquares.o obj/utils/FileHandler.o
	$(MPICC) -o $(GENMPI) obj/ls/LatinSquare.o obj/mpi/GenerateSquares.o

$(GENCUDA): obj/ls/LatinSquare.o obj/cuda/GenerateSquares.o obj/utils/FileHandler.o
	$(NVCC) -o $(GENCUDA) obj/ls/LatinSquare.o obj/cuda/GenerateSquares.o

$(CHECKSER): obj/ls/LatinSquare.o obj/serial/CheckMOLS.o
	$(CC) -o $(CHECKSER) obj/ls/LatinSquare.o obj/serial/CheckMOLS.o

$(UTILS): obj/utils/Utils.o
	$(CC) -o $(UTILS) obj/utils/Utils.o

#generate_squares_mpi: obj/LatinSquare.o obj/GenerateSquaresMPI.o
#	$(MPICC) -o $(GENSQPTH) obj/LatinSquare.o obj/GenerateSquaresMPI.o

#obj/%.o: %.cu
#	$(NVCC) -c -o $@ $< $(CFLAGS)

#generate_squares_cuda: obj/LatinSquare.o obj/GenerateSquaresCuda.o
#	$(NVCC) -o $(GENCUDA) obj/LatinSquare.o obj obj/GenerateSquaresCuda.o

.PHONY: clean
clean:
	rm obj/ls/*.o obj/serial/*.o obj/mpi/*.o obj/cuda/*.o obj/cuda/*.o obj/utils/*.o $(EXES)
