CC=g++
NVCC=nvcc
MPICC=mpic++
CUDAFLAGS=-O3 --std=c++11 -I. -lpthread
CFLAGS= -O3 --std=c++11 -I. -Wall -lpthread
UTILS=utilities

all: generate $(UTILS)

# creates the .o file for the utilities executable
obj/utils/%.o: utils/%.cpp
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -c -o $@ $<

# creates the .o file for the Latin square object
obj/ls/%.o: ../LatinSquare/%.cpp
	$(DIR_GUARD)
	$(CC) $(CFLAGS) -c -o $@ $< 

generate: obj/ls/LatinSquare.o
	$(CC) $(CFLAGS) $< main.cpp -o generate 

# creates the utlities executable provided the .o file
$(UTILS): obj/utils/Utils.o
	$(CC) -o $(UTILS) obj/utils/Utils.o