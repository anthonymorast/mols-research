CC=g++
CFLAGS= -O3 --std=c++11 -I. -Wall
DEPS= LatinSquare.h
UTIL=utils
GENSQ=generate_squares
EXES=utils generate_squares

all: utils generate_squares

obj/%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

obj/%.o: ../LatinSquare/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

utils: obj/Utils.o
	$(CC) -o $(UTIL) obj/Utils.o

generate_squares: obj/LatinSquare.o obj/GenerateSquares.o
	$(CC) -o $(GENSQ) obj/LatinSquare.o obj/GenerateSquares.o

.PHONY: clean
clean:
	rm obj/*.o $(EXES)
