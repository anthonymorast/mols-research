CC=g++
CFLAGS= -O3 --std=c++11 -I. -Wall
DEPS= LatinSquare.h
UTIL=utils
GENSQSER=generate_squares_serial
GENSQPTH=generate_squares_pthreads
EXES=$(UTIL) $(GENSQSER) $(GENSQPTH)

all: utils generate_squares_serial generate_squares_pthreads

obj/%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

obj/%.o: ../LatinSquare/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

utils: obj/Utils.o
	$(CC) -o $(UTIL) obj/Utils.o

generate_squares_serial: obj/LatinSquare.o obj/GenerateSquaresSerial.o
	$(CC) -o $(GENSQSER) obj/LatinSquare.o obj/GenerateSquaresSerial.o

generate_squares_pthreads: obj/LatinSquare.o obj/GenerateSquaresPthreads.o
		$(CC) -o $(GENSQPTH) obj/LatinSquare.o obj/GenerateSquaresPthreads.o


.PHONY: clean
clean:
	rm obj/*.o $(EXES)
