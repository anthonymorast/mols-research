all: LatinSquare.o Utils.o GenerateSquares.o GenerateSquares Utils
	
LatinSquare.o: LatinSquare.cpp
	g++ -std=c++11 -O2 -c LatinSquare.cpp
Utils.o: Utils.cpp
	g++ -std=c++11 -O2 -c Utils.cpp
Utils:
	g++ -std=c++11 -O2 LatinSquare.o Utils.o -o utilities
GenerateSquares.o: GenerateSquares.cpp
	g++ -std=c++11 -O2 -c GenerateSquares.cpp
GenerateSquares: LatinSquare.o GenerateSquares.o
	g++ -std=c++11 -O2 LatinSquare.o GenerateSquares.o -o generate_squares

clean:
	rm *.o generate_squares utilities
