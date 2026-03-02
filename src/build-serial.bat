@echo off
REM Build serial targets from src/ (run from src directory or project root)
set SRC_DIR=%~dp0
if not "%CD%"=="%SRC_DIR%" cd /d "%SRC_DIR%"

if not exist obj\ls mkdir obj\ls
if not exist obj\serial mkdir obj\serial

echo Compiling LatinSquare...
g++ -O3 -std=c++11 -I. -Wall -c -o obj\ls\LatinSquare.o ..\LatinSquare\LatinSquare.cpp
if errorlevel 1 exit /b 1

echo Compiling serial/GenerateSquares...
g++ -O3 -std=c++11 -I. -Wall -c -o obj\serial\GenerateSquares.o serial\GenerateSquares.cpp
if errorlevel 1 exit /b 1

echo Compiling serial/CheckMOLS...
g++ -O3 -std=c++11 -I. -Wall -c -o obj\serial\CheckMOLS.o serial\CheckMOLS.cpp
if errorlevel 1 exit /b 1

echo Linking generate_squares_serial...
g++ -O3 -std=c++11 -o generate_squares_serial.exe obj\ls\LatinSquare.o obj\serial\GenerateSquares.o
if errorlevel 1 exit /b 1

echo Linking check_squares_serial...
g++ -O3 -std=c++11 -o check_squares_serial.exe obj\ls\LatinSquare.o obj\serial\CheckMOLS.o
if errorlevel 1 exit /b 1

echo Done. Built generate_squares_serial.exe and check_squares_serial.exe
