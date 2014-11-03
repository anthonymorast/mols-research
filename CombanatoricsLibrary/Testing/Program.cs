using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CombanatoricsLibrary;

namespace Testing
{
    class Program
    {
        static void Main(string[] args)
       {
            LatinSquare square = new CombanatoricsLibrary.LatinSquare();
            List<int> squareVals = new List<int>(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            int[][] latinSquare = new int[3][];
            for (int i = 0; i < 3; i++)
            {
                latinSquare[i] = new int[3];
            }

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    latinSquare[i][j] = i + j;

            Console.WriteLine(square.Size(latinSquare));
            Console.WriteLine(square.IsLatinSquare(latinSquare));
            latinSquare[0][0] = 1; latinSquare[0][1] = 2; latinSquare[0][2] = 3;
            latinSquare[1][0] = 2; latinSquare[1][1] = 3; latinSquare[1][2] = 1;
            latinSquare[2][0] = 3; latinSquare[2][1] = 1; latinSquare[2][2] = 2;
            Console.WriteLine(square.IsLatinSquare(latinSquare));
        }
    }
}
