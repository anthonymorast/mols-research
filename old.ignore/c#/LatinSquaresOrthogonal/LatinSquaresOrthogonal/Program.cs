using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Combanatorics;

namespace LatinSquaresOrthogonal
{
    class Program
    {
        static void Main(string[] args)
        {
            //OrderFiveOrthogonal.FindOrthogonalOrder5(args);
            int order = 6;
            List<LatinSquare> squares = GenerateReducedOrder6.GenerateReduced();
            using (StreamWriter output = new StreamWriter("6_reduced.dat"))
            {
                var count = 1;
                foreach (var sq in squares)
                {
                    if (count % 1000 == 0)
                        Console.WriteLine(count);
                    var sqValues = "";
                    for (int i = 1; i <= order; i++)
                    {
                        for (int j = 1; j <= order; j++)
                        {
                            sqValues += (sq.GetElementAtPosition(i, j) - 1).ToString();
                        }
                    }
                    output.WriteLine(sqValues);
                    count++;
                }
            }
            //OrthogonalOrder6.FindOrthogonalOrder6(squares);
        }
    }
}
