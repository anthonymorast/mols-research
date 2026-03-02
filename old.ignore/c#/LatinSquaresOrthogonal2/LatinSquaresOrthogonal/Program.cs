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
            List<LatinSquare> squares = GenerateReducedOrder6.GenerateReduced();
            OrthogonalOrder6.FindOrthogonalOrder6(squares);
        }
    }
}
