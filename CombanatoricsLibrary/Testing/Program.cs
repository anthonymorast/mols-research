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
            LatinSquare square = new LatinSquare(4);
            LatinSquare square2 = new LatinSquare(4);
           
            square.SetValues(new List<int> { 5, 6, 7, 8, 2, 1, 11, 4, 3, 10, 1, 19, 1, 2, 3, 20 });
            Console.WriteLine(square2.ToString() + square.ToString());
            bool mo = square.IsMutuallyOrthogonal(square2);

            if (square2.IsSameIsotopyClass(square) || square.IsSameMainClass(square2))
                Console.WriteLine("These square belong to the same isotopy class or the same main class.");
        }       
    }
}
