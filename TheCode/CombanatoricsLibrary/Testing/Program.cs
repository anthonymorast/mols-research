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

            LatinSquare square1 = new LatinSquare(3);
            square1.Fill(new List<int> { 1, 2, 3, 2, 3, 1, 3, 1, 2 });

            square1.GetElementAtPosition(3, 3);
            square1.GetElementAtPosition(4, 4);
            
            if (square2.IsSameIsotopyClass(square) || square.IsSameMainClass(square2))
                Console.WriteLine("These square belong to the same isotopy class or the same main class.");
        }       
    }
}
