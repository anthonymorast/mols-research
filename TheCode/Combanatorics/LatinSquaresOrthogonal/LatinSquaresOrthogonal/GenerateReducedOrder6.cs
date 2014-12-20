using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Combanatorics;
using System.IO;

namespace LatinSquaresOrthogonal
{
    class GenerateReducedOrder6
    {
        public static List<LatinSquare> GenerateReduced()
        {
            List<LatinSquare> squares = ReadInSquares("order6norm.txt");
            List<List<int>> permutations = ReadInPermutations("5_perm.txt");

            List<LatinSquare> permutedSquares = new List<LatinSquare>();
            // 012345 103254 234501 325410 450123 541032
            //LatinSquare ls = new LatinSquare(6, new List<int>{1, 2, 3, 4, 5, 6, 2, 1, 4, 3, 6, 5, 3, 4, 5, 6, 1, 2, 
            //                                        4, 3, 6, 5, 2, 1, 5, 6, 1, 2, 3, 4, 6, 5, 2, 1, 4, 3});

            for (int i = 0; i < squares.Count; i++)
            {
                LatinSquare currentSquare = squares[i];
                for (int j = 0; j < permutations.Count; j++)
                    permutedSquares.Add(currentSquare.PermuteRows(permutations[j]));
            }

            return permutedSquares;
        }

        private static List<LatinSquare> ReadInSquares (string filename)
        {
            List<LatinSquare> returnValue = new List<LatinSquare>();
            using (StreamReader sr = new StreamReader(filename))
            {
                List<int> current = new List<int>();
                string line = "";

                while ((line = sr.ReadLine()) != null)
                {
                    current = new List<int>();
                    string[] ne = line.Split(new char[] { ' ' });
                    for (int i = 0; i < ne.Count(); i++)
                    {
                        long curr = long.Parse(ne[i]);
                        if ( i == 0 )
                        {
                            current.Add(1);
                            current.Add((int)((curr / 10000)) + 1);
                            current.Add((int)((curr % 10000) / 1000) + 1);
                            current.Add((int)((curr % 1000) / 100) + 1);
                            current.Add((int)((curr % 100) / 10) + 1);
                            current.Add((int)((curr % 10)) + 1);        
                        }
                        else
                        {
                            current.Add((int)(curr / 100000) + 1);
                            current.Add((int)((curr % 100000) / 10000) + 1);
                            current.Add((int)((curr % 10000) / 1000) + 1);
                            current.Add((int)((curr % 1000) / 100) + 1);
                            current.Add((int)((curr % 100) / 10) + 1);
                            current.Add((int)((curr % 10)) + 1);
                        }
                    }
                    returnValue.Add(new LatinSquare(6, current));
                }
            }

            return returnValue;
        }

        private static List<List<int>> ReadInPermutations(string filename)
        {
            List<List<int>> returnVal = new List<List<int>>();
            List<int> current = new List<int>{ 1 };

            using (StreamReader sr = new StreamReader(filename))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    current = new List<int> { 1 };
                    string[] ne = line.Split(new char[] { ' ' });
                    ne = ne.Where(x => !string.IsNullOrEmpty(x) && !string.IsNullOrWhiteSpace(x)).ToArray();
                    foreach (var item in ne)
                        current.Add(int.Parse(item)+1);
                    returnVal.Add(current);
                }
            }

            return returnVal;
        }
    }
}
