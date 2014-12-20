using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Combanatorics;
using System.IO;

namespace LatinSquaresOrthogonal
{
    class OrderFiveOrthogonal
    {
        public static void FindOrthogonalOrder5 (String[] args)
        {
            LatinSquare square1 = new LatinSquare(int.Parse(args[0]));
            LatinSquare square2 = new LatinSquare(int.Parse(args[0]));
            LatinSquare square3 = new LatinSquare(int.Parse(args[0]));

            List<LatinSquare> squares = ReadInSquares("reduced_squares_o5.dat", int.Parse(args[0]));

            List<Tuple<int, int>> orthogonalPairs = new List<Tuple<int, int>>();
            List<Tuple<int, int, int>> orthogonalTriplets = new List<Tuple<int, int, int>>();

            // Get pairs of MOLS
            for (int i = 0; i < squares.Count; i++)
            {
                square1 = squares[i];
                for (int j = i; j < squares.Count; j++)
                {
                    square2 = squares[j];

                    if (square1.IsOrthogonal(square2))
                        orthogonalPairs.Add(new Tuple<int, int>(i, j));
                }
            }

            // Get element indicies that will be used for further testing 
            List<int> distinct = new List<int>();

            foreach (var pair in orthogonalPairs)
            {
                if (!distinct.Contains(pair.Item1))
                    distinct.Add(pair.Item1);
                if (!distinct.Contains(pair.Item2))
                    distinct.Add(pair.Item2);
            }     

            // Get triplets of MOLS
            for (int i = 0; i < orthogonalPairs.Count; i++)
            {
                square1 = squares[orthogonalPairs[i].Item1];
                square2 = squares[orthogonalPairs[i].Item2];

                for (int j = 0; j < distinct.Count; j++)
                {
                    // square1 == square3 or square2 == square3
                    if (distinct[j] == (orthogonalPairs[i].Item2) || distinct[j] == (orthogonalPairs[i].Item1))
                        continue;

                    square3 = squares[distinct[j]];
                    var currentTrip = new Tuple<int, int, int>(orthogonalPairs[i].Item1, orthogonalPairs[i].Item2, distinct[j]);
                    if (square3.IsOrthogonal(square1) && square3.IsOrthogonal(square2) 
                        && !Contains(currentTrip, orthogonalTriplets))
                        orthogonalTriplets.Add(currentTrip);
                }
            }
            
            Console.WriteLine(orthogonalPairs.Count + " " + orthogonalTriplets.Count);
            Console.Read(); 
        }

        // Read the squares in from the file
        private static List<LatinSquare> ReadInSquares(string filename, int order)
        {
            StreamReader reader = new StreamReader(filename);
            List<LatinSquare> squares = new List<LatinSquare>();

            string line;
            List<int> current = new List<int>();
            
            while ((line = reader.ReadLine()) != null)
            {
                string[] ne = line.Split(new char[] { ' ' });
                ne = ne.Where(x => !string.IsNullOrEmpty(x) && !string.IsNullOrWhiteSpace(x)).ToArray();
                current = ne.Select(x => int.Parse(x)).ToList();
                squares.Add(new LatinSquare(order, current));
            }
            
            return squares;
        }

        private static bool Contains(Tuple<int,int,int> current, List<Tuple<int,int,int>> orthogonalSets)
        {
            int count = 0;
            foreach (var item in orthogonalSets)
            {
                count = 0;

                if (item.Item1 == current.Item1 || item.Item1 == current.Item2 || item.Item1 == current.Item3)
                    count++;
                if (item.Item2 == current.Item1 || item.Item2 == current.Item2 || item.Item2 == current.Item3)
                    count++;
                if (item.Item3 == current.Item1 || item.Item3 == current.Item2 || item.Item3 == current.Item3)
                    count++;

                if (count == 3) break;
            }

            return count == 3 ? true : false;
        }

        private static bool Contains(Tuple<int, int, int, int> current, List<Tuple<int, int, int,int>> orthogonalSets)
        {
            int count = 0;
            foreach (var item in orthogonalSets)
            {
                count = 0;

                if (item.Item1 == current.Item1 || item.Item1 == current.Item2 || item.Item1 == current.Item3 || item.Item1 == current.Item4)
                    count++;
                if (item.Item2 == current.Item1 || item.Item2 == current.Item2 || item.Item2 == current.Item3 || item.Item2 == current.Item4)
                    count++;
                if (item.Item3 == current.Item1 || item.Item3 == current.Item2 || item.Item3 == current.Item3 || item.Item3 == current.Item4)
                    count++;
                if (item.Item4 == current.Item1 || item.Item4 == current.Item2 || item.Item4 == current.Item3 || item.Item4 == current.Item4)
                    count++;

                if (count == 4) break;
            }

            return count == 4 ? true : false;
        }
    }
}
