using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace CombanatoricsLibrary
{
    /// <summary>
    /// This class will implement many common operations used in combinatorics. These will be
    /// similar to combinations and permutations, producing all combinations of a certain number,
    /// etc. 
    /// </summary>
    class Combinatorics
    {
        /// <summary>
        /// This method takes in an integer and produces all the possible permutations of the integer.
        /// For example, passing in '4' will produce a List of Arrays of the format enty one = 1,2,3,4 
        /// entry two = 1,2,4,3; etc.
        /// </summary>
        /// <param name="order">The of which to find permutations</param>
        public static List<long[]> ProducePermutations(long order)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            return new List<long[]>();
        }

        /// <summary>
        /// This method takes in an integer and produces all the possible permutations of the integer.
        /// For example, passing in '4' will produce a List of Arrays of the format enty one = 1,2,3,4 
        /// entry two = 1,2,4,3; etc. It will also write these values out to a file.
        /// </summary>
        /// <param name="order"> The of which to find permutations</param>
        /// <param name="filename"> THe file path with filename to output permutations.</param>
        /// <returns></returns>
        public static List<long[]> ProducePermutations (long order, string filename)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            return new List<long[]>();
        }

        public static List<long[]> ProduceCombinations (long order)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            return new List<long[]>();
        }

        /// <summary>
        /// Returns the number of permutations of a given set of numbers, n Pick k.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        public static long Permutations(long n, long k)
        {
            if (k > n)
                throw new ApplicationException(string.Format(
                        "Cannot find number of permutations for {0} pick {1}, k > n.", n, k));

            if (k == 0)
                return 1;
            else if (k == 1)
                return n;

            long result = 1;
            long stop = (n - k);

            for (long i = n; i > stop; i--)
                result *= i;

            return result;
        }

        /// <summary>
        /// Returns the total number of combinations of a given set of number, n Choose k.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        public static long Combinations(long n, long k)
        {
            if (k > n)
                throw new ApplicationException(string.Format(
                          "Cannot find number of combinations for {0} choose {1}, k > n.", n, k));
            if (k == 0)
                return 1;
            else if (k == 1)
                return n;

            long result = 1;  

            for (long d = 1; d <= k; d++)
            {
                result *= n--;
                result /= d;
            }

            return result;
        }
    }
}
