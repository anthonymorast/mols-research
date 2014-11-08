using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        public List<long[]> ProducePermutations(long order)
        {
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
        public List<long[]> ProducePermutations (long order, string filename)
        {
            return new List<long[]>();
        }

        /// <summary>
        /// Returns the number of permutations of a given number, calculated the typical way.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        public long Permutations (long order)
        {
            return 0;
        }

        /// <summary>
        /// Returns the total number of combinations of a given number.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        public long Combinations (long order)
        {
            return 0;
        }
    }
}
