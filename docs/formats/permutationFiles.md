##Usages
  The Permutations file is a file containing all possible permutations of the set of numbers leading to a particular number. 
  For example, the permutations file for the number 5 would be a file containing all possible permutations of the numbers
  1, 2, 3, 4, and 5. 
  
  This file is used when finding the reduced squares of a particular order. The file required will be the permutations of 
  the set of numbers that is one less than the Latin square's order. Example: for finding the mutually orthogonal Latin squares
  of order 6 we would need a file containing the permutations of the numbers 1, 2, 3, 4, and 5 (the 5 permutations file). This
  is because the file will be used to permute the last n-1 rows of the Latin square to form a set of reduced Latin squares.
  
##Format
  In order to properly find the mutually orthogonal squares of a particular order the permutations file must follow a
  particular format. The format is as follows, 
  + Row 1: The first row in the file must contain the number of elements being permuted, e.g. '5' for the permutations of the
    numbers 1, 2, 3, 4, and 5.
  + All subsequent rows: Each subsequent row in the file must  contain a permutation
    of the numbers. Each number in the permutation should be separated by a space, i.e. '1 3 4 5 2' vs. '13452'. 

##Notes
  For particular definitions of the terms used here, e.g. normalized and reduced, please reference the paper title 
  undergradReport.pdf in the root of the 'docs' directory in this repository. There seems to be no standard definitions of 
  some of these terms and this report outlines the definitions we used for the sake of this research. 

  Blank lines are not handled by this program. Do no have blank lines in your files. A file with a blank line SHOULD work, but
  is not gauranteed to work.
  
  ** This format is subject to change. Be sure to check back if you are having issues. 
