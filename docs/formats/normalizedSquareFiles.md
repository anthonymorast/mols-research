##Usages
  The main purpose of the Normalized Square File is to provide the normalized Latin squares of a particular order which will
 be used when generating the reduced Latin squares of that order. 

##Format
  The Normalized Latin Squares file should have the following format:
  + Line 1: The first line in the file should be the order of the Latin squares
  + Subsequent Lines: Each other line in the file should contain one Latin square
    
  Furthermore each line in the file representing a Latin square should take this form:
  + Each  line will have *n* sections each containing *n* numbers which form the Latin square. The sections will be 0-based 
    sequences and will be separated by a space.
    + Section: 01234...*n*
    + Line: 0123...*n* 1324...*n* ... 
      
##Notes
  For particular definitions of the terms used here, e.g. normalized and reduced, please reference the paper
  titled undergradReport.pdf in the root of the 'docs' directory in this repository. There seems to be no standard definitions
  of some of these terms and this report outlines the definitions we used for the sake of this research.

** This format is subject to change. Be sure to check back if you are having issues.
