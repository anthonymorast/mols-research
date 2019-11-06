S(ymbol) R(ow) C(olumn) Permute Algorihtm:

for each isoptopy class representative:
	create 'D' array	# 2D array where D[0] = row values (1, 1, 1, ..., 1, 2, 2, 2, ..., 2, ..., n, n, ..., n)
						# D[1] are col values (1, 2, 3, ..., n, 1, 2, 3, ..., n, ...)
						# D[2] are isotopy class repr. square values.
	for each 3-perm (permutation of {1, 2, 3}):
		permute the rows, columns, and symbols by swapping the rows of D
		turn D into a latin square (newLS[D[0][i]][D[1][i]] = D[2][i])
		permute symbols of newLS and normalize
			- read in permutations of {1, 2, 3, ..., n} and set square values based on this 
				- newLS(i, j) = sigma(newLS(i, j))
			- re-normalize newLS
			- if normalized square has not been generated before, write to file, otherwise ignore
				- alternatively, we could write all squares to a file and weed out duplicates later, this is how it was done in the original matlab
