#include "../GenerateSquares.h"

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		print_usage();
		return 0;
	}

	short order = stoi(string(argv[1]));
	string filename_iso = string(argv[2]);
	string filename_3 = "3_perm.dat";
	string filename_n = to_string(order) + "_perm.dat";
	bool cont = true;

	// we need this for Main class equivalence (interchanging RCS) not for isotopy class equivalence
	/*if(!file_exists(filename_3))
	{
		cout << filename_3 << " does not exist. Please use the utilites to generate the file." << endl;
		cont = false;
	}*/
   	if (!file_exists(filename_n))
	{
		cout << filename_n << " does not exist. Please use the utilites to generate the file." << endl;
		cont = false;
	}
	if(!file_exists(filename_iso))
	{
		cout << filename_iso << " does not exist." << endl;
		cont = false;
	}

	if (!cont)
		return 0;

	ifstream isofile; isofile.open(filename_iso);
	ofstream sqfile; sqfile.open(to_string(order) + "_squares.dat");

	string line;
	unordered_set<string> allSqs;
	vector<LatinSquare> checkSqs;		// squares to permute, do not permute all squares everytime
	while(getline(isofile, line))
	{
		// get the party started by loading the squares vector with the isotopy
		// class representatives
		LatinSquare isoSq(order, get_array_from_line(line, order*order));
		allSqs.insert(isoSq.flatstring_no_space());
		checkSqs.push_back(isoSq);
	}
	isofile.close();

	vector<short*> permVec;
	ifstream permfile; permfile.open(filename_n);
	string permline;
	while(getline(permfile, permline))
	{
		permVec.push_back(get_array_from_line(permline, order));
	}
	permfile.close();

	time_t start, end;
	start = clock();

	// keep processing while new squares are added to allSqs
	long unsigned int numSqs;
  	do {
		// set numSqs to current size of the allSqs vector
		numSqs = allSqs.size();
		unordered_set<string> newSquares;
		int sqsToCheck = checkSqs.size();
		int count = 0;

		// for each square to be permuted
		for(auto it = checkSqs.begin(); it != checkSqs.end(); it++)
		{
			if(count > 0 && count % SQ_CHECK_REPORT == 0)
			{
				cout << "Checking square " << count << " of " << sqsToCheck << endl;
			}

			// perform all permutations of row, col, sym
			for(auto permIt = permVec.begin(); permIt != permVec.end(); permIt++)
			{
				LatinSquare baseSq = (*it);
				short* permArr = (*permIt);
				LatinSquare rowSq = baseSq;
				LatinSquare colSq = baseSq;
				LatinSquare symSq = baseSq;

				rowSq.permute_rows(permArr);
				colSq.permute_cols(permArr);
				symSq.permute_symbols(permArr);

				if(!rowSq.is_valid() || !colSq.is_valid() || !symSq.is_valid())
				{
						cout << "ERROR!" << endl;
						cout << "Generated invalid square while applying permutation: " << endl
							 << endl << permline << " to the square " << endl << endl
							 << baseSq.tostring() << endl << endl
							 << "This created the following row, col, and sym squares, "
							 << "respectively" << endl << endl << rowSq.tostring() << endl << endl
							 << colSq.tostring() << endl << endl << symSq.tostring() << endl;
						exit(0);
				}

				// TODO: take advantage of normalized square efficiencies by 
				// not using string representations
				newSquares.insert(rowSq.flatstring_no_space());
				newSquares.insert(colSq.flatstring_no_space());
				newSquares.insert(symSq.flatstring_no_space());
			}
			count++;
		}

		// these squares were checked so delete
		checkSqs.clear();
		pair<unordered_set<string>::iterator, bool> returnValue;
		for(auto it = newSquares.begin(); it != newSquares.end(); it++)
		{
			string lsString = (*it);
			returnValue = allSqs.insert(lsString);
			if(returnValue.second)
			{
				LatinSquare ls = LatinSquare(order, get_array_from_line(lsString, order*order));
				checkSqs.push_back(ls);
			}
		}
		newSquares.clear();

		// process until the number of squares at the end of the while loop
		// is the same as it was at the start (i.e. until no new squares are added)
		cout << "Start Count: " << numSqs << ", End Count: " << allSqs.size() << endl;
	} while (numSqs < allSqs.size());


	end = clock();
	double timeTaken = double(end-start) / double(CLOCKS_PER_SEC);
	cout << "SERIAL Time Taken: " << timeTaken << " seconds" << endl;

	// write all squares to a file
	for(auto it = allSqs.begin(); it != allSqs.end(); it++)
		sqfile << (*it) << endl;

	sqfile.close();
	return 0;
}
