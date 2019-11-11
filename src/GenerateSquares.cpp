#include "LatinSquare.h"
#include "Utils.h"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

short* get_array_from_line(string line, int size)
{
	line.erase(remove(line.begin(), line.end(), ' '), line.end());
	short *vals = new short[size];
	const char* linearr = line.c_str();
	for(int i = 0; i < size; i++)
		vals[i] = linearr[i] - '0';
	return vals;
}

void print_usage()
{
	cout << "Usage:" << endl;
	cout << "\tgenerate_sqaures <order> <iso_reps filename>" << endl;
}

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

	if(!file_exists(filename_3))
	{
		cout << filename_3 << " does not exist. Please use the utilites to generate the file." << endl;
		cont = false;
	}
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
	vector<LatinSquare> allsqs;
	//while(getline(isofile, line))
	{
		getline(isofile, line);
		ifstream permnfile; permnfile.open(filename_n);
		short* sqvalues = get_array_from_line(line, order*order);
		LatinSquare isosq(order, sqvalues);
		allsqs.push_back(isosq);

//		string symline;
//		while(getline(permnfile, symline))
//		{
//			short *symperm = get_array_from_line(symline, order);
//			LatinSquare newsq = isosq;
//			newsq.sym_permutation(symperm);
//			if(find(allsqs.begin(), allsqs.end(), newsq) == allsqs.end())
//				allsqs.push_back(newsq);
//		}

		// a simple hack for adding to vector while iterating
		int size = allsqs.size();
		for(int i = 0; i < size; i++)
		{
			ifstream perm3file; perm3file.open(filename_3);
			string rcsline;
			while(getline(perm3file, rcsline))
			{
				LatinSquare sq = allsqs[i];
				cout << "is symmetric? " << sq.is_symmetric() << "  ;";
				sq.print_flat();
				short *rcs = get_array_from_line(rcsline, 3);
				LatinSquare rcssq = sq;
				rcssq.rcs_permutation(rcs);
				if(find(allsqs.begin(), allsqs.end(), rcssq) == allsqs.end())
				{
					allsqs.push_back(rcssq);
					size++;
				}
			}
			perm3file.close();
		}

		permnfile.close();
	}

	for(auto it = allsqs.begin(); it != allsqs.end(); it++)
		sqfile << (*it).flatstring();

	isofile.close();
	sqfile.close();
	return 0;
}
