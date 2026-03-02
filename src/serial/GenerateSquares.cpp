#include "../GenerateSquares.h"
#include <chrono>
#include <iomanip>
#include <cstdint>

using Clock = std::chrono::steady_clock;
using Sec = std::chrono::duration<double>;

// Packed square: stores up to 100 values at 4 bits each (supports order ≤ 10).
// 100 × 4 bits = 400 bits = 7 uint64_t (448 bits, 56 bytes).
static const int PACK_WORDS = 7;

struct PackedSquare {
	uint64_t w[PACK_WORDS] = {};

	bool operator==(const PackedSquare& o) const {
		for (int i = 0; i < PACK_WORDS; i++)
			if (w[i] != o.w[i]) return false;
		return true;
	}
};

struct PackedHash {
	size_t operator()(const PackedSquare& p) const {
		size_t h = 0;
		for (int i = 0; i < PACK_WORDS; i++)
			h ^= p.w[i] * (0x9e3779b97f4a7c15ULL + i);
		return h;
	}
};

PackedSquare pack_buf(const short* buf, int o_sq)
{
	PackedSquare p;
	for (int i = 0; i < o_sq; i++) {
		int word = i / 16;           // 16 nibbles per uint64_t
		int shift = (i % 16) * 4;
		p.w[word] |= (uint64_t)buf[i] << shift;
	}
	return p;
}

void unpack_buf(const PackedSquare& p, short* buf, int o_sq)
{
	for (int i = 0; i < o_sq; i++) {
		int word = i / 16;
		int shift = (i % 16) * 4;
		buf[i] = (p.w[word] >> shift) & 0xF;
	}
}

// Buffer-based permutation functions (no allocation, write directly into dest)
void permute_rows_into(short order, const short* src, short* dest, const short* perm)
{
	for (short i = 0; i < order; i++)
		for (short j = 0; j < order; j++)
			dest[i * order + j] = src[perm[i] * order + j];
}

void permute_cols_into(short order, const short* src, short* dest, const short* perm)
{
	for (short i = 0; i < order; i++)
		for (short j = 0; j < order; j++)
			dest[j * order + i] = src[j * order + perm[i]];
}

void permute_symbols_into(int o_sq, const short* src, short* dest, const short* perm)
{
	for (int i = 0; i < o_sq; i++)
		dest[i] = perm[src[i]];
}

void reduce_buf(short order, short* buf, short* temp)
{
	short perm[order];
	for (int i = 0; i < order; i++)
		perm[buf[i]] = i;
	int o_sq = order * order;
	memcpy(temp, buf, sizeof(short) * o_sq);
	for (short i = 0; i < order; i++)
		for (short j = 0; j < order; j++)
			buf[j * order + i] = temp[j * order + perm[i]];
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		print_usage();
		return 0;
	}

	short order = stoi(string(argv[1]));
	int o_sq = order * order;
	string filename_iso = string(argv[2]);
	string filename_3 = "3_perm.dat";
	string filename_n = to_string(order) + "_perm.dat";
	bool cont = true;

	double t_load_iso = 0, t_load_perm = 0, t_permute_phase = 0, t_write = 0;
	double t_permute_ops = 0, t_reduce_ops = 0, t_insert_ops = 0;

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
	ofstream sqfile; sqfile.open("squares/" + to_string(order) + "_squares.dat");

	string line;
	unordered_set<PackedSquare, PackedHash> allSqs;
	vector<PackedSquare> checkSqs;

	{
		auto t0 = Clock::now();
		short* tmpBuf = new short[o_sq];
		while(getline(isofile, line))
		{
			short* arr = get_array_from_line(line, o_sq);
			PackedSquare ps = pack_buf(arr, o_sq);
			delete[] arr;
			allSqs.insert(ps);
			checkSqs.push_back(ps);
		}
		delete[] tmpBuf;
		isofile.close();
		t_load_iso = std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
	}

	vector<short*> permVec;
	{
		auto t0 = Clock::now();
		ifstream permfile; permfile.open(filename_n);
		string permline;
		while(getline(permfile, permline))
		{
			permVec.push_back(get_array_from_line(permline, order));
		}
		permfile.close();
		t_load_perm = std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
	}

	// Pre-allocate reusable buffers
	short* baseBuf = new short[o_sq];
	short* rowBuf = new short[o_sq];
	short* colBuf = new short[o_sq];
	short* symBuf = new short[o_sq];
	short* tempBuf = new short[o_sq];

	size_t startIdx = 0;
	do {
		size_t endIdx = checkSqs.size();
		cout << "checkSqs.size() = " << checkSqs.size() << ", processing [" << startIdx << ", " << endIdx << ")" << endl;

		{
			auto t0 = Clock::now();
			for (size_t i = startIdx; i < endIdx; i++)
			{
				if (i > startIdx && (i - startIdx) % SQ_CHECK_REPORT == 0)
				{
					cout << "Checking square " << (i - startIdx) << " of " << (endIdx - startIdx) << endl;
				}

				// Unpack base square into buffer (push_back may invalidate pointers)
				unpack_buf(checkSqs[i], baseBuf, o_sq);

				for (auto permIt = permVec.begin(); permIt != permVec.end(); permIt++)
				{
					short* permArr = (*permIt);

					{
						auto tp = Clock::now();
						permute_rows_into(order, baseBuf, rowBuf, permArr);
						permute_cols_into(order, baseBuf, colBuf, permArr);
						permute_symbols_into(o_sq, baseBuf, symBuf, permArr);
						t_permute_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}

					{
						auto tp = Clock::now();
						reduce_buf(order, rowBuf, tempBuf);
						reduce_buf(order, colBuf, tempBuf);
						reduce_buf(order, symBuf, tempBuf);
						t_reduce_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}

					{
						auto tp = Clock::now();
						PackedSquare rp = pack_buf(rowBuf, o_sq);
						if (allSqs.insert(rp).second)
							checkSqs.push_back(rp);
						PackedSquare cp = pack_buf(colBuf, o_sq);
						if (allSqs.insert(cp).second)
							checkSqs.push_back(cp);
						PackedSquare sp = pack_buf(symBuf, o_sq);
						if (allSqs.insert(sp).second)
							checkSqs.push_back(sp);
						t_insert_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
				}
			}
			t_permute_phase += std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
		}

		startIdx = endIdx;
		cout << "Start Count: " << allSqs.size() << ", next batch from " << startIdx << endl;
	} while (startIdx < checkSqs.size());

	delete[] baseBuf; delete[] rowBuf; delete[] colBuf; delete[] symBuf; delete[] tempBuf;

	// --- Write output: unpack each packed square and print values ---
	{
		auto t0 = Clock::now();
		short* outBuf = new short[o_sq];
		for (const auto& ps : allSqs)
		{
			unpack_buf(ps, outBuf, o_sq);
			for (int j = 0; j < o_sq; j++)
				sqfile << outBuf[j];
			sqfile << '\n';
		}
		delete[] outBuf;
		sqfile.close();
		t_write = std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
	}

	// Timing report
	double total = t_load_iso + t_load_perm + t_permute_phase + t_write;
	cout << "\n--- Serial timing (seconds) ---\n";
	cout << std::fixed << std::setprecision(3);
	double pct = (total > 0) ? 1.0 / total : 0;
	cout << "  Load isotopy reps:  " << std::setw(8) << t_load_iso  << "  (" << (100.0 * t_load_iso  * pct) << "%)\n";
	cout << "  Load permutations:  " << std::setw(8) << t_load_perm << "  (" << (100.0 * t_load_perm * pct) << "%)\n";
	cout << "  Permute phase (inner loop): " << std::setw(8) << t_permute_phase << "  (" << (100.0 * t_permute_phase * pct) << "%)\n";
	cout << "    - permute row/col/sym: " << std::setw(8) << t_permute_ops << "  (" << (100.0 * t_permute_ops * pct) << "%)\n";
	cout << "    - reduce (x3):         " << std::setw(8) << t_reduce_ops << "  (" << (100.0 * t_reduce_ops * pct) << "%)\n";
	cout << "    - pack + insert:        " << std::setw(8) << t_insert_ops << "  (" << (100.0 * t_insert_ops * pct) << "%)\n";
	cout << "  Write output:       " << std::setw(8) << t_write << "  (" << (100.0 * t_write * pct) << "%)\n";
	cout << "  TOTAL:              " << std::setw(8) << total << "\n";

	return 0;
}
