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
		int word = i / 16;
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

// Swap-based generator functions: copy src into dest, then swap two rows/cols/symbols.
void swap_rows(const short* src, short* dest, int o_sq, short order, short a, short b)
{
	memcpy(dest, src, sizeof(short) * o_sq);
	for (short j = 0; j < order; j++)
		std::swap(dest[a * order + j], dest[b * order + j]);
}

void swap_cols(const short* src, short* dest, int o_sq, short order, short a, short b)
{
	memcpy(dest, src, sizeof(short) * o_sq);
	for (short j = 0; j < order; j++)
		std::swap(dest[j * order + a], dest[j * order + b]);
}

void swap_symbols(const short* src, short* dest, int o_sq, short a, short b)
{
	for (int i = 0; i < o_sq; i++) {
		if (src[i] == a)      dest[i] = b;
		else if (src[i] == b) dest[i] = a;
		else                  dest[i] = src[i];
	}
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
	bool cont = true;

	double t_load_iso = 0, t_permute_phase = 0, t_write = 0;
	double t_swap_ops = 0, t_reduce_ops = 0, t_insert_ops = 0;

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
	unordered_set<PackedSquare, PackedHash> allSqs;
	vector<PackedSquare> checkSqs;

	{
		auto t0 = Clock::now();
		while(getline(isofile, line))
		{
			short* arr = get_array_from_line(line, o_sq);
			PackedSquare ps = pack_buf(arr, o_sq);
			delete[] arr;
			allSqs.insert(ps);
			checkSqs.push_back(ps);
		}
		isofile.close();
		t_load_iso = std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
	}

	// Pre-allocate reusable buffers
	short* baseBuf = new short[o_sq];
	short* swapBuf = new short[o_sq];
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

				unpack_buf(checkSqs[i], baseBuf, o_sq);

				// Apply (n-1) adjacent transposition generators to rows, cols, symbols
				for (short a = 0; a < order - 1; a++)
				{
					// Row swap
					{
						auto tp = Clock::now();
						swap_rows(baseBuf, swapBuf, o_sq, order, a, (short)(a + 1));
						t_swap_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
					{
						auto tp = Clock::now();
						reduce_buf(order, swapBuf, tempBuf);
						t_reduce_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
					{
						auto tp = Clock::now();
						PackedSquare ps = pack_buf(swapBuf, o_sq);
						if (allSqs.insert(ps).second)
							checkSqs.push_back(ps);
						t_insert_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}

					// Col swap
					{
						auto tp = Clock::now();
						swap_cols(baseBuf, swapBuf, o_sq, order, a, (short)(a + 1));
						t_swap_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
					{
						auto tp = Clock::now();
						reduce_buf(order, swapBuf, tempBuf);
						t_reduce_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
					{
						auto tp = Clock::now();
						PackedSquare ps = pack_buf(swapBuf, o_sq);
						if (allSqs.insert(ps).second)
							checkSqs.push_back(ps);
						t_insert_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}

					// Symbol swap
					{
						auto tp = Clock::now();
						swap_symbols(baseBuf, swapBuf, o_sq, a, (short)(a + 1));
						t_swap_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
					{
						auto tp = Clock::now();
						reduce_buf(order, swapBuf, tempBuf);
						t_reduce_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
					{
						auto tp = Clock::now();
						PackedSquare ps = pack_buf(swapBuf, o_sq);
						if (allSqs.insert(ps).second)
							checkSqs.push_back(ps);
						t_insert_ops += std::chrono::duration_cast<Sec>(Clock::now() - tp).count();
					}
				}
			}
			t_permute_phase += std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
		}

		startIdx = endIdx;
		cout << "Start Count: " << allSqs.size() << ", next batch from " << startIdx << endl;
	} while (startIdx < checkSqs.size());

	delete[] baseBuf; delete[] swapBuf; delete[] tempBuf;

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
	double total = t_load_iso + t_permute_phase + t_write;
	cout << "\n--- Serial timing (seconds) ---\n";
	cout << std::fixed << std::setprecision(3);
	double pct = (total > 0) ? 1.0 / total : 0;
	cout << "  Load isotopy reps:  " << std::setw(8) << t_load_iso  << "  (" << (100.0 * t_load_iso  * pct) << "%)\n";
	cout << "  Generator phase:    " << std::setw(8) << t_permute_phase << "  (" << (100.0 * t_permute_phase * pct) << "%)\n";
	cout << "    - swap row/col/sym:  " << std::setw(8) << t_swap_ops << "  (" << (100.0 * t_swap_ops * pct) << "%)\n";
	cout << "    - reduce:            " << std::setw(8) << t_reduce_ops << "  (" << (100.0 * t_reduce_ops * pct) << "%)\n";
	cout << "    - pack + insert:     " << std::setw(8) << t_insert_ops << "  (" << (100.0 * t_insert_ops * pct) << "%)\n";
	cout << "  Write output:       " << std::setw(8) << t_write << "  (" << (100.0 * t_write * pct) << "%)\n";
	cout << "  TOTAL:              " << std::setw(8) << total << "\n";
	cout << "  Total squares:      " << allSqs.size() << "\n";

	return 0;
}
