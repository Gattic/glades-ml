// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ohe-test.h"

#include "../../unit-test.h"

#include <string>

#include "Backend/Database/GTable.h"
#include "Backend/Database/GVector.h"
#include "../../../Backend/Machine Learning/GMath/OHE.h"

// === Primary unit testing function ===
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

static bool vec_is_one_hot_at(const shmea::GVector<float>& v, unsigned int hotIdx)
{
	if (v.size() == 0u)
		return false;
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		const float want = (i == hotIdx) ? 1.0f : 0.0f;
		if (v[i] != want)
			return false;
	}
	return true;
}

static bool vec_is_all_zeros(const shmea::GVector<float>& v)
{
	for (unsigned int i = 0; i < v.size(); ++i)
		if (v[i] != 0.0f)
			return false;
	return true;
}

static std::string to_std_string_safe(const shmea::GString& s)
{
	const char* p = s.c_str();
	return p ? std::string(p) : std::string();
}

void OHEUnitTest()
{
	// Basic construction
	{
		glades::OHE ohe;
		G_assert(__FILE__, __LINE__, "Initialization failed: OHE size is not 0", ohe.size() == 0u);
		G_assert(__FILE__, __LINE__, "Initialization failed: contains() should be false", !ohe.contains("cat"));
		G_assert(__FILE__, __LINE__, "Initialization failed: indexAt() should be -1", ohe.indexAt("cat") == -1);
	}

	// addString()/contains()/indexAt()/operator[] encoding
	{
		glades::OHE ohe;
		ohe.addString("cat");
		ohe.addString("dog");
		ohe.addString("cat"); // duplicate

		G_assert(__FILE__, __LINE__, "Failed to add strings or handle duplicates correctly", ohe.size() == 2u);
		G_assert(__FILE__, __LINE__, "OHE does not contain 'cat'", ohe.contains("cat"));
		G_assert(__FILE__, __LINE__, "OHE does not contain 'dog'", ohe.contains("dog"));

		G_assert(__FILE__, __LINE__, "Index retrieval for 'cat' failed", ohe.indexAt("cat") == 0);
		G_assert(__FILE__, __LINE__, "Index retrieval for 'dog' failed", ohe.indexAt("dog") == 1);
		G_assert(__FILE__, __LINE__, "Index retrieval for non-existing string failed", ohe.indexAt("bird") == -1);

		const shmea::GVector<float> catEnc = ohe["cat"];
		const shmea::GVector<float> dogEnc = ohe["dog"];
		const shmea::GVector<float> birdEnc = ohe["bird"]; // unknown => all-zeros

		G_assert(__FILE__, __LINE__, "One-hot encoding size mismatch", catEnc.size() == 2u);
		G_assert(__FILE__, __LINE__, "One-hot encoding size mismatch", dogEnc.size() == 2u);
		G_assert(__FILE__, __LINE__, "Unknown encoding size mismatch", birdEnc.size() == 2u);

		G_assert(__FILE__, __LINE__, "One-hot encoding mismatch for 'cat' (must be strict 0/1)", vec_is_one_hot_at(catEnc, 0u));
		G_assert(__FILE__, __LINE__, "One-hot encoding mismatch for 'dog' (must be strict 0/1)", vec_is_one_hot_at(dogEnc, 1u));
		G_assert(__FILE__, __LINE__, "Unknown encoding must be all-zeros", vec_is_all_zeros(birdEnc));

		// Decode (int one-hot) and (float argmax)
		shmea::GVector<int> catInt(2u, 0);
		catInt[0] = 1;
		shmea::GVector<float> dogFloat(2u, 0.0f);
		dogFloat[1] = 0.25f;
		dogFloat[0] = 0.10f;

		const shmea::GString catDec = ohe[catInt];
		const shmea::GString dogDec = ohe[dogFloat];
		G_assert(__FILE__, __LINE__, "Decode from int one-hot failed", to_std_string_safe(catDec) == "cat");
		G_assert(__FILE__, __LINE__, "Decode from float argmax failed", to_std_string_safe(dogDec) == "dog");

		shmea::GVector<float> allZero(2u, 0.0f);
		const shmea::GString unkDec = ohe[allZero];
		G_assert(__FILE__, __LINE__, "All-zeros decode should return empty string", to_std_string_safe(unkDec).empty());
	}

	// Copy constructor must preserve lookup behavior (indexByString rebuilt)
	{
		glades::OHE ohe;
		ohe.addString("red");
		ohe.addString("green");
		ohe.addString("blue");

		glades::OHE copy(ohe);
		G_assert(__FILE__, __LINE__, "Copy ctor: size mismatch", copy.size() == ohe.size());
		G_assert(__FILE__, __LINE__, "Copy ctor: indexAt mismatch for 'red'", copy.indexAt("red") == ohe.indexAt("red"));
		G_assert(__FILE__, __LINE__, "Copy ctor: indexAt mismatch for 'blue'", copy.indexAt("blue") == ohe.indexAt("blue"));
		G_assert(__FILE__, __LINE__, "Copy ctor: unknown lookup should be -1", copy.indexAt("purple") == -1);
	}

	// mapFeatureSpace() should add unique string values from a column
	{
		shmea::GString sampleData("First,Last,Age\nMickey,Mouse,100\nDonald,Duck,99\nMickey,Mouse,101\n");
		shmea::GTable tbl(sampleData, ',', shmea::GTable::TYPE_STRING);

		glades::OHE ohe;
		ohe.mapFeatureSpace(tbl, 0); // First name column

		G_assert(__FILE__, __LINE__, "mapFeatureSpace: missing 'Mickey'", ohe.contains("Mickey"));
		G_assert(__FILE__, __LINE__, "mapFeatureSpace: missing 'Donald'", ohe.contains("Donald"));
		G_assert(__FILE__, __LINE__, "mapFeatureSpace: expected 2 unique strings in column", ohe.size() == 2u);
	}
}


