// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#include "cmatrix.h"
#include "../../../include/Backend/Database/GList.h"
#include "../../../include/Backend/Database/gtable.h"
#include "../../../include/Backend/Database/gtype.h"
#include "gmath.h"

using namespace glades;

glades::CMatrix::CMatrix()
{
	//
}

glades::CMatrix::~CMatrix()
{
	clean();
}

void glades::CMatrix::build(unsigned int numberOfClasses)
{
	clean();

	// build an nxn gtable of int 0's
	for (unsigned int i = 0; i < numberOfClasses; ++i)
	{
		shmea::GList newList;

		for (unsigned int j = 0; j < numberOfClasses; ++j)
			newList.addInt(0);

		matrix.addRow(newList);

		// build params
		truePositive.addInt(0);
		trueNegative.addInt(0);
		falsePositive.addInt(0);
		falseNegative.addInt(0);
	}
}

void glades::CMatrix::addResult(const shmea::GList& result)
{
	// error checks
	if ((result.size() % 2) != 0)
	{
		printf("[CMATRIX] Bad result list size [0]: %d\n", result.size());
		return;
	}
	if ((result.size() / 2) != matrix.numberOfCols())
	{
		printf("[CMATRIX] Bad result list size [1]: %d\n", result.size());
		return;
	}

	// split into expected list and predicted list
	shmea::GList expected, predicted;
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		if ((i % 2) == 0)
			expected.addFloat(result.getFloat(i));
		else
			predicted.addFloat(result.getFloat(i));
	}

	// run lists through vector decomp
	shmea::GList expectedDecomp = GMath::naiveVectorDecomp(expected);
	shmea::GList predictedDecomp = GMath::naiveVectorDecomp(predicted);

	// increment confusion matrix
	// schema: expected -> row; predicted -> col
	for (unsigned int i = 0; i < expectedDecomp.size(); ++i)
	{
		if (expectedDecomp.getInt(i) == 1)
		{
			for (unsigned int j = 0; j < predictedDecomp.size(); ++j)
			{
				int newCount = (matrix.getCell(i, j).getInt() + predictedDecomp.getInt(j));
				matrix.setCell(i, j, newCount);
			}
		}
	}
}

void glades::CMatrix::updateResultParams()
{
	// schema: expected -> row; predicted -> col
	int size = matrix.numberOfRows();

	// order( 2(n^2) - n ) operation to update params
	for (int row = 0; row < size; ++row)
	{
		for (int col = 0; col < size; ++col)
		{
			shmea::GType value(matrix.getCell(row, col));

			if (row != col)
			{
				falsePositive.setGType(col, (falsePositive.getInt(col) + value.getInt()));
				falseNegative.setGType(row, (falseNegative.getInt(row) + value.getInt()));
			}
			else
			{
				// row == col needs third loop for TN's
				for (int itr = 0; itr < size; ++itr)
				{
					if (itr == row)
						truePositive.setGType(itr, value);
					else
						trueNegative.setGType(itr, (trueNegative.getInt(itr) + value.getInt()));
				}
			}
		}
	}
}

shmea::GTable glades::CMatrix::getMatrix() const
{
	return matrix;
}

float glades::CMatrix::getClassAccuracy(unsigned int index) const
{
	if (index > matrix.numberOfRows())
		return 0.0f;

	float truth = ((float)(truePositive.getInt(index) + trueNegative.getInt(index)));
	float untruth = ((float)(falsePositive.getInt(index) + falseNegative.getInt(index)));

	return (truth / (truth + untruth));
}

float glades::CMatrix::getOverallAccuracy() const
{
	float totalAccuracy = 0.0f;

	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
		totalAccuracy += getClassAccuracy(i);

	totalAccuracy /= ((float)(matrix.numberOfRows()));

	return totalAccuracy;
}

float glades::CMatrix::getClassPrecision(unsigned int index) const
{
	if (index > matrix.numberOfRows())
		return 0.0f;

	return (((float)(truePositive.getInt(index))) /
			((float)(truePositive.getInt(index) + falsePositive.getInt(index))));
}

float glades::CMatrix::getOverallPrecision() const
{
	float totalPrecision = 0.0f;

	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
		totalPrecision += getClassPrecision(i);

	totalPrecision /= ((float)(matrix.numberOfRows()));

	return totalPrecision;
}

float glades::CMatrix::getClassRecall(unsigned int index) const
{
	if (index > matrix.numberOfRows())
		return 0.0f;

	return (((float)(truePositive.getInt(index))) /
			((float)(truePositive.getInt(index) + falseNegative.getInt(index))));
}

float glades::CMatrix::getOverallRecall() const
{
	float totalRecall = 0.0f;

	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
		totalRecall += getClassRecall(i);

	totalRecall /= ((float)(matrix.numberOfRows()));

	return totalRecall;
}

float glades::CMatrix::getClassSpecificity(unsigned int index) const
{
	if (index > matrix.numberOfRows())
		return 0.0f;

	return (((float)(trueNegative.getInt(index))) /
			((float)(trueNegative.getInt(index) + falsePositive.getInt(index))));
}

float glades::CMatrix::getOverallSpecificity() const
{
	float totalSpecificity = 0.0f;

	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
		totalSpecificity += getClassSpecificity(i);

	totalSpecificity /= ((float)(matrix.numberOfRows()));

	return totalSpecificity;
}

float glades::CMatrix::getClassFalseAlarm(unsigned int index) const
{
	if (index > matrix.numberOfRows())
		return 0.0f;

	// also 1.0f - specificity
	return (((float)(falsePositive.getInt(index))) /
			((float)(trueNegative.getInt(index) + falsePositive.getInt(index))));
}

float glades::CMatrix::getOverallFalseAlarm() const
{
	float totalFalseAlarm = 0.0f;

	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
		totalFalseAlarm += getClassFalseAlarm(i);

	totalFalseAlarm /= ((float)(matrix.numberOfRows()));

	return totalFalseAlarm;
}

float glades::CMatrix::getClassF1Score(unsigned int index) const
{
	if (index > matrix.numberOfRows())
		return 0.0f;

	float cRecall = getClassRecall(index);
	float cPrecision = getClassPrecision(index);

	return ((2.0f * cRecall * cPrecision) / (cRecall + cPrecision));
}

float glades::CMatrix::getOverallF1Score() const
{
	float totalF1Score = 0.0f;

	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
		totalF1Score += getClassF1Score(i);

	totalF1Score /= ((float)(matrix.numberOfRows()));

	return totalF1Score;
}

void glades::CMatrix::print() const
{
	// bare print for now, pretty it up later
	printf("[CMATRIX] Confusion Matrix...");
	matrix.print();

	printf("[CMATRIX] True Positives...\n");
	truePositive.print();

	printf("[CMATRIX] True Negatives...\n");
	trueNegative.print();

	printf("[CMATRIX] False Positives...\n");
	falsePositive.print();

	printf("[CMATRIX] False Negatives...\n");
	falseNegative.print();
}

void glades::CMatrix::reset()
{
	// maintain shape, reset all cells to int 0's and reset params
	for (unsigned int i = 0; i < matrix.numberOfRows(); ++i)
	{
		for (unsigned int j = 0; j < matrix.numberOfCols(); ++j)
			matrix.setCell(i, j, shmea::GType(0));

		truePositive.setGType(i, shmea::GType(0));
		trueNegative.setGType(i, shmea::GType(0));
		falsePositive.setGType(i, shmea::GType(0));
		falseNegative.setGType(i, shmea::GType(0));
	}
}

void glades::CMatrix::clean()
{
	matrix.clear();

	truePositive.clear();
	trueNegative.clear();
	falsePositive.clear();
	falseNegative.clear();
}
