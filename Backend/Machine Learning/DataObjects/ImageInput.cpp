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
#include "ImageInput.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "../GMath/OHE.h"

using namespace glades;

void ImageInput::import(shmea::GString newName)
{
    if(loaded)
	return;

    name = newName;
    shmea::GString fname = "datasets/images/" + name + "/";

    //
    shmea::GString trainFName = fname + "train.csv";
    shmea::GString testFName = fname + "test.csv";
    int importType = shmea::GTable::TYPE_FILE;

    trainingLegend = shmea::GTable(trainFName, ',', importType);
    testingLegend = shmea::GTable(testFName, ',', importType);

    if ((trainingLegend.numberOfRows() == 0) || (testingLegend.numberOfRows() == 0))
    {
	printf("[NNDATA] Could not load data\n");
	return;
    }


    float fMin = 0.0f;
    float fMax = 0.0f;
    float fMean = 0.0f;

    // Load training images
    unsigned int inputCol = 0;
    unsigned int outputCol = 1;
    for(unsigned int r = 0; r < trainingLegend.numberOfRows(); ++r)
    {
	shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(r, outputCol).getInt());
	shmea::GString path = fname + trainingLegend.getCell(r, inputCol).c_str();
	printf("[NNDATA] Loading %s\n", path.c_str());
	shmea::GPointer<shmea::Image> img(new shmea::Image());
	img->LoadPNG(path);

	// Convert the label to a string for classification
	trainingLegend.setCell(r, outputCol, label);

	if (r == 0)
	{
	    // Really only need it for the output column for images so the first OHE will be empty
	    for(unsigned int c = 0; c < trainingLegend.numberOfCols(); ++c)
	    {
		OHE* cOHE = new OHE();
		featureIsCategorical.push_back(false);
		OHEMaps.push_back(cOHE);
	    }
	}

	float cell = 0.0f;
	const shmea::GType& cCell = trainingLegend.getCell(r, outputCol); // get the first cell of the col
	if (cCell.getType() == shmea::GType::STRING_TYPE)
	{
		shmea::GString strCell = cCell;
		OHEMaps[outputCol]->addString(cCell);
		featureIsCategorical[outputCol] = true;
		//continue;
	}
	else if (cCell.getType() == shmea::GType::CHAR_TYPE)
		cell = cCell.getChar();
	else if (cCell.getType() == shmea::GType::SHORT_TYPE)
		cell = cCell.getShort();
	else if (cCell.getType() == shmea::GType::INT_TYPE)
		cell = cCell.getInt();
	else if (cCell.getType() == shmea::GType::LONG_TYPE)
		cell = cCell.getLong();
	else if (cCell.getType() == shmea::GType::FLOAT_TYPE)
		cell = cCell.getFloat();
	else if (cCell.getType() == shmea::GType::DOUBLE_TYPE)
		cell = cCell.getDouble();
	else if (cCell.getType() == shmea::GType::BOOLEAN_TYPE)
		cell = cCell.getBoolean() ? 1.0f : 0.0f;

	if (r == 0)
	{
		fMin = cell;
		fMax = cell;
	}

	// Check the mins and maxes
	if (cell < fMin)
		fMin = cell;
	if (cell > fMax)
		fMax = cell;

	// update mean
	fMean += cell;





	// Add a label if it doesn't exist
	if(trainImages.find(label) == trainImages.end())
	{
	    trainImages.insert(
		std::pair<shmea::GString, std::map<shmea::GString, shmea::GPointer<shmea::Image> > >
		    (label, std::map<shmea::GString, shmea::GPointer<shmea::Image> >()));
	}

	// Add the image to the label
	if(trainImages[label].find(path) == trainImages[label].end())
	{
	    trainImages[label].insert(std::pair<shmea::GString, shmea::GPointer<shmea::Image> >(path, img));
	    //trainingData.addRow(img->flatten());
	    //printf("trainImages[%s].size() = %lu\n", label.c_str(), trainImages[label].size());
	}
    }

    fMean /= trainingLegend.numberOfRows();
    OHEMaps[outputCol]->setMin(fMin);
    OHEMaps[outputCol]->setMax(fMax);
    OHEMaps[outputCol]->setMean(fMean);

    // Load testing data
    for(unsigned int r = 0; r < testingLegend.numberOfRows(); ++r)
    {
	shmea::GString label = shmea::GString::intTOstring(testingLegend.getCell(r, outputCol).getInt());
	shmea::GString path = fname + testingLegend.getCell(r, inputCol).c_str();
	printf("[NNDATA] Loading %s\n", path.c_str());
	shmea::GPointer<shmea::Image> img(new shmea::Image());
	img->LoadPNG(path);

    // Convert the label to a string for classification
    testingLegend.setCell(r, outputCol, label);
	// Add a label if it doesn't exist
	if(testImages.find(label) == testImages.end())
	{
	    testImages.insert(
		std::pair<shmea::GString, std::map<shmea::GString, shmea::GPointer<shmea::Image> > >
		    (label, std::map<shmea::GString, shmea::GPointer<shmea::Image> >()));
	}

	// Add the image to the label
	if(testImages[label].find(path) == testImages[label].end())
	{
	    testImages[label].insert(std::pair<shmea::GString, shmea::GPointer<shmea::Image> >(path, img));
	    //testingData.addRow(img->flatten());
	    //printf("testImages[%s].size() = %lu\n", label.c_str(), testImages[label].size());
	}
    }

    //printf("OHEMaps.size() = %lu\n", OHEMaps.size());

    // Set the loaded flag
    loaded = true;
}

const shmea::GPointer<shmea::Image> ImageInput::getTrainImage(unsigned int row) const
{
    if(row >= trainingLegend.numberOfRows())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    shmea::GString label = trainingLegend.getCell(row, 1);
    shmea::GString fname = "datasets/images/" + name + "/" + trainingLegend.getCell(row, 0).c_str();

    // Check if the label exists
    if(trainImages.find(label) == trainImages.end())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    // Check if the image exists
    std::map<shmea::GString, shmea::GPointer<shmea::Image> >::const_iterator itr
	= trainImages.at(label).find(fname);
    if(itr == trainImages.at(label).end())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    // Return the image
    return itr->second;
}

const shmea::GPointer<shmea::Image> ImageInput::getTestImage(unsigned int row) const
{
    if(row >= testingLegend.numberOfRows())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    shmea::GString label = testingLegend.getCell(row, 1);
    shmea::GString fname = "datasets/images/" + name + "/" + testingLegend.getCell(row, 0).c_str();

    // Check if the label exists
    if(testImages.find(label) == testImages.end())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    // Check if the image exists
    std::map<shmea::GString, shmea::GPointer<shmea::Image> >::const_iterator itr
	= testImages.at(label).find(fname);
    if(itr == testImages.at(label).end())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    // Return the image
    return itr->second;
}

shmea::GList ImageInput::getTrainRow(unsigned int index) const
{
    int inputType = glades::DataInput::IMAGE;
    static const unsigned int numRows = trainingLegend.numberOfRows(); // Cache number of rows
    if (index >= numRows)
        return emptyRow;

    shmea::GString label = trainingLegend.getCell(index, 1);
    shmea::GString fname = "datasets/images/" + name + "/" + trainingLegend.getCell(index, 0).c_str();

    // Check if the label exists
    if(trainImages.find(label) == trainImages.end())
	return emptyRow;
	
    // Check if the image exists
    std::map<shmea::GString, shmea::GPointer<shmea::Image> >::const_iterator itr
	= trainImages.at(label).find(fname);
    if(itr == trainImages.at(label).end())
	return emptyRow;
	
    // Return the image
    shmea::GList retList = itr->second->flatten();
    retList.standardize(inputType);
    return retList;
}

shmea::GList ImageInput::getTrainExpectedRow(unsigned int index) const
{
    if(index >= trainingLegend.numberOfRows())
	return emptyRow;

    // TODO: FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(index, 1).getInt());
    OHE* OHEVector = OHEMaps[1];
    //return OHEVector->getOHEVector(label); // HERE!!!!!!!




    shmea::GString cCell = shmea::GString::intTOstring(trainingLegend.getCell(index, 1).getInt());

    // translate string to cell value for this col
    shmea::GString cString = cCell.c_str();
    std::vector<float> featureVector = (*OHEVector)[cString];

    shmea::GList retRow;
    for(unsigned int i=0;i<featureVector.size();++i)
	retRow.addFloat(featureVector[i]);

    return retRow;
}

shmea::GList ImageInput::getTestExpectedRow(unsigned int index) const
{
    if(index >= testingLegend.numberOfRows())
	return shmea::GList();

    // TODO: FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //shmea::GString label = shmea::GString::intTOstring(testingLegend.getCell(index, 1).getInt());
    OHE* OHEVector = OHEMaps[1];
    //return OHEVector->getOHEVector(label); // HERE!!!!!!!




    shmea::GString cCell = shmea::GString::intTOstring(testingLegend.getCell(index, 1).getInt());

    // translate string to cell value for this col
    shmea::GString cString = cCell.c_str();
    std::vector<float> featureVector = (*OHEVector)[cString];

    shmea::GList retRow;
    for(unsigned int i=0;i<featureVector.size();++i)
	retRow.addFloat(featureVector[i]);
    return retRow;
}

shmea::GList ImageInput::getTestRow(unsigned int index) const
{
    int inputType = glades::DataInput::IMAGE;
    if(index >= testingLegend.numberOfRows())
	return shmea::GList();

    shmea::GString label = shmea::GString::intTOstring(testingLegend.getCell(index, 1).getInt());
    shmea::GString fname = "datasets/images/" + name + "/" + testingLegend.getCell(index, 0).c_str();

    // Check if the label exists
    if(testImages.find(label) == testImages.end())
	return shmea::GList();
	
    // Check if the image exists
    std::map<shmea::GString, shmea::GPointer<shmea::Image> >::const_iterator itr
	= testImages.at(label).find(fname);
    if(itr == testImages.at(label).end())
	return shmea::GList();
	
    // Return the image
    shmea::GList retList = itr->second->flatten();
    retList.standardize(inputType);
    return retList;
}


unsigned int ImageInput::getTrainSize() const
{
    return trainingLegend.numberOfRows();
}

unsigned int ImageInput::getTestSize() const
{
    return testingLegend.numberOfRows();
}

unsigned int ImageInput::getFeatureCount() const
{
	if(trainImages.size() == 0)
		return 0;

	unsigned int retVal = trainImages.begin()->second.begin()->second->getPixelCount();
	return retVal;
}

int ImageInput::getType() const
{
    return IMAGE;
}
