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

    // Load training images
    for(unsigned int i = 0; i < trainingLegend.numberOfRows(); ++i)
    {
	shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(i, 1).getInt());
	shmea::GString path = fname + trainingLegend.getCell(i, 0).c_str();
	printf("[NNDATA] Loading %s\n", path.c_str());
	shmea::GPointer<shmea::Image> img(new shmea::Image());
	img->LoadPNG(path);

    // Convert the label to a string for classification
    trainingLegend.setCell(i, 1, label);

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

    // Load testing data
    for(unsigned int i = 0; i < testingLegend.numberOfRows(); ++i)
    {
	shmea::GString label = shmea::GString::intTOstring(testingLegend.getCell(i, 1).getInt());
	shmea::GString path = fname + testingLegend.getCell(i, 0).c_str();
	printf("[NNDATA] Loading %s\n", path.c_str());
	shmea::GPointer<shmea::Image> img(new shmea::Image());
	img->LoadPNG(path);

    // Convert the label to a string for classification
    testingLegend.setCell(i, 1, label);
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

    // TODO FIX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // Setup the Classifier object
    for(unsigned int c = 0; c < trainingLegend.numberOfCols(); ++c)
    {
	for(unsigned int r = 0; r < trainingLegend.numberOfRows(); ++r)
	{
	    shmea::GString label = trainingLegend.getCell(r, c);
	    //printf("label = %s\n", label.c_str());
	}
	OHE* cOHE = new OHE();
	featureIsCategorical.push_back(false);

	const shmea::GType& cCell = trainingLegend.getCell(0, c); // get the first cell of the col
	if (cCell.getType() == shmea::GType::STRING_TYPE)
	{
	    //printf("Col %d is categorical\n", c);
	    cOHE->mapFeatureSpace(trainingLegend, c);
	    featureIsCategorical[c] = true;
	    //cOHE->print();
	}

	OHEMaps.push_back(cOHE);
    }

    //printf("OHEMaps.size() = %lu\n", OHEMaps.size());

    // Set the loaded flag
    loaded = true;
}

const shmea::GPointer<shmea::Image> ImageInput::getTrainImage(unsigned int row) const
{
    if(row >= trainingLegend.numberOfRows())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(row, 1).getInt());
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

    shmea::GString label = shmea::GString::intTOstring(testingLegend.getCell(row, 1).getInt());
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

    shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(index, 1).getInt());
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
    printf("Number of rows in trainingLegend = %u\n", trainingLegend.numberOfRows());
    if(index >= trainingLegend.numberOfRows())
	return emptyRow;

    // TODO: FIX THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(index, 1).getInt());
    OHE* OHEVector = OHEMaps[1];
    //return OHEVector->getOHEVector(label); // HERE!!!!!!!




    shmea::GString cCell = shmea::GString::intTOstring(trainingLegend.getCell(index, 1).getInt());
    printf("label[%u] = %s\n", index, cCell.c_str());

    // translate string to cell value for this col
    shmea::GString cString = cCell.c_str();
    std::vector<float> featureVector = (*OHEVector)[cString];
    printf("featureVector.size() = %lu\n", featureVector.size());
    for(unsigned int i=0;i<featureVector.size();++i)
	printf("%f ", featureVector[i]);
    printf("\n");

    shmea::GList retRow;
    for(unsigned int i=0;i<featureVector.size();++i)
	retRow.addFloat(featureVector[i]);

    printf("retRow.size() = %lu\n", retRow.size());
    retRow.print();
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
