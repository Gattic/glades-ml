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

using namespace glades;

void ImageInput::import(shmea::GString fname)
{
    if(loaded)
	return;

    name = fname;

    //
    shmea::GString basePath = "datasets/images/" + fname + "/";

    shmea::GString trainFName = basePath + "train.csv";
    shmea::GString testFName = basePath + "test.csv";
    int importType = shmea::GTable::TYPE_FILE;

    trainingLegend = shmea::GTable(trainFName, ',', importType);
    testingLegend = shmea::GTable(testFName, ',', importType);

    if ((trainingLegend.numberOfRows() == 0) || (testingLegend.numberOfRows() == 0))
    {
	printf("[NNDATA] Could not load data\n");
	return;
    }

    // Load training images
    printf("Rows: %d\n", trainingLegend.numberOfRows());
    for(unsigned int i = 0; i < trainingLegend.numberOfRows(); ++i)
    {
	shmea::GString label = shmea::GString::intTOstring(trainingLegend.getCell(i, 1).getInt());
	shmea::GString path = basePath + trainingLegend.getCell(i, 0).c_str();
	printf("[NNDATA] Loading %s\n", path.c_str());
	shmea::GPointer<shmea::Image> img(new shmea::Image());
	img->LoadPNG(path);

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
	    trainingData.addRow(img->flatten());
	    //printf("trainImages[%s].size() = %lu\n", label.c_str(), trainImages[label].size());
	}
    }

    // Load testing data
    for(unsigned int i = 0; i < testingLegend.numberOfRows(); ++i)
    {
	shmea::GString label = shmea::GString::intTOstring(testingLegend.getCell(i, 1).getInt());
	shmea::GString path = basePath + testingLegend.getCell(i, 0).c_str();
	printf("[NNDATA] Loading %s\n", path.c_str());
	shmea::GPointer<shmea::Image> img(new shmea::Image());
	img->LoadPNG(path);

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
	    testingData.addRow(img->flatten());
	    //printf("testImages[%s].size() = %lu\n", label.c_str(), testImages[label].size());
	}
    }

    // Set the loaded flag
    loaded = true;
}

const shmea::GPointer<shmea::Image> ImageInput::getTrainingImage(unsigned int row) const
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

const shmea::GPointer<shmea::Image> ImageInput::getTestingImage(unsigned int row) const
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

const shmea::GTable& ImageInput::getTrainingTable() const
{
    return trainingData;
}

const shmea::GTable& ImageInput::getTestingTable() const
{
    return testingData;
}
