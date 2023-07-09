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

    trainingData = shmea::GTable(trainFName, ',', importType);
    testingData = shmea::GTable(testFName, ',', importType);

    if ((trainingData.numberOfRows() == 0) || (testingData.numberOfRows() == 0))
    {
	printf("[NNDATA] Could not load data\n");
	return;
    }

    // Load training images
    for(unsigned int i = 0; i < trainingData.numberOfRows(); ++i)
    {
	shmea::GString label = shmea::GString::intTOstring(trainingData.getCell(i, 1).getInt());
	shmea::GString path = basePath + trainingData.getCell(i, 0).c_str();
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
	    //printf("trainImages[%s].size() = %lu\n", label.c_str(), trainImages[label].size());
	}
    }

    // Load testing data
    for(unsigned int i = 0; i < testingData.numberOfRows(); ++i)
    {
	shmea::GString label = shmea::GString::intTOstring(testingData.getCell(i, 1).getInt());
	shmea::GString path = basePath + testingData.getCell(i, 0).c_str();
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
	    //printf("testImages[%s].size() = %lu\n", label.c_str(), trainImages[label].size());
	}
    }

    // Set the loaded flag
    loaded = true;
}

const shmea::GPointer<shmea::Image> ImageInput::getTrainingImage(unsigned int row) const
{
    if(row >= trainingData.numberOfRows())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    shmea::GString label = shmea::GString::intTOstring(trainingData.getCell(row, 1).getInt());
    shmea::GString fname = "datasets/images/" + name + "/" + trainingData.getCell(row, 0).c_str();

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
    if(row >= testingData.numberOfRows())
	return shmea::GPointer<shmea::Image>(new shmea::Image());

    shmea::GString label = shmea::GString::intTOstring(testingData.getCell(row, 1).getInt());
    shmea::GString fname = "datasets/images/" + name + "/" + testingData.getCell(row, 0).c_str();

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

shmea::GTable ImageInput::getTrainingTable() const
{
    //
}

shmea::GTable ImageInput::getTestingTable() const
{
    //
}
