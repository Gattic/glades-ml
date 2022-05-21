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
#ifndef _GPOINTER
#define _GPOINTER

#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace shmea {

template <typename T>
class GPointer
{
protected:

	T* data;
	unsigned int refCount;
	pthread_mutex_t* refMutex;

public:

	explicit GPointer(T* newData = NULL)
	{
		data = newData;
		refCount = 1;//inc

		refMutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init(refMutex, NULL);
	}

	GPointer(const GPointer<T>& g2)
	{
		pthread_mutex_lock(g2.refMutex);
		data = g2.data;
		refCount = g2.refCount+1;//inc
		refMutex = g2.refMutex;
		pthread_mutex_unlock(g2.refMutex);
	}

	virtual ~GPointer()
	{
		reset();
	}

	void reset()
	{
		if(decrement() == 0)
		{
			delete data;
			data=NULL;
			refCount = 0;

			pthread_mutex_destroy(refMutex);
			if (refMutex)
				free(refMutex);
			refMutex=NULL;
		}
	}

	T* get() const
	{
		return data;
	}

	unsigned int decrement()
	{
		//pthread_mutex_lock(refMutex);
		--refCount; //dec
		//pthread_mutex_unlock(refMutex);

		return refCount;
	}

	T& operator*()
	{
		return *data;
	}

	T* operator->()
	{
		return data;
	}

	operator T*() const
	{
		return data;
	}

	operator bool() const
	{
		return (data!=NULL);
	}

	GPointer<T>& operator=(const GPointer<T>& g2)
	{
		if(this != &g2)
		{
			// All done?
			if((decrement() == 0) && (data != NULL))
			{
				delete data;
				refCount = 0;
			}

			pthread_mutex_lock(refMutex);
			data = g2.data;
			refCount = g2.refCount+1;
			refMutex = g2.refMutex;
			pthread_mutex_unlock(refMutex);
		}

		return *this;
	}
};
};

#endif
