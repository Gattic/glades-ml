// Copyright 2024 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#ifndef GVECTOR_H_
#define GVECTOR_H_

// TODO: REPLACE ASSERT WITH GRESULT
#include "GPointer.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace shmea
{
template<typename T>
class GVector
{
public:
	// TODO: Implement custom iterators
	typedef shmea::GPointer<T> iterator;
	typedef const shmea::GPointer<T> const_iterator;
	typedef unsigned int size_type;
private:
	size_type m_size;
	size_type m_capacity;
	shmea::GPointer<T, array_deleter<T> > m_data;
public:
	const static unsigned int npos = -1;

	GVector() : m_size(0), m_capacity(0), m_data(0) {}

	GVector(size_type capacity) :
		m_size(0),
		m_capacity(capacity),
		m_data(capacity > 0 ? new T[capacity] : NULL) {}

	GVector(size_type capacity, const T& value) :
		m_size(0),
		m_capacity(capacity),
		m_data(new T[capacity])
	{
		for(size_type i = 0; i < capacity; ++i)
			this->push_back(value);
	}
	GVector(const GVector& value) :
		m_size(0),
		m_capacity(value.m_capacity),
		m_data(new T[value.m_capacity])
	{
		for (size_type i = 0; i < value.m_size; i++)
			this->push_back(value[i]);
	}
	virtual ~GVector() 
	{ 
		clear();
	}

	iterator begin() { return m_data; }
	const_iterator cbegin() const { return m_data; }
	iterator end() { return &m_data[this->size()]; }
	const_iterator cend() const { return &m_data[this->size()]; }

	size_type max_size() const { return 0 - 1; }
	size_type size() const { return m_size; }
	size_type capacity() const { return m_capacity; }
	bool empty() const { return (m_size == 0 || !m_data); }

	// Raw pointer access (contiguous storage).
	// NOTE: Returned pointer is invalidated by reserve()/expand()/clear()/assignment.
	T* data() { return m_data.get(); }
	const T* data() const { return m_data.get(); }

	void reserve(size_type new_cap)
	{
		if (new_cap <= m_capacity) return;

		shmea::GPointer<T, array_deleter<T> > newBuffer(new T[new_cap]);

		for (unsigned int i = 0; i < m_size; i++)
			newBuffer[i] = m_data[i];

		m_capacity = new_cap;
		m_data = newBuffer;
	}
	void clear() 
	{ 
		if (m_data)
		{
			m_size = 0;
			// Let the GPointer handle deletion when it goes out of scope
			m_data.reset();
			m_capacity = 0;
		}
	}
	void push_back(T newValue)
	{
		if (m_size >= m_capacity) 
			this->expand();
		
		if (!m_data)
			return;

		m_data[m_size] = newValue;
		++m_size;
	}
	T pop_back() 
	{ 
		if (!m_data || m_size == 0)
			return T();
		return m_data[--m_size]; 
	}
	void erase(size_type idx)
	{
		if (empty())
			return;

		assert(idx <= m_size);
		if (m_size == 1)
		{
			this->clear();
			return;
		}

		if (idx == m_size)
		{
			this->pop_back();
		}
		else
		{
			 // Manual element-wise copy instead of memmove
			for (size_type i = idx; i < m_size - 1; ++i)
			{
				m_data[i] = m_data[i+1];
			}
			m_size--;
		}
	}

	void insert(size_type idx, T value)
	{
		assert(idx <= m_size);
		if (idx == m_size)
		{
			this->push_back(value);
		}
		else
		{
			if (m_size >= m_capacity)
				this->expand();

			// Manual element-wise copy instead of memmove
			for (size_type i = m_size; i > idx; --i)
			{
				m_data[i] = m_data[i-1];
			}
			m_data[idx] = value;
			m_size++;
		}
	}

	T& at(size_type idx)
	{
		assert(idx < m_size);
		return m_data[idx];
	}
	const T& at(size_type idx) const
	{
		assert(idx < m_size);
		return m_data[idx];
	}
	T& operator[](size_type idx) { return at(idx); }
	const T& operator[](size_type idx) const { return at(idx); }

	bool contains(const T& value) const
	{
		for (size_type i = 0; i < m_size; ++i)
		{
			if (valuesEqual(m_data[i], value))
				return true;
		}
		return false;
	}

	size_type find(const T& value) const
	{
		for (size_type i = 0; i < m_size; ++i)
		{
			if (valuesEqual(m_data[i], value))
				return i;
		}
		return npos;
	}

	GVector& operator=(const GVector& other)
	{
		if (this != &other)
		{
			// Create new array and copy data before clearing old one
			T* newArray = new T[other.m_capacity];
			for (size_type i = 0; i < other.m_size; ++i)
			{
				new (&newArray[i]) T(other.m_data[i]);
			}
			
			// Only after new data is ready, clear old data
			clear();
			m_size = other.m_size;
			m_capacity = other.m_capacity;
			m_data = shmea::GPointer<T, array_deleter<T> >(newArray);
		}
		return *this;
	}

private:
	// Helper: default equality uses operator==
	template<typename X>
	static bool valuesEqual(const X& lhs, const X& rhs)
	{
		return lhs == rhs;
	}

	// Helper overload: if T is a GPointer<U, D>, compare pointed values when both non-null
	template<typename U, void(*D)(U*)>
	static bool valuesEqual(const shmea::GPointer<U, D>& lhs, const shmea::GPointer<U, D>& rhs)
	{
		if (!lhs && !rhs) return true;
		if (!lhs || !rhs) return false;
		return (*lhs) == (*rhs);
	}

	void expand()
	{
		size_type newCap = (m_capacity == 0) ? 1 : m_capacity * 2;
		T* newArray = new T[newCap];
		
		// Copy existing elements before modifying m_data
		if (m_data && m_size > 0)
		{
			for (size_type i = 0; i < m_size; ++i)
			{
				// Use copy constructor
				new (&newArray[i]) T(m_data[i]);
			}
		}

		// Create new pointer and only then release old one
		shmea::GPointer<T, array_deleter<T> > newBuffer(newArray);
		m_data.reset();  // Reset after new data is ready
		m_data = newBuffer;
		m_capacity = newCap;
	}
};

typedef GVector<GVector<float> > GMatrix;

// FLOAT HELPERS

inline static GVector<float> vectorStandardize(const GVector<float>& vec)
{
    GVector<float> newVec(vec.size(), 0.0f);

    // 1) If there's no data, nothing to do
    if (vec.size() == 0)
        return newVec;

    // Find min & max via first pass
    float xMin = 0.0f;
    float xMax = 0.0f;

    bool firstNumericValue = true;
    for (unsigned int r = 0; r < vec.size(); ++r)
    {
        const float& cell = vec[r];

        // Update xMin, xMax
        if (firstNumericValue)
        {
            xMin = cell;
            xMax = cell;
            firstNumericValue = false;
        }
        else
        {
            if (cell < xMin) xMin = cell;
            if (cell > xMax) xMax = cell;
        }
    }

    // 3) Compute xRange
    float xRange = xMax - xMin;
    if (xRange == 0.0f)
        return vec; // All values are the same => no transformation needed

    // 4) Second pass: normalize + shift in-place
    for (unsigned int r = 0; r < vec.size(); ++r)
    {
        float cell = vec[r];

        // Scale from [xMin..xMax] to [0..1], then shift => [-0.5..+0.5]
        cell = ((cell - xMin) / xRange) - 0.5f;

        newVec[r] = cell;
    }

    return newVec;
}

// GVector print function
inline static void printVector(const GVector<float>& vec)
{
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		if(i == vec.size() - 1)
		    printf("%f", vec[i]);
		else
		    printf("%f, ", vec[i]);
	}

	printf("\n");

}

// GMatrix print function
inline static void printMatrix(const GMatrix& matrix)
{
	for (unsigned int i = 0; i < matrix.size(); i++)
	{
		for (unsigned int j = 0; j < matrix[i].size(); j++)
		{
			if(j == matrix[i].size() - 1)
			    printf("%f", matrix[i][j]);
			else
			    printf("%f, ", matrix[i][j]);
		}
		printf("\n");
	}

}

};

#endif // !GVECTOR_H_
