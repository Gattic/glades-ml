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
#ifndef _GPOINTER
#define _GPOINTER

#include "GDeleter.h"

#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>

namespace shmea {

template <typename T, void(*Deleter)(T*) = default_deleter<T>>
class GPointer
{
private:
	template <typename U, void(*UDeleter)(U*)>
	friend class GPointer;

	std::shared_ptr<T> data;

public:
	GPointer() = default;

	// Boundary adoption constructor. First-party call sites should prefer
	// make_gpointer(); foreign APIs may use this to adopt a result immediately.
	explicit GPointer(T* newData)
		: data(newData, Deleter)
	{
	}

	explicit GPointer(std::shared_ptr<T> owner)
		: data(std::move(owner))
	{
	}

	GPointer(const GPointer&) = default;
	GPointer(GPointer&&) noexcept = default;
	GPointer& operator=(const GPointer&) = default;
	GPointer& operator=(GPointer&&) noexcept = default;
	~GPointer() = default;

	template <typename U, void(*UDeleter)(U*)>
	requires std::convertible_to<U*, T*>
	GPointer(const GPointer<U, UDeleter>& other)
		: data(other.data)
	{
	}

	template <typename U, void(*UDeleter)(U*)>
	requires std::convertible_to<U*, T*>
	GPointer& operator=(const GPointer<U, UDeleter>& other)
	{
		data = other.data;
		return *this;
	}

	void reset() noexcept { data.reset(); }
	T* get() const noexcept { return data.get(); }
	long use_count() const noexcept { return data.use_count(); }

	T& operator*() const { return *data; }
	T* operator->() const noexcept { return data.get(); }
	T& operator[](std::size_t index) const { return data.get()[index]; }
	explicit operator bool() const noexcept { return static_cast<bool>(data); }

	bool operator==(const GPointer& other) const noexcept { return data == other.data; }
	bool operator!=(const GPointer& other) const noexcept { return data != other.data; }

	GPointer& copy(const GPointer& other)
	{
		data = other.data;
		return *this;
	}
};

template <typename T, typename... Args>
GPointer<T> make_gpointer(Args&&... args)
{
	return GPointer<T>(std::make_shared<T>(std::forward<Args>(args)...));
}

} // namespace shmea

#endif
