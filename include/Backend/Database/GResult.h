#ifndef _GRESULT
#define _GRESULT

#include "GString.h"
#include <cassert>
#include <cstdio>
namespace shmea {
namespace result {
	enum Result
	{
		SUCCESS,
		ERROR_INSERTION_FAILURE,
		ERROR_OUT_OF_RANGE,
		ERROR_TYPE_ERROR,
		ERROR_NULL_POINTER,
	};
};

template <typename T>
class GResult
{
private:
	T value;
	result::Result result;
public:
	GResult(result::Result result) : result(result) {}
	GResult(T value) : result(result::SUCCESS), value(value) {}
	result::Result getResult()
	{
		return this->result;
	}
	T unwrap()
	{
		if (this->getResult() != result::SUCCESS) {
			fprintf(stderr, "Error: %s\n", this->format());
			assert(this->getResult() == result::SUCCESS);
		}
		return this->value;
	}
	const char* format() {
		switch (result) {
			case result::SUCCESS:
				return "SUCCESS";
			case result::ERROR_INSERTION_FAILURE:
				return "ERROR_INSERTION_FAILURE";
			case result::ERROR_OUT_OF_RANGE:
				return "ERROR_OUT_OF_RANGE";
			case result::ERROR_TYPE_ERROR:
				return "ERROR_TYPE_ERROR";
			case result::ERROR_NULL_POINTER:
				return "ERROR_NULL_POINTER";
		}
	}
};

template <>
class GResult<void>
{
private:
	result::Result result;
public:
	GResult<void>(result::Result result) : result(result) {}
	result::Result getResult()
	{
		return this->result;
	}
};
};

#endif // !_GRESULT