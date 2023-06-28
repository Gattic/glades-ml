#ifndef GDELETER_H_
#define GDELETER_H_

namespace shmea
{
template<typename T>
void default_deleter(T* mem)
{
	delete mem;
}
template<typename T>
void array_deleter(T* mem)
{
	delete [] mem;
}
}

#endif // !GDELETER_H_