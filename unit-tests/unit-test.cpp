// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "unit-test.h"

void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)
{
	if(!expr)
	{
		printf("Unit Test Error %s[%d]: %s\n", fileName, lineNo, failureMsg);
		assert (expr);
	}
	else
	{
		printf("Unit Test Success %s[%d]\n", fileName, lineNo);
	}
}
