// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#ifndef _UT_UNIT_TEST
#define _UT_UNIT_TEST

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr);

#define ASSERT(failmsg, predicate) \
	G_assert(__FILE__, __LINE__, failmsg, predicate)

#endif