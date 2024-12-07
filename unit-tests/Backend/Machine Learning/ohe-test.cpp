// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.

#include "ohe-test.h"
#include "Backend/Database/GTable.h"

#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/GMath/OHE.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GVector.h"


// === Primary unit testing function ===
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void OHEUnitTest() {
    // Check Initialization of object
    glades::OHE ohe;
    G_assert(__FILE__, __LINE__, "Initialization failed: OHE size is not 0", ohe.size() == 0);

    //  Add strings and verify
    ohe.addString("cat");
    ohe.addString("dog");
    ohe.addString("cat"); // Duplicate

// Test Duplicate string handling
    G_assert(__FILE__, __LINE__, "Failed to add strings or handle duplicates correctly", ohe.size() == 2);
    G_assert(__FILE__, __LINE__, "OHE does not contain 'cat'", ohe.contains("cat"));
    G_assert(__FILE__, __LINE__, "OHE does not contain 'dog'", ohe.contains("dog"));

    //  One-hot encoding vector for strings
    std::vector<float> catEncoding = ohe["cat"];
    std::vector<float> dogEncoding = ohe["dog"];

    G_assert(__FILE__, __LINE__, "One-hot encoding size mismatch", catEncoding.size() == 2);
    G_assert(__FILE__, __LINE__, "One-hot encoding size mismatch", dogEncoding.size() == 2);
    G_assert(__FILE__, __LINE__, "One-hot encoding mismatch for 'cat'", catEncoding[0] == 0.99f && catEncoding[1] == 0.01f);
    G_assert(__FILE__, __LINE__, "One-hot encoding mismatch for 'dog'", dogEncoding[0] == 0.01f && dogEncoding[1] == 0.99f);


    // Index retrieval for strings
    G_assert(__FILE__, __LINE__, "Index retrieval for 'cat' failed", ohe.indexAt("cat") == 0);
    G_assert(__FILE__, __LINE__, "Index retrieval for 'dog' failed", ohe.indexAt("dog") == 1);
    G_assert(__FILE__, __LINE__, "Index retrieval for non-existing string failed", ohe.indexAt("bird") == -1);


      
    // Test mapFeatureSpace
  	shmea::GString sampleData( "First,Last,Age\nMickey,Mouse,100\nDonald,Duck,99\n" );
	shmea::GTable sampleDataTable( sampleData, ',', shmea::GTable::TYPE_STRING );

    ohe.mapFeatureSpace(sampleDataTable, 0); // Column 0

    G_assert(__FILE__, __LINE__, "Failed to get string 'Mickey'", ohe.contains("Mickey"));
    G_assert(__FILE__, __LINE__, "Failed to get string 'Donald'", ohe.contains("Donald"));
    
    
    // Test Print features (visual validation or mock testing for printf)
    ohe.printFeatures(); // Expected output: "[OHE] cat dog Mickey Donald "

  
    printf("OHEUnitTest completed successfully.\n");

}


