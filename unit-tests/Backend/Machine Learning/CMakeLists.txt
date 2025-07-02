set(PCATests_src_files
nn-test.cpp
nn-test-cf.cpp
pca-test.cpp
kmeans-test.cpp
bayes-test.cpp
bayes-optimizer-test.cpp
ohe-test.cpp
)
add_library(PCATests ${PCATests_src_files})
