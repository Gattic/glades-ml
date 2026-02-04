set(PCATests_src_files
nn-test.cpp
nn-cv-test.cpp
nn-save-load-test.cpp
nn-benchmarks.cpp
pca-test.cpp
kmeans-test.cpp
bayes-test.cpp
bayes-optimizer-test.cpp
ohe-test.cpp
)
add_library(PCATests ${PCATests_src_files})
