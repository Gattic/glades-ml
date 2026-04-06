set(PCATests_src_files
nn-test.cpp
nn-cv-test.cpp
 "../../../Backend/Machine Learning/Networks/cross_validation.cpp"
nn-save-load-test.cpp
nn-mixed-precision-test.cpp
nn-benchmarks.cpp
transformer-serving-layer-test.cpp
prop-fuzz-test.cpp
parallel-test.cpp
pca-test.cpp
kmeans-test.cpp
bayes-test.cpp
bayes-optimizer-test.cpp
ohe-test.cpp
mapped-dataset-test.cpp
 "../../../Backend/Machine Learning/DataObjects/MappedMatrix.cpp"
 "../../../Backend/Machine Learning/DataObjects/MappedNumberInput.cpp"
ddp-test.cpp
 "../../../Backend/Machine Learning/DataObjects/DDPDataInputWrapper.cpp"
transformer-improvements-test.cpp
transformer-gradient-test.cpp
gpu-training-test.cpp
cnn-test.cpp
garch-test.cpp
egarch-test.cpp
gan-test.cpp
hyperparameter-tuner-test.cpp
atlas-test.cpp
atlas-bench.cpp
sfcka-test.cpp
)
add_library(PCATests ${PCATests_src_files})
