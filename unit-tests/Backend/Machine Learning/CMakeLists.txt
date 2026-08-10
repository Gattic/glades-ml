set(PCATests_src_files
nn-test.cpp
nn-cv-test.cpp
 "../../../Backend/Machine Learning/Networks/cross_validation.cpp"
nn-save-load-test.cpp
nn-mixed-precision-test.cpp
nn-benchmarks.cpp
transformer-gpu-bench.cpp
transformer-verification-test.cpp
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
atlas-alt-bench.cpp
vesta-test.cpp
helios-test.cpp
chiron-test.cpp
chiron-rank-cause-test.cpp
chiron-orbit-test.cpp
chiron-vitals-test.cpp
chiron-model-test.cpp
chiron-generate-test.cpp
sfcka-test.cpp
simd-parity-test.cpp
sfa-parity-test.cpp
sfa-bench-test.cpp
sampling-test.cpp
attention-backward-test.cpp
transformer-ops-test.cpp
transformer-kernels-test.cpp
numerical-edge-test.cpp
)
add_library(PCATests ${PCATests_src_files})

if(GLADES_BUILD_TRANSFORMER_INFER_FUZZER)
	add_executable(fuzz_transformer_infer EXCLUDE_FROM_ALL
		fuzz_transformer_infer.cpp
	)
	target_link_libraries(fuzz_transformer_infer
		glades
		shmea
	)
	target_include_directories(fuzz_transformer_infer PRIVATE
		"${CMAKE_SOURCE_DIR}/../Backend"
		"${CMAKE_SOURCE_DIR}/../include"
		"${CMAKE_SOURCE_DIR}/../services"
	)
	target_compile_options(fuzz_transformer_infer PRIVATE
		${GLADES_LIBFUZZER_SANITIZER_FLAGS}
		-fno-omit-frame-pointer
	)
	set_property(TARGET fuzz_transformer_infer APPEND_STRING PROPERTY LINK_FLAGS
		" ${GLADES_LIBFUZZER_SANITIZER_FLAGS}")
	if(GLADES_ENABLE_CUDA)
		set_target_properties(fuzz_transformer_infer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	endif()

	add_custom_target(fuzz_transformer_infer_smoke
		COMMAND $<TARGET_FILE:fuzz_transformer_infer> -runs=1 -seed=1
		DEPENDS fuzz_transformer_infer
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)
endif()
