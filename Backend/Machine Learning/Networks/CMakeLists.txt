set(Networks_src_files
	metanetwork.cpp
	metanetwork.h
	network.cpp
	network.h
	transformer_config.cpp
	transformer_config.h
	cross_validation.cpp
	cross_validation.h
	model_persistence.cpp
	tokenizer_artifacts.cpp
	checkpoint_persistence.cpp
	chiron_checkpoint.cpp
	chiron_serving.cpp
	chiron_generate.cpp
	sgd_dff.cpp
	sgd_rnn.cpp
	sgd_gru.cpp
	sgd_lstm.cpp
	sgd_cnn.cpp
	sgd_transformer.cpp
	transformer_train_detail.cpp
	atlas_optimizer.cpp
	vesta_optimizer.cpp
	helios_optimizer.cpp
	glades_thread_pool.cpp
	transformer_infer.cpp
	transformer_generate.cpp
	transformer_serving_layer.cpp
	transformer_serving_layer.h
	trainer.cpp
	trainer.h
	bayes.cpp
	bayes-optimizer.cpp
	transformer_ops.h
	ddp_comm.cpp
	ddp_comm.h
	gan.cpp
	gan.h
	hyperparameter_tuner.cpp
	hyperparameter_tuner.h
)
add_library(Networks ${Networks_src_files})
target_link_libraries(Networks pthread)

# CBLAS acceleration for ATLAS optimizer (auto-detected at root level)
if(GLADES_CBLAS_LIBRARIES)
	target_link_libraries(Networks ${GLADES_CBLAS_LIBRARIES})
endif()

# CUDA GPU offloading
if(GLADES_ENABLE_CUDA)
	add_subdirectory(cuda)
	# Use --whole-archive so static-archive symbols (e.g. gpu_init.cu's
	# initGlorotUniform / initNormal) are unconditionally pulled into
	# libNetworks.a -> libglades.so even though the linker's symbol-driven
	# extraction can miss them when the call site is itself in a static
	# archive being processed in the same pass.  Past failure mode:
	# `nm libglades.so | grep initGlorotUniform` empty, runtime dlsym fail
	# the moment ensureGpuState() ran on a deferred-init transformer.
	target_link_libraries(Networks
		"-Wl,--whole-archive"
		GladesCUDA
		"-Wl,--no-whole-archive")
	target_include_directories(Networks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
	target_include_directories(Networks PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	target_compile_definitions(Networks PRIVATE GLADES_HAVE_CUDA=1)
endif()

install(TARGETS Networks EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
