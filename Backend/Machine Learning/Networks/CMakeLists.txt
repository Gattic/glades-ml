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
	target_link_libraries(Networks GladesCUDA)
	target_include_directories(Networks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
	target_include_directories(Networks PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	target_compile_definitions(Networks PRIVATE GLADES_HAVE_CUDA=1)
endif()

install(TARGETS Networks EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
