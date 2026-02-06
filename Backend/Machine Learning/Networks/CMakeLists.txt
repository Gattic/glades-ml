set(Networks_src_files
	metanetwork.cpp
	metanetwork.h
	network.cpp
	network.h
	cross_validation.cpp
	cross_validation.h
	model_persistence.cpp
	tokenizer_artifacts.cpp
	checkpoint_persistence.cpp
	sgd_dff.cpp
	sgd_rnn.cpp
	sgd_gru.cpp
	sgd_lstm.cpp
	sgd_transformer.cpp
	transformer_infer.cpp
	transformer_generate.cpp
	transformer_serving_layer.cpp
	transformer_serving_layer.h
	trainer.cpp
	trainer.h
	bayes.cpp
	bayes-optimizer.cpp
	transformer_ops.h
)
add_library(Networks ${Networks_src_files})

install(TARGETS Networks EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
