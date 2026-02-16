set(DO_src_files
	DataInput.cpp
	NumberInput.cpp
	ImageInput.cpp
	TokenInput.cpp
	tabular_preprocessing.cpp
	MappedMatrix.cpp
	MappedNumberInput.cpp
	DDPDataInputWrapper.cpp
)
add_library(DataObjects ${DO_src_files})

install(TARGETS DataObjects EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
