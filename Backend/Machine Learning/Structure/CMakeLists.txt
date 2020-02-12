set(MLStructure_src_files
	layerinfo.cpp
	layerinfo.h
	inputlayerinfo.cpp
	inputlayerinfo.h
	hiddenlayerinfo.cpp
	hiddenlayerinfo.h
	outputlayerinfo.cpp
	outputlayerinfo.h
	nninfo.cpp
	nninfo.h
)
add_library(MLStructure ${MLStructure_src_files})

#Link libraries
target_link_libraries(MLStructure)

install(TARGETS MLStructure EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT gladesConfig DESTINATION share/glades/cmake)

export(TARGETS MLStructure FILE gladesConfig.cmake)
