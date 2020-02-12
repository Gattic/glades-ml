set(MLState_src_files
	layer.cpp
	layer.h
	edge.cpp
	edge.h
	node.cpp
	node.h
	NetworkState.cpp
	NetworkState.h
	LayerBuilder.cpp
	LayerBuilder.h
	Terminator.cpp
	Terminator.h
)
add_library(MLState ${MLState_src_files})

install(TARGETS MLState EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT gladesConfig DESTINATION share/glades/cmake)

export(TARGETS MLState FILE gladesConfig.cmake)
