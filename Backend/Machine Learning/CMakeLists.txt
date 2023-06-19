add_subdirectory("Structure")
add_subdirectory("State")
add_subdirectory("GMath")

#Threading lib
include(FindPkgConfig)
find_package(Threads REQUIRED)

set(ML_src_files
	glades.cpp
	glades.h
	metanetwork.cpp
	metanetwork.h
	network.cpp
	network.h
	bayes.cpp
)
add_library(ML ${ML_src_files})

#Link libraries
target_link_libraries(ML MLStructure MLState GMath ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ML EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT gladesConfig DESTINATION share/glades/cmake)

export(TARGETS ML FILE gladesConfig.cmake)
