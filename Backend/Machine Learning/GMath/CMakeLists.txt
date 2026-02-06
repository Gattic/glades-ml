set(GMath_src_files
	kmeans.cpp
	pca.cpp
	cmatrix.cpp
	cmatrix.h
	OHE.cpp
	OHE.h
	gmath.cpp
	gmath.h
)
add_library(GMath ${GMath_src_files})

#Link libraries
target_link_libraries(GMath ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS GMath EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
