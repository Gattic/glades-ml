set(GMath_src_files
	kmeans.cpp
	pca.cpp
	garch.cpp
	garch.h
	egarch.cpp
	egarch.h
	cmatrix.cpp
	cmatrix.h
	OHE.cpp
	OHE.h
	gmath.cpp
	gmath.h
	fft.cpp
	fft.h
	fisher_transform.cpp
	fisher_transform.h
	kelly.cpp
	kelly.h
	qp_solver.cpp
	qp_solver.h
	garch_utils.cpp
	garch_utils.h
	dwt.cpp
	dwt.h
	permutation_entropy.cpp
	permutation_entropy.h
	wasserstein.cpp
	wasserstein.h
)
add_library(GMath ${GMath_src_files})

#Link libraries
target_link_libraries(GMath ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS GMath EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
