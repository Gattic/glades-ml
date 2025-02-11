set(Networks_src_files
	metanetwork.cpp
	metanetwork.h
	network.cpp
	network.h
	bayes.cpp
	bayes-optimizer.cpp
)
add_library(Networks ${Networks_src_files})

install(TARGETS Networks EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT gladesConfig DESTINATION share/glades/cmake)

export(TARGETS Networks FILE gladesConfig.cmake)
