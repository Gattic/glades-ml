set(MLState_src_files
	Terminator.cpp
	Terminator.h
)
add_library(MLState ${MLState_src_files})

install(TARGETS MLState EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
