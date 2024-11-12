add_subdirectory("Structure")
add_subdirectory("State")
add_subdirectory("GMath")
add_subdirectory("Networks")
add_subdirectory("DataObjects")

set(ML_src_files
	main.cpp
	main.h
)
add_library(ML ${ML_src_files})

#Link libraries
target_link_libraries(ML Networks MLStructure MLState DataObjects GMath)

install(TARGETS ML EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT gladesConfig DESTINATION share/glades/cmake)

export(TARGETS ML FILE gladesConfig.cmake)
