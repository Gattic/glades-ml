set(DO_src_files
	DataInput.cpp
	NumberInput.cpp
	ImageInput.cpp
)
add_library(DataObjects ${DO_src_files})

install(TARGETS DataObjects EXPORT gladesConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT gladesConfig DESTINATION share/glades/cmake)

export(TARGETS DataObjects FILE gladesConfig.cmake)
