# Embed a text file as a C++ raw string.
#   cmake -DINPUT= -DOUTPUT= -DVAR=kName -P cmake/embed_text.cmake
if(NOT INPUT OR NOT OUTPUT OR NOT VAR)
  message(FATAL_ERROR "INPUT, OUTPUT, and VAR are required")
endif()
if(NOT EXISTS "${INPUT}")
  message(FATAL_ERROR "INPUT not found: ${INPUT}")
endif()
file(READ "${INPUT}" _body)
string(FIND "${_body}" ")hvaxhtml" _pos)
if(NOT _pos EQUAL -1)
  message(FATAL_ERROR "${INPUT} contains the raw-string delimiter )hvaxhtml")
endif()
get_filename_component(_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_dir}")
set(_hdr "#pragma once\ninline constexpr char ${VAR}[] = R\"hvaxhtml(")
set(_ftr ")hvaxhtml\";\n")
set(_temporary "${OUTPUT}.tmp")
file(WRITE "${_temporary}" "${_hdr}${_body}${_ftr}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_temporary}" "${OUTPUT}"
  COMMAND_ERROR_IS_FATAL ANY
)
file(REMOVE "${_temporary}")
