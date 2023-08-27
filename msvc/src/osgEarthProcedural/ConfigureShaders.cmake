# configureshaders.cmake.in

set(source_dir      "N:/Development/Dev_Base/osgearth-3.4.0/src/osgEarthProcedural")
set(bin_dir         "N:/Development/Dev_Base/osgearth-3.4.0/msvc/src/osgEarthProcedural")
set(glsl_files      "Procedural.Vegetation.glsl;Procedural.TextureSplatting.glsl;Procedural.HexTiling.glsl")
set(template_file   "ProceduralShaders.cpp.in")
set(output_cpp_file "N:/Development/Dev_Base/osgearth-3.4.0/msvc/src/osgEarthProcedural/AutoGenShaders.cpp")

# modify the contents for inlining; replace input with output (var: file)
# i.e., the file name (in the form ) gets replaced with the
# actual contents of the named file and then processed a bit.
foreach(file ${glsl_files})

    # read the file into 'contents':
    file(READ ${source_dir}/${file} contents)

    # compress whitespace.
    # "contents" must be quoted, otherwise semicolons get dropped for some reason.
    string(REGEX REPLACE "\n\n+" "\n" tempString "${contents}")
    
    set(${file} "\nR\"(${tempString})\"")

endforeach(file)

# send the processed glsl_files to the next template to create the
# shader source file.
configure_file(
	${source_dir}/${template_file}
	${output_cpp_file}
	@ONLY )