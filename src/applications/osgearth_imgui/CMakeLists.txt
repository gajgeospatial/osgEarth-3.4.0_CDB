set(TARGET_LIBRARIES_VARS OSGDB_LIBRARY OSGUTIL_LIBRARY OSG_LIBRARY OSGVIEWER_LIBRARY OPENTHREADS_LIBRARY)

set(TARGET_SRC osgearth_imgui.cpp)
SETUP_IMGUI_APPLICATION(osgearth_imgui)
