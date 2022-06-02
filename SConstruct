env = Environment(CPPPATH=[".", "/usr/src/jetson_multimedia_api/include"],
                  LIBPATH = ["/usr/lib", "/usr/local/lib", "/usr/lib/aarch64-linux-gnu/tegra/"])

multimedia_api_classes = "/usr/src/jetson_multimedia_api/samples/common/classes"

#env.Object(Glob(f"{multimedia_api_classes}/*.cpp"))
env.Object("/usr/src/jetson_multimedia_api/samples/common/classes/NvVideoEncoder.cpp")

#env.Program("hello.cpp", LIBS=["realsense2", "nvbuf_utils", "v4l2"])