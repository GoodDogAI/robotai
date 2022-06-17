// camerad talks to the Intel RealSense camera and runs a visionipc server to send the frames to other services

#include <chrono>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"

int main(int argc, char * argv[]) {
    VisionIpcServer server("camerad");
    server.create_buffers(VISION_STREAM_ROAD, 1, false, 100, 100);
    server.start_listener();
}