#include "visionbuf.h"

#include <stdlib.h>
#include <iostream>

VisionBuf::VisionBuf(uint32_t size, uint32_t index)
         :index(index)
{
    uint32_t i;

    n_planes = 1;
    for (i = 0; i < n_planes; i++)
    {
        this->planes[i].fd = -1;
        this->planes[i].mem_offset = 0;
        this->planes[i].length = 0;

        this->planes[i].data = NULL;
        this->planes[i].bytesused = 0;
        this->planes[i].fmt.sizeimage = size;
    }

    this->n_planes = 1;
}

VisionBuf::VisionBuf(uint32_t n_planes, BufferPlaneFormat *fmt, uint32_t index)
         :index(index), n_planes(n_planes)
{
    for (uint32_t i = 0; i < n_planes; i++)
    {
        this->planes[i].fd = -1;
        this->planes[i].mem_offset = 0;
        this->planes[i].length = 0;

        this->planes[i].data = NULL;
        this->planes[i].bytesused = 0;
        this->planes[i].fmt = fmt[i];
    }
}


void VisionBuf::map(uint32_t index, v4l2_buf_type type) {
    // struct v4l2_buffer buffer = {0};
    // struct v4l2_plane planes[MAX_PLANES];

    // buffer.type = type;
    // buffer.memory = V4L2_MEMORY_MMAP;
    // buffer.index = i;

    // check

    // for (uint32_t i = 0; i < n_planes; i++)
    // {
    //     this->planes[i].data = (uint8_t *)malloc(this->planes[i].fmt.sizeimage);
    // }
}

void VisionBuf::unmap() {
//    for (uint32_t i = 0; i < n_planes; i++)
//    {
//        ::free(this->planes[i].data);
//    }
}