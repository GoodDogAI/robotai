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


void VisionBuf::allocate() {
    for (uint32_t i = 0; i < n_planes; i++)
    {
        this->planes[i].data = (uint8_t *)malloc(this->planes[i].fmt.sizeimage);
    }
}

void VisionBuf::free() {
   for (uint32_t i = 0; i < n_planes; i++)
   {
       ::free(this->planes[i].data);
   }
}