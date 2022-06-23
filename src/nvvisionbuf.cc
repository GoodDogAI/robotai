#include "nvvisionbuf.h"

#include <stdlib.h>
#include <iostream>

NVVisionBuf::NVVisionBuf(uint32_t size, uint32_t idx)
         :index(idx), is_queued(false)
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

NVVisionBuf::NVVisionBuf(uint32_t np, BufferPlaneFormat *fmt, uint32_t idx)
         :index(idx), n_planes(np), is_queued(false)
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


