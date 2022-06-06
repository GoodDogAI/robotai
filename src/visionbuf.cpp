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
        this->planes[i].data = NULL;
        this->planes[i].bytesused = 0;
        this->planes[i].mem_offset = 0;
        this->planes[i].length = 0;
        this->planes[i].fmt.sizeimage = size;
    }

    this->n_planes = 1;
}


void VisionBuf::allocate() {
    size_t len = 0;

    for (uint32_t i = 0; i < n_planes; i++)
    {
        len += planes[i].fmt.sizeimage;
    }

    this->len = len;
    this->addr = malloc(len);

    std::cout << "Allocing " << len << " bytes at " << this->addr << std::endl;
}

void VisionBuf::free() {
    ::free(this->addr);
}