#pragma once

#include <stddef.h>
#include <stdint.h>

#include <linux/videodev2.h>
#include <libv4l2.h>

#define MAX_PLANES 3

typedef struct
{
    uint32_t width;             
    uint32_t height;           

    uint32_t bytesperpixel;     

    uint32_t stride;           
    uint32_t sizeimage;    
} BufferPlaneFormat;

typedef struct
{
    BufferPlaneFormat fmt;   

    uint8_t *data;        
    uint32_t bytesused;      

    int fd;                     
    uint32_t mem_offset;       
    uint32_t length; 
} BufferPlane;

class VisionBuf {
 public:
  BufferPlane planes[MAX_PLANES];  
  uint32_t n_planes; 
  const uint32_t index;     

  // Single plane for just storing output data
  VisionBuf(uint32_t size, uint32_t index);   

  // Multi plane color buffer
  VisionBuf(uint32_t n_planes, BufferPlaneFormat *fmt, uint32_t index);                             

  void allocate();
  void free();
};

