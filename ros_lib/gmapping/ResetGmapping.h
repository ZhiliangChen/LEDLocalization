#ifndef _ROS_SERVICE_ResetGmapping_h
#define _ROS_SERVICE_ResetGmapping_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace gmapping
{

static const char RESETGMAPPING[] = "gmapping/ResetGmapping";

  class ResetGmappingRequest : public ros::Msg
  {
    public:
      typedef int32_t _a_type;
      _a_type a;

    ResetGmappingRequest():
      a(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        int32_t real;
        uint32_t base;
      } u_a;
      u_a.real = this->a;
      *(outbuffer + offset + 0) = (u_a.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_a.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_a.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_a.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->a);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        int32_t real;
        uint32_t base;
      } u_a;
      u_a.base = 0;
      u_a.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_a.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_a.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_a.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->a = u_a.real;
      offset += sizeof(this->a);
     return offset;
    }

    const char * getType(){ return RESETGMAPPING; };
    const char * getMD5(){ return "5c9fb1a886e81e3162a5c87bf55c072b"; };

  };

  class ResetGmappingResponse : public ros::Msg
  {
    public:
      typedef int32_t _rescode_type;
      _rescode_type rescode;

    ResetGmappingResponse():
      rescode(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        int32_t real;
        uint32_t base;
      } u_rescode;
      u_rescode.real = this->rescode;
      *(outbuffer + offset + 0) = (u_rescode.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_rescode.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_rescode.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_rescode.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->rescode);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        int32_t real;
        uint32_t base;
      } u_rescode;
      u_rescode.base = 0;
      u_rescode.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_rescode.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_rescode.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_rescode.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->rescode = u_rescode.real;
      offset += sizeof(this->rescode);
     return offset;
    }

    const char * getType(){ return RESETGMAPPING; };
    const char * getMD5(){ return "38f8ed02a219ad7121fc12c79e529a98"; };

  };

  class ResetGmapping {
    public:
    typedef ResetGmappingRequest Request;
    typedef ResetGmappingResponse Response;
  };

}
#endif
