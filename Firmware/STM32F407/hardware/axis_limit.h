#ifndef AXIS_LIMIT_H
#define AXIS_LIMIT_H

#include <stdint.h>

typedef struct
{
    uint32_t position;
    uint32_t maximum;
} AxisLimit_t;

void AxisLimit_Init(AxisLimit_t *limit, uint32_t maximum);
void AxisLimit_Reset(AxisLimit_t *limit);
uint32_t AxisLimit_Allow(const AxisLimit_t *limit, uint8_t direction,
                         uint32_t requested);
void AxisLimit_Commit(AxisLimit_t *limit, uint8_t direction, uint32_t pulses);
uint32_t AxisLimit_GetPosition(const AxisLimit_t *limit);

#endif
