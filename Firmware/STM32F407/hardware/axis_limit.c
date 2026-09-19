#include "axis_limit.h"

void AxisLimit_Init(AxisLimit_t *limit, uint32_t maximum)
{
    if (limit == 0) return;
    limit->position = 0u;
    limit->maximum = maximum;
}

void AxisLimit_Reset(AxisLimit_t *limit)
{
    if (limit != 0) limit->position = 0u;
}

uint32_t AxisLimit_Allow(const AxisLimit_t *limit, uint8_t direction,
                         uint32_t requested)
{
    uint32_t available;
    if (limit == 0) return 0u;
    available = direction ? (limit->maximum - limit->position) : limit->position;
    return (requested < available) ? requested : available;
}

void AxisLimit_Commit(AxisLimit_t *limit, uint8_t direction, uint32_t pulses)
{
    uint32_t allowed;
    if (limit == 0) return;
    allowed = AxisLimit_Allow(limit, direction, pulses);
    if (direction) limit->position += allowed;
    else limit->position -= allowed;
}

uint32_t AxisLimit_GetPosition(const AxisLimit_t *limit)
{
    return (limit != 0) ? limit->position : 0u;
}
