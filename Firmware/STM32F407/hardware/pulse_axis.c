#include "pulse_axis.h"

#include <string.h>

uint16_t PulseAxis_RpmToHalfPeriodTicks(uint32_t rpm, uint32_t pulses_per_rev,
                                        uint32_t timer_hz)
{
    uint64_t denominator;
    uint64_t ticks;

    if ((rpm == 0u) || (pulses_per_rev == 0u) || (timer_hz == 0u))
    {
        return 0u;
    }
    denominator = 2ull * (uint64_t)pulses_per_rev * (uint64_t)rpm;
    ticks = (60ull * (uint64_t)timer_hz) / denominator;
    if (ticks == 0ull) return 1u;
    if (ticks > 65535ull) return 65535u;
    return (uint16_t)ticks;
}

void PulseAxis_Init(PulseAxis_t *axis)
{
    if (axis != 0)
    {
        memset(axis, 0, sizeof(*axis));
    }
}

void PulseAxis_Start(PulseAxis_t *axis, uint32_t pulses, uint8_t direction)
{
    if (axis == 0)
    {
        return;
    }
    axis->target_pulses = pulses;
    axis->emitted_pulses = 0u;
    axis->direction = direction ? 1u : 0u;
    axis->paused = 0u;
    axis->continuous = 0u;
    axis->complete = (pulses == 0u) ? 1u : 0u;
    axis->running = (pulses == 0u) ? 0u : 1u;
}

void PulseAxis_StartContinuous(PulseAxis_t *axis, uint8_t direction)
{
    if (axis == 0)
    {
        return;
    }
    axis->target_pulses = 0u;
    axis->emitted_pulses = 0u;
    axis->direction = direction ? 1u : 0u;
    axis->running = 1u;
    axis->paused = 0u;
    axis->continuous = 1u;
    axis->complete = 0u;
}

void PulseAxis_OnPulse(PulseAxis_t *axis)
{
    if ((axis == 0) || !axis->running || axis->paused)
    {
        return;
    }
    ++axis->emitted_pulses;
    if (!axis->continuous && (axis->emitted_pulses >= axis->target_pulses))
    {
        axis->emitted_pulses = axis->target_pulses;
        axis->running = 0u;
        axis->complete = 1u;
    }
}

void PulseAxis_Pause(PulseAxis_t *axis)
{
    if ((axis != 0) && axis->running)
    {
        axis->paused = 1u;
    }
}

void PulseAxis_Resume(PulseAxis_t *axis)
{
    if ((axis != 0) && axis->running)
    {
        axis->paused = 0u;
    }
}

void PulseAxis_Abort(PulseAxis_t *axis)
{
    if (axis == 0)
    {
        return;
    }
    axis->running = 0u;
    axis->paused = 0u;
    axis->continuous = 0u;
    axis->complete = 0u;
}

uint32_t PulseAxis_GetEmitted(const PulseAxis_t *axis)
{
    return (axis == 0) ? 0u : axis->emitted_pulses;
}

uint8_t PulseAxis_GetProgress(const PulseAxis_t *axis)
{
    uint32_t percent;

    if ((axis == 0) || axis->continuous || (axis->target_pulses == 0u))
    {
        return 0u;
    }
    percent = (axis->emitted_pulses * 100u) / axis->target_pulses;
    return (percent > 100u) ? 100u : (uint8_t)percent;
}

bool PulseAxis_IsRunning(const PulseAxis_t *axis)
{
    return (axis != 0) && (axis->running != 0u) && (axis->paused == 0u);
}

bool PulseAxis_IsComplete(const PulseAxis_t *axis)
{
    return (axis != 0) && (axis->complete != 0u);
}
