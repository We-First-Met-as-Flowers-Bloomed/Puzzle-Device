#ifndef PULSE_AXIS_H
#define PULSE_AXIS_H

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint32_t target_pulses;
    uint32_t emitted_pulses;
    uint8_t direction;
    uint8_t running;
    uint8_t paused;
    uint8_t continuous;
    uint8_t complete;
} PulseAxis_t;

void PulseAxis_Init(PulseAxis_t *axis);
void PulseAxis_Start(PulseAxis_t *axis, uint32_t pulses, uint8_t direction);
void PulseAxis_StartContinuous(PulseAxis_t *axis, uint8_t direction);
void PulseAxis_OnPulse(PulseAxis_t *axis);
void PulseAxis_Pause(PulseAxis_t *axis);
void PulseAxis_Resume(PulseAxis_t *axis);
void PulseAxis_Abort(PulseAxis_t *axis);

uint32_t PulseAxis_GetEmitted(const PulseAxis_t *axis);
uint8_t PulseAxis_GetProgress(const PulseAxis_t *axis);
bool PulseAxis_IsRunning(const PulseAxis_t *axis);
bool PulseAxis_IsComplete(const PulseAxis_t *axis);
uint16_t PulseAxis_RpmToHalfPeriodTicks(uint32_t rpm, uint32_t pulses_per_rev,
                                        uint32_t timer_hz);

#endif
