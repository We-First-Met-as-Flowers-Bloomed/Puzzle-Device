#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "pulse_axis.h"

int main(void)
{
    PulseAxis_t axis;
    uint32_t i;

    PulseAxis_Init(&axis);
    assert(PulseAxis_RpmToHalfPeriodTicks(300u, 3200u, 1000000u) == 31u);
    assert(PulseAxis_RpmToHalfPeriodTicks(0u, 3200u, 1000000u) == 0u);
    assert(PulseAxis_RpmToHalfPeriodTicks(300u, 0u, 1000000u) == 0u);
    PulseAxis_Start(&axis, 3200u, 1u);
    assert(PulseAxis_IsRunning(&axis));

    for (i = 0u; i < 1600u; ++i)
    {
        PulseAxis_OnPulse(&axis);
    }
    assert(PulseAxis_GetEmitted(&axis) == 1600u);
    assert(PulseAxis_GetProgress(&axis) == 50u);

    PulseAxis_Pause(&axis);
    PulseAxis_OnPulse(&axis);
    assert(PulseAxis_GetEmitted(&axis) == 1600u);
    PulseAxis_Resume(&axis);

    for (i = 1600u; i < 3200u; ++i)
    {
        PulseAxis_OnPulse(&axis);
    }
    assert(PulseAxis_IsComplete(&axis));
    assert(PulseAxis_GetEmitted(&axis) == 3200u);
    assert(PulseAxis_GetProgress(&axis) == 100u);

    PulseAxis_StartContinuous(&axis, 0u);
    for (i = 0u; i < 10u; ++i)
    {
        PulseAxis_OnPulse(&axis);
    }
    assert(PulseAxis_IsRunning(&axis));
    PulseAxis_Abort(&axis);
    assert(!PulseAxis_IsRunning(&axis));
    assert(!PulseAxis_IsComplete(&axis));

    puts("pulse_axis: PASS");
    return 0;
}
