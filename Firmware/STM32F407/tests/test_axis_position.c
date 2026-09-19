#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "axis_position.h"

int main(void)
{
    uint8_t direction = 0u;
    uint32_t pulses = 0u;

    assert(AxisPosition_CommandPulsesToEncoder(3200u) == 65536u);
    assert(AxisPosition_CommandPulsesToEncoder(24000u) == 491520u);
    assert(!AxisPosition_HasReached(1000, 66371, 3200u, 8u));
    assert(AxisPosition_HasReached(1000, 66372, 3200u, 8u));
    assert(AxisPosition_HasReached(1000, 66536, 3200u, 0u));
    assert(AxisPosition_HasReached(66536, 1164, 3200u, 8u));
    assert(!AxisPosition_HasReached(66536, 1165, 3200u, 8u));
    assert(AxisPosition_HasReached(-1000, 64536, 3200u, 0u));
    assert(AxisPosition_HasReached(1000, 1000, 0u, 8u));

    assert(AxisPosition_HasMovedInDirection(1000, 1200, 1u, 0u));
    assert(!AxisPosition_HasMovedInDirection(1000, 1000, 1u, 0u));
    assert(!AxisPosition_HasMovedInDirection(1000, 800, 1u, 0u));
    assert(AxisPosition_HasMovedInDirection(1000, 800, 0u, 0u));
    assert(AxisPosition_HasMovedInDirection(1000, 800, 1u, 1u));
    assert(!AxisPosition_HasMovedInDirection(1000, 1200, 1u, 1u));

    /* 20 command pulses = about 410 encoder units: remain in the deadband. */
    assert(!AxisPosition_SelectCorrection(400, 20u, 500u, 0u,
                                          &direction, &pulses));
    /* Positive X/Y encoder error is corrected in logical direction zero. */
    assert(AxisPosition_SelectCorrection(6554, 20u, 500u, 0u,
                                         &direction, &pulses));
    assert(direction == 0u && pulses == 320u);
    /* Negative X/Y error is corrected in logical direction one. */
    assert(AxisPosition_SelectCorrection(-6554, 20u, 500u, 0u,
                                         &direction, &pulses));
    assert(direction == 1u && pulses == 320u);
    /* R logical-positive encoder polarity is negative. */
    assert(AxisPosition_SelectCorrection(-6554, 20u, 500u, 1u,
                                         &direction, &pulses));
    assert(direction == 0u && pulses == 320u);
    /* Implausibly large residuals are bounded to one 500-pulse attempt. */
    assert(AxisPosition_SelectCorrection(65536, 20u, 500u, 0u,
                                         &direction, &pulses));
    assert(direction == 0u && pulses == 500u);

    puts("axis_position: PASS");
    return 0;
}
