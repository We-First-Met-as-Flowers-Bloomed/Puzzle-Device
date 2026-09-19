#include <assert.h>
#include <stdint.h>
#include <stdio.h>

uint32_t Task_CameraPulseToMechanical(uint8_t axis, uint32_t camera_pulse);

int main(void)
{
    assert(Task_CameraPulseToMechanical(0u, 0u) == 0u);
    assert(Task_CameraPulseToMechanical(0u, 12000u) == 14750u);
    assert(Task_CameraPulseToMechanical(0u, 24000u) == 29500u);
    assert(Task_CameraPulseToMechanical(0u, 24001u) == 29500u);
    assert(Task_CameraPulseToMechanical(1u, 8350u) == 8400u);
    assert(Task_CameraPulseToMechanical(1u, 16700u) == 16800u);
    assert(Task_CameraPulseToMechanical(1u, 16701u) == 16800u);
    puts("camera_axis_mapping: PASS");
    return 0;
}
