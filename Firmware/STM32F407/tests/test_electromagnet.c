#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "electromagnet.h"

typedef struct
{
    uint8_t forward;
    uint8_t reverse;
    uint16_t compare;
} Output_t;

static Output_t outputs[2];

static void capture(uint8_t channel, uint8_t forward, uint8_t reverse,
                    uint16_t compare)
{
    outputs[channel].forward = forward;
    outputs[channel].reverse = reverse;
    outputs[channel].compare = compare;
}

int main(void)
{
    assert(Electromagnet_DutyToCompare(20000u, 0u) == 0u);
    assert(Electromagnet_DutyToCompare(20000u, 5000u) == 10000u);
    assert(Electromagnet_DutyToCompare(20000u, 10000u) == 20000u);
    assert(Electromagnet_DutyToCompare(20000u, 12000u) == 20000u);

    Electromagnet_Init(capture);
    assert(outputs[0].compare == 0u && outputs[1].compare == 0u);

    Electromagnet_SetBoth(ELECTROMAGNET_ATTRACT, 10000u);
    assert(outputs[0].forward == 1u && outputs[0].reverse == 0u);
    assert(outputs[1].forward == 1u && outputs[1].reverse == 0u);
    assert(outputs[0].compare == 10000u && outputs[1].compare == 10000u);

    Electromagnet_SetBoth(ELECTROMAGNET_REPEL, 3000u);
    assert(outputs[0].forward == 0u && outputs[0].reverse == 1u);
    assert(outputs[1].forward == 0u && outputs[1].reverse == 1u);
    assert(outputs[0].compare == 3000u && outputs[1].compare == 3000u);

    Electromagnet_SetBoth(ELECTROMAGNET_OFF, 10000u);
    assert(outputs[0].forward == 0u && outputs[0].reverse == 0u);
    assert(outputs[1].forward == 0u && outputs[1].reverse == 0u);
    assert(outputs[0].compare == 0u && outputs[1].compare == 0u);

    puts("electromagnet: PASS");
    return 0;
}
