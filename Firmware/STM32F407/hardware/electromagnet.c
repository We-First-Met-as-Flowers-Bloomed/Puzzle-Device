#include "electromagnet.h"

#define ELECTROMAGNET_CHANNEL_COUNT 2u
#define ELECTROMAGNET_DUTY_MAX 10000u

static ElectromagnetOutputFn write_output;

uint32_t Electromagnet_DutyToCompare(uint32_t period_counts,
                                    uint16_t duty_permyriad)
{
    uint16_t duty = duty_permyriad;
    if (duty > ELECTROMAGNET_DUTY_MAX) duty = ELECTROMAGNET_DUTY_MAX;
    return (period_counts * duty) / ELECTROMAGNET_DUTY_MAX;
}

void Electromagnet_Init(ElectromagnetOutputFn output)
{
    write_output = output;
    Electromagnet_SetBoth(ELECTROMAGNET_OFF, 0u);
}

void Electromagnet_Set(uint8_t channel, ElectromagnetMode_t mode,
                       uint16_t duty_permyriad)
{
    uint8_t forward = 0u;
    uint8_t reverse = 0u;
    uint16_t compare = duty_permyriad;

    if ((channel >= ELECTROMAGNET_CHANNEL_COUNT) || (write_output == 0)) return;
    if (compare > ELECTROMAGNET_DUTY_MAX) compare = ELECTROMAGNET_DUTY_MAX;

    if (mode == ELECTROMAGNET_ATTRACT)
    {
        forward = 1u;
    }
    else if (mode == ELECTROMAGNET_REPEL)
    {
        reverse = 1u;
    }
    else
    {
        compare = 0u;
    }
    write_output(channel, forward, reverse, compare);
}

void Electromagnet_SetBoth(ElectromagnetMode_t mode, uint16_t duty_permyriad)
{
    Electromagnet_Set(0u, mode, duty_permyriad);
    Electromagnet_Set(1u, mode, duty_permyriad);
}
