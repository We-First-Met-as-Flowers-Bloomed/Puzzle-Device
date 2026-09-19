#ifndef ELECTROMAGNET_H
#define ELECTROMAGNET_H

#include <stdint.h>

typedef enum
{
    ELECTROMAGNET_OFF = 0,
    ELECTROMAGNET_ATTRACT,
    ELECTROMAGNET_REPEL
} ElectromagnetMode_t;

typedef void (*ElectromagnetOutputFn)(uint8_t channel, uint8_t forward,
                                      uint8_t reverse, uint16_t compare);

void Electromagnet_Init(ElectromagnetOutputFn output);
void Electromagnet_Set(uint8_t channel, ElectromagnetMode_t mode,
                       uint16_t duty_permyriad);
void Electromagnet_SetBoth(ElectromagnetMode_t mode, uint16_t duty_permyriad);
uint32_t Electromagnet_DutyToCompare(uint32_t period_counts,
                                    uint16_t duty_permyriad);
void Electromagnet_HardwareInit(void);

#endif
