#ifndef KEY_INPUT_H
#define KEY_INPUT_H

#include <stdint.h>

typedef enum
{
    KEY_EVENT_NONE = 0,
    KEY_EVENT_NEXT,
    KEY_EVENT_TOGGLE,
    KEY_EVENT_EXIT,
    KEY_EVENT_RESET
} KeyEvent_t;

enum
{
    KEY1_PIN_ID = 1,
    KEY2_PIN_ID,
    KEY3_PIN_ID,
    KEY4_PIN_ID
};

void KeyInput_Init(void);
void KeyInput_RecordIsr(uint8_t pin_id, uint32_t tick_ms);
void KeyInput_UpdateLevel(uint8_t pin_id, uint8_t pressed, uint32_t tick_ms);
KeyEvent_t KeyInput_TakeEvent(void);

#endif
