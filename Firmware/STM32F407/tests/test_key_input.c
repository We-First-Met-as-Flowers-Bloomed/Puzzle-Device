#include <assert.h>
#include <stdio.h>

#include "key_input.h"

int main(void)
{
    KeyInput_Init();

    KeyInput_RecordIsr(KEY1_PIN_ID, 100u);
    assert(KeyInput_TakeEvent() == KEY_EVENT_NEXT);
    assert(KeyInput_TakeEvent() == KEY_EVENT_NONE);

    KeyInput_RecordIsr(KEY2_PIN_ID, 200u);
    KeyInput_RecordIsr(KEY2_PIN_ID, 220u);
    assert(KeyInput_TakeEvent() == KEY_EVENT_TOGGLE);
    assert(KeyInput_TakeEvent() == KEY_EVENT_NONE);

    KeyInput_RecordIsr(KEY2_PIN_ID, 231u);
    assert(KeyInput_TakeEvent() == KEY_EVENT_NONE);
    KeyInput_UpdateLevel(KEY2_PIN_ID, 0u, 240u);
    KeyInput_UpdateLevel(KEY2_PIN_ID, 0u, 269u);
    KeyInput_RecordIsr(KEY2_PIN_ID, 270u);
    assert(KeyInput_TakeEvent() == KEY_EVENT_NONE);
    KeyInput_UpdateLevel(KEY2_PIN_ID, 0u, 270u);
    KeyInput_RecordIsr(KEY2_PIN_ID, 271u);
    assert(KeyInput_TakeEvent() == KEY_EVENT_TOGGLE);

    KeyInput_RecordIsr(KEY3_PIN_ID, 300u);
    KeyInput_RecordIsr(KEY4_PIN_ID, 301u);
    assert(KeyInput_TakeEvent() == KEY_EVENT_EXIT);
    assert(KeyInput_TakeEvent() == KEY_EVENT_RESET);
    assert(KeyInput_TakeEvent() == KEY_EVENT_NONE);

    puts("key_input: PASS");
    return 0;
}
