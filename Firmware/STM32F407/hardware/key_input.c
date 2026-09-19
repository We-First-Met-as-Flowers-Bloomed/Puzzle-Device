#include "key_input.h"

#include <string.h>

#define KEY_COUNT 4u
#define KEY_DEBOUNCE_MS 30u
#define KEY_QUEUE_SIZE 8u

static volatile KeyEvent_t event_queue[KEY_QUEUE_SIZE];
static volatile uint8_t queue_head;
static volatile uint8_t queue_tail;
static uint32_t last_tick[KEY_COUNT];
static uint8_t has_tick[KEY_COUNT];
static uint8_t locked[KEY_COUNT];
static uint8_t release_timing[KEY_COUNT];
static uint32_t release_tick[KEY_COUNT];

static KeyEvent_t event_from_pin(uint8_t pin_id)
{
    static const KeyEvent_t events[KEY_COUNT] = {
        KEY_EVENT_NEXT,
        KEY_EVENT_TOGGLE,
        KEY_EVENT_EXIT,
        KEY_EVENT_RESET
    };

    if ((pin_id < KEY1_PIN_ID) || (pin_id > KEY4_PIN_ID))
    {
        return KEY_EVENT_NONE;
    }
    return events[pin_id - KEY1_PIN_ID];
}

void KeyInput_Init(void)
{
    queue_head = 0u;
    queue_tail = 0u;
    memset(last_tick, 0, sizeof(last_tick));
    memset(has_tick, 0, sizeof(has_tick));
    memset(locked, 0, sizeof(locked));
    memset(release_timing, 0, sizeof(release_timing));
}

void KeyInput_RecordIsr(uint8_t pin_id, uint32_t tick_ms)
{
    KeyEvent_t event = event_from_pin(pin_id);
    uint8_t index;
    uint8_t next_head;

    if (event == KEY_EVENT_NONE)
    {
        return;
    }

    index = (uint8_t)(pin_id - KEY1_PIN_ID);
    if (locked[index])
    {
        return;
    }
    if (has_tick[index] && ((uint32_t)(tick_ms - last_tick[index]) < KEY_DEBOUNCE_MS))
    {
        return;
    }
    has_tick[index] = 1u;
    last_tick[index] = tick_ms;
    locked[index] = 1u;
    release_timing[index] = 0u;

    next_head = (uint8_t)((queue_head + 1u) % KEY_QUEUE_SIZE);
    if (next_head == queue_tail)
    {
        return;
    }
    event_queue[queue_head] = event;
    queue_head = next_head;
}

void KeyInput_UpdateLevel(uint8_t pin_id, uint8_t pressed, uint32_t tick_ms)
{
    uint8_t index;

    if ((pin_id < KEY1_PIN_ID) || (pin_id > KEY4_PIN_ID)) return;
    index = (uint8_t)(pin_id - KEY1_PIN_ID);
    if (!locked[index]) return;

    if (pressed)
    {
        release_timing[index] = 0u;
        return;
    }
    if (!release_timing[index])
    {
        release_timing[index] = 1u;
        release_tick[index] = tick_ms;
        return;
    }
    if ((uint32_t)(tick_ms - release_tick[index]) >= KEY_DEBOUNCE_MS)
    {
        locked[index] = 0u;
        release_timing[index] = 0u;
        has_tick[index] = 0u;
    }
}

KeyEvent_t KeyInput_TakeEvent(void)
{
    KeyEvent_t event;

    if (queue_tail == queue_head)
    {
        return KEY_EVENT_NONE;
    }
    event = event_queue[queue_tail];
    queue_tail = (uint8_t)((queue_tail + 1u) % KEY_QUEUE_SIZE);
    return event;
}
