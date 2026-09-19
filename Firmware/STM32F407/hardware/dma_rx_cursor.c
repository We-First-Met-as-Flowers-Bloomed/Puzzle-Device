#include "dma_rx_cursor.h"

void DmaRxCursor_Init(DmaRxCursor_t *cursor, uint16_t capacity)
{
    if (cursor == 0) return;
    cursor->capacity = capacity;
    cursor->consumer = 0u;
}

void DmaRxCursor_Reset(DmaRxCursor_t *cursor)
{
    if (cursor != 0) cursor->consumer = 0u;
}

uint8_t DmaRxCursor_Update(DmaRxCursor_t *cursor, uint16_t producer,
                           DmaRxSpan_t spans[2])
{
    uint16_t normalized;
    uint8_t count = 0u;
    if ((cursor == 0) || (spans == 0) || (cursor->capacity == 0u) ||
        (producer > cursor->capacity)) return 0u;
    normalized = (producer == cursor->capacity) ? 0u : producer;
    if ((producer == cursor->capacity) && (cursor->consumer == 0u))
    {
        spans[0].offset = 0u;
        spans[0].size = cursor->capacity;
        count = 1u;
    }
    else if (normalized > cursor->consumer)
    {
        spans[0].offset = cursor->consumer;
        spans[0].size = (uint16_t)(normalized - cursor->consumer);
        count = 1u;
    }
    else if (normalized < cursor->consumer)
    {
        spans[0].offset = cursor->consumer;
        spans[0].size = (uint16_t)(cursor->capacity - cursor->consumer);
        if (normalized != 0u)
        {
            spans[1].offset = 0u;
            spans[1].size = normalized;
            count = 2u;
        }
        else count = 1u;
    }
    cursor->consumer = normalized;
    return count;
}
