#ifndef DMA_RX_CURSOR_H
#define DMA_RX_CURSOR_H

#include <stdint.h>

typedef struct
{
    uint16_t offset;
    uint16_t size;
} DmaRxSpan_t;

typedef struct
{
    uint16_t capacity;
    uint16_t consumer;
} DmaRxCursor_t;

void DmaRxCursor_Init(DmaRxCursor_t *cursor, uint16_t capacity);
void DmaRxCursor_Reset(DmaRxCursor_t *cursor);
uint8_t DmaRxCursor_Update(DmaRxCursor_t *cursor, uint16_t producer,
                           DmaRxSpan_t spans[2]);

#endif
