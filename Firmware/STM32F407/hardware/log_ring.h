#ifndef LOG_RING_H
#define LOG_RING_H

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint8_t *storage;
    uint16_t capacity;
    uint16_t head;
    uint16_t tail;
    uint16_t used;
    uint32_t dropped_records;
} LogRing_t;

void LogRing_Init(LogRing_t *ring, uint8_t *storage, uint16_t capacity);
bool LogRing_EnqueueRecord(LogRing_t *ring, const uint8_t *data, uint16_t size);
uint16_t LogRing_PeekContiguous(const LogRing_t *ring, const uint8_t **data);
void LogRing_Consume(LogRing_t *ring, uint16_t size);
uint32_t LogRing_TakeDroppedRecords(LogRing_t *ring);

#endif
