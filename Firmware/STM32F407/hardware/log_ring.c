#include "log_ring.h"

#include <string.h>

void LogRing_Init(LogRing_t *ring, uint8_t *storage, uint16_t capacity)
{
    if (ring == NULL) return;
    ring->storage = storage;
    ring->capacity = capacity;
    ring->head = 0u;
    ring->tail = 0u;
    ring->used = 0u;
    ring->dropped_records = 0u;
}

bool LogRing_EnqueueRecord(LogRing_t *ring, const uint8_t *data, uint16_t size)
{
    uint16_t first;
    if ((ring == NULL) || (ring->storage == NULL) || (data == NULL) ||
        (ring->capacity == 0u) || (size == 0u) ||
        (size > (uint16_t)(ring->capacity - ring->used)))
    {
        if (ring != NULL) ++ring->dropped_records;
        return false;
    }
    first = (uint16_t)(ring->capacity - ring->head);
    if (first > size) first = size;
    memcpy(&ring->storage[ring->head], data, first);
    if (size > first) memcpy(ring->storage, &data[first], size - first);
    ring->head = (uint16_t)((ring->head + size) % ring->capacity);
    ring->used = (uint16_t)(ring->used + size);
    return true;
}

uint16_t LogRing_PeekContiguous(const LogRing_t *ring, const uint8_t **data)
{
    uint16_t size;
    if (data != NULL) *data = NULL;
    if ((ring == NULL) || (ring->storage == NULL) || (data == NULL) ||
        (ring->used == 0u)) return 0u;
    *data = &ring->storage[ring->tail];
    size = (uint16_t)(ring->capacity - ring->tail);
    return (ring->used < size) ? ring->used : size;
}

void LogRing_Consume(LogRing_t *ring, uint16_t size)
{
    if ((ring == NULL) || (ring->capacity == 0u)) return;
    if (size > ring->used) size = ring->used;
    ring->tail = (uint16_t)((ring->tail + size) % ring->capacity);
    ring->used = (uint16_t)(ring->used - size);
}

uint32_t LogRing_TakeDroppedRecords(LogRing_t *ring)
{
    uint32_t count;
    if (ring == NULL) return 0u;
    count = ring->dropped_records;
    ring->dropped_records = 0u;
    return count;
}
