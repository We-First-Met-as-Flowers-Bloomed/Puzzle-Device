#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "log_ring.h"

int main(void)
{
    uint8_t storage[8];
    LogRing_t ring;
    const uint8_t *data;
    uint16_t size;

    LogRing_Init(&ring, storage, sizeof(storage));
    assert(LogRing_EnqueueRecord(&ring, (const uint8_t *)"ABCDE", 5u));
    size = LogRing_PeekContiguous(&ring, &data);
    assert(size == 5u && memcmp(data, "ABCDE", 5u) == 0);
    LogRing_Consume(&ring, 3u);
    assert(LogRing_EnqueueRecord(&ring, (const uint8_t *)"12345", 5u));
    size = LogRing_PeekContiguous(&ring, &data);
    assert(size == 5u && memcmp(data, "DE123", 5u) == 0);
    LogRing_Consume(&ring, 5u);
    size = LogRing_PeekContiguous(&ring, &data);
    assert(size == 2u && memcmp(data, "45", 2u) == 0);
    assert(!LogRing_EnqueueRecord(&ring, (const uint8_t *)"TOOLONG", 7u));
    assert(LogRing_TakeDroppedRecords(&ring) == 1u);
    assert(LogRing_TakeDroppedRecords(&ring) == 0u);
    LogRing_Consume(&ring, 99u);
    assert(LogRing_EnqueueRecord(&ring, (const uint8_t *)"12345678", 8u));
    assert(!LogRing_EnqueueRecord(&ring, (const uint8_t *)"X", 1u));
    assert(LogRing_TakeDroppedRecords(&ring) == 1u);
    puts("log_ring: PASS");
    return 0;
}
