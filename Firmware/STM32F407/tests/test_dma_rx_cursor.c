#include <assert.h>
#include <stdio.h>

#include "dma_rx_cursor.h"

int main(void)
{
    DmaRxCursor_t cursor;
    DmaRxSpan_t spans[2];
    DmaRxCursor_Init(&cursor, 8u);
    assert(DmaRxCursor_Update(&cursor, 0u, spans) == 0u);
    assert(DmaRxCursor_Update(&cursor, 3u, spans) == 1u);
    assert(spans[0].offset == 0u && spans[0].size == 3u);
    assert(DmaRxCursor_Update(&cursor, 7u, spans) == 1u);
    assert(spans[0].offset == 3u && spans[0].size == 4u);
    assert(DmaRxCursor_Update(&cursor, 2u, spans) == 2u);
    assert(spans[0].offset == 7u && spans[0].size == 1u);
    assert(spans[1].offset == 0u && spans[1].size == 2u);
    DmaRxCursor_Reset(&cursor);
    assert(DmaRxCursor_Update(&cursor, 8u, spans) == 1u);
    assert(spans[0].offset == 0u && spans[0].size == 8u);
    DmaRxCursor_Reset(&cursor);
    assert(DmaRxCursor_Update(&cursor, 4u, spans) == 1u);
    DmaRxCursor_Reset(&cursor);
    assert(DmaRxCursor_Update(&cursor, 0u, spans) == 0u);
    puts("dma_rx_cursor: PASS");
    return 0;
}
