#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "debug_uart.h"

int main(void)
{
    char output[128];
    const uint8_t frame[] = {0x01u, 0x36u, 0x00u, 0x2Au, 0x6Bu};
    size_t size = DebugUart_FormatTextRx(output, sizeof(output), "USART2",
                                         "CAM", "PIECE,4");
    assert(size == strlen("[USART2 RX][CAM] PIECE,4\r\n"));
    assert(strcmp(output, "[USART2 RX][CAM] PIECE,4\r\n") == 0);
    size = DebugUart_FormatFrameRx(output, sizeof(output), "UART5", "R",
                                  frame, sizeof(frame));
    assert(size == strlen("[UART5 RX][R] 01 36 00 2A 6B\r\n"));
    assert(strcmp(output, "[UART5 RX][R] 01 36 00 2A 6B\r\n") == 0);
    assert(DebugUart_FormatTextRx(output, 8u, "USART2", "CAM", "LONG") == 0u);
    puts("debug_uart_format: PASS");
    return 0;
}
