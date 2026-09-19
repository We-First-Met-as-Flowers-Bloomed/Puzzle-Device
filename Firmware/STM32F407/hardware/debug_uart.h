#ifndef DEBUG_UART_H
#define DEBUG_UART_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef DEBUG_UART_FORMAT_ONLY
#include "stm32f4xx_hal.h"
#endif

size_t DebugUart_FormatTextRx(char *out, size_t capacity,
                              const char *port, const char *source,
                              const char *line);
size_t DebugUart_FormatFrameRx(char *out, size_t capacity,
                               const char *port, const char *axis,
                               const uint8_t *frame, size_t frame_size);

#ifndef DEBUG_UART_FORMAT_ONLY
void DebugUart_Init(void);
bool DebugUart_EnqueueRecord(const char *record, size_t size);
void DebugUart_Printf(const char *format, ...);
void DebugUart_PutChar(int ch);
void DebugUart_LogTextRx(const char *port, const char *source, const char *line);
void DebugUart_LogFrameRx(const char *port, const char *axis,
                          const uint8_t *frame, size_t frame_size);
void DebugUart_TxComplete(UART_HandleTypeDef *uart);
#endif

#endif
