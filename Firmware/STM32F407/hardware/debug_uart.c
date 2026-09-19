#include "debug_uart.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

size_t DebugUart_FormatTextRx(char *out, size_t capacity,
                              const char *port, const char *source,
                              const char *line)
{
    int written;
    if ((out == NULL) || (capacity == 0u) || (port == NULL) ||
        (source == NULL) || (line == NULL)) return 0u;
    written = snprintf(out, capacity, "[%s RX][%s] %s\r\n",
                       port, source, line);
    if ((written < 0) || ((size_t)written >= capacity))
    {
        out[0] = '\0';
        return 0u;
    }
    return (size_t)written;
}

size_t DebugUart_FormatFrameRx(char *out, size_t capacity,
                               const char *port, const char *axis,
                               const uint8_t *frame, size_t frame_size)
{
    size_t used;
    size_t i;
    int written;
    if ((out == NULL) || (capacity == 0u) || (port == NULL) ||
        (axis == NULL) || (frame == NULL) || (frame_size == 0u)) return 0u;
    written = snprintf(out, capacity, "[%s RX][%s]", port, axis);
    if ((written < 0) || ((size_t)written >= capacity)) return 0u;
    used = (size_t)written;
    for (i = 0u; i < frame_size; ++i)
    {
        written = snprintf(&out[used], capacity - used, " %02X", frame[i]);
        if ((written < 0) || ((size_t)written >= (capacity - used)))
        {
            out[0] = '\0';
            return 0u;
        }
        used += (size_t)written;
    }
    if ((capacity - used) < 3u)
    {
        out[0] = '\0';
        return 0u;
    }
    out[used++] = '\r';
    out[used++] = '\n';
    out[used] = '\0';
    return used;
}

#ifndef DEBUG_UART_FORMAT_ONLY
#include "log_ring.h"
#include "usart.h"

#define DEBUG_UART_QUEUE_SIZE 4096u
#define DEBUG_UART_RECORD_SIZE 256u

static uint8_t queue_storage[DEBUG_UART_QUEUE_SIZE];
static LogRing_t queue;
static uint16_t active_dma_size;
static uint8_t dma_active;
static char stdio_line[DEBUG_UART_RECORD_SIZE];
static uint16_t stdio_line_size;

static uint32_t critical_enter(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    return primask;
}

static void critical_exit(uint32_t primask)
{
    if (primask == 0u) __enable_irq();
}

static void kick_locked(void)
{
    const uint8_t *data;
    uint16_t size;
    if (dma_active) return;
    size = LogRing_PeekContiguous(&queue, &data);
    if (size == 0u) return;
    active_dma_size = size;
    dma_active = 1u;
    if (HAL_UART_Transmit_DMA(&huart1, (uint8_t *)data, size) != HAL_OK)
    {
        dma_active = 0u;
        active_dma_size = 0u;
    }
}

void DebugUart_Init(void)
{
    uint32_t primask = critical_enter();
    LogRing_Init(&queue, queue_storage, sizeof(queue_storage));
    active_dma_size = 0u;
    dma_active = 0u;
    stdio_line_size = 0u;
    critical_exit(primask);
}

bool DebugUart_EnqueueRecord(const char *record, size_t size)
{
    bool queued;
    uint32_t primask;
    if ((record == NULL) || (size == 0u) || (size > UINT16_MAX)) return false;
    primask = critical_enter();
    queued = LogRing_EnqueueRecord(&queue, (const uint8_t *)record,
                                   (uint16_t)size);
    kick_locked();
    critical_exit(primask);
    return queued;
}

void DebugUart_Printf(const char *format, ...)
{
    char record[DEBUG_UART_RECORD_SIZE];
    va_list args;
    int written;
    va_start(args, format);
    written = vsnprintf(record, sizeof(record), format, args);
    va_end(args);
    if ((written > 0) && ((size_t)written < sizeof(record)))
        (void)DebugUart_EnqueueRecord(record, (size_t)written);
}

void DebugUart_PutChar(int ch)
{
    if (stdio_line_size >= sizeof(stdio_line)) stdio_line_size = 0u;
    stdio_line[stdio_line_size++] = (char)ch;
    if ((ch == '\n') || (stdio_line_size == sizeof(stdio_line)))
    {
        (void)DebugUart_EnqueueRecord(stdio_line, stdio_line_size);
        stdio_line_size = 0u;
    }
}

void DebugUart_LogTextRx(const char *port, const char *source, const char *line)
{
    char record[DEBUG_UART_RECORD_SIZE];
    size_t size = DebugUart_FormatTextRx(record, sizeof(record), port,
                                         source, line);
    if (size != 0u) (void)DebugUart_EnqueueRecord(record, size);
}

void DebugUart_LogFrameRx(const char *port, const char *axis,
                          const uint8_t *frame, size_t frame_size)
{
    char record[DEBUG_UART_RECORD_SIZE];
    size_t size = DebugUart_FormatFrameRx(record, sizeof(record), port, axis,
                                          frame, frame_size);
    if (size != 0u) (void)DebugUart_EnqueueRecord(record, size);
}

void DebugUart_TxComplete(UART_HandleTypeDef *uart)
{
    uint32_t primask;
    uint32_t dropped;
    char report[48];
    int written;
    if (uart != &huart1) return;
    primask = critical_enter();
    LogRing_Consume(&queue, active_dma_size);
    active_dma_size = 0u;
    dma_active = 0u;
    dropped = LogRing_TakeDroppedRecords(&queue);
    if (dropped != 0u)
    {
        written = snprintf(report, sizeof(report), "[LOG] DROPPED=%lu\r\n",
                           (unsigned long)dropped);
        if ((written > 0) && ((size_t)written < sizeof(report)))
            (void)LogRing_EnqueueRecord(&queue, (const uint8_t *)report,
                                        (uint16_t)written);
    }
    kick_locked();
    critical_exit(primask);
}
#endif
