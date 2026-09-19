#ifndef EMM42_H
#define EMM42_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EMM42_MAX_PACKET_SIZE 16u
#define EMM42_STATUS_ENABLED 0x01u
#define EMM42_STATUS_IN_POSITION 0x02u
#define EMM42_STATUS_STALLED 0x04u
#define EMM42_STATUS_STALL_PROTECTION 0x08u

typedef struct
{
    uint8_t address;
    uint8_t expected_function;
    uint8_t expected_size;
    uint8_t index;
    uint8_t skip_remaining;
    uint8_t frame[8];
} Emm42FrameScanner_t;

typedef bool (*Emm42ReadAttemptFn)(void *context);

void Emm42_FrameScannerInit(Emm42FrameScanner_t *scanner, uint8_t address,
                            uint8_t expected_function, uint8_t expected_size);
bool Emm42_FrameScannerFeed(Emm42FrameScanner_t *scanner, uint8_t byte,
                            uint8_t *frame);
bool Emm42_RunReadRetries(Emm42ReadAttemptFn attempt, void *context,
                          uint8_t maximum_attempts, uint8_t *attempts_used);

size_t Emm42_BuildEnable(uint8_t address, bool enable, bool synchronized,
                         uint8_t *packet);
size_t Emm42_BuildPosition(uint8_t address, bool counter_clockwise,
                           uint16_t speed_rpm, uint8_t acceleration,
                           uint32_t pulses, bool absolute,
                           bool synchronized, uint8_t *packet);
size_t Emm42_BuildStop(uint8_t address, bool synchronized, uint8_t *packet);
size_t Emm42_BuildSpeed(uint8_t address, bool counter_clockwise,
                        uint16_t speed_rpm, uint8_t acceleration,
                        bool synchronized, uint8_t *packet);
size_t Emm42_BuildReadPosition(uint8_t address, uint8_t *packet);
size_t Emm42_BuildReadStatus(uint8_t address, uint8_t *packet);
size_t Emm42_BuildZeroPosition(uint8_t address, uint8_t *packet);
size_t Emm42_BuildReturnToZero(uint8_t address, uint8_t mode, bool synchronized,
                               uint8_t *packet);
size_t Emm42_BuildReadHomingState(uint8_t address, uint8_t *packet);
size_t Emm42_BuildSetHomingZero(uint8_t address, uint8_t store, uint8_t *packet);
bool Emm42_ParseHomingState(uint8_t address, const uint8_t *reply,
                            size_t reply_size, uint8_t *state);
bool Emm42_ParsePosition(uint8_t address, const uint8_t *reply,
                         size_t reply_size, int32_t *position);
bool Emm42_ParseStatus(uint8_t address, const uint8_t *reply,
                       size_t reply_size, uint8_t *status);

#endif
