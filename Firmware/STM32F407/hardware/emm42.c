#include "emm42.h"

#include <string.h>

#define EMM42_CHECKSUM 0x6Bu

static uint8_t response_size_for_function(uint8_t function)
{
    return (function == 0x36u) ? 8u : 4u;
}

void Emm42_FrameScannerInit(Emm42FrameScanner_t *scanner, uint8_t address,
                            uint8_t expected_function, uint8_t expected_size)
{
    if (scanner == NULL) return;
    memset(scanner, 0, sizeof(*scanner));
    scanner->address = address;
    scanner->expected_function = expected_function;
    scanner->expected_size = expected_size;
}

bool Emm42_FrameScannerFeed(Emm42FrameScanner_t *scanner, uint8_t byte,
                            uint8_t *frame)
{
    uint8_t unrelated_size;
    if ((scanner == NULL) || (frame == NULL) ||
        (scanner->expected_size < 3u) ||
        (scanner->expected_size > sizeof(scanner->frame))) return false;
    if (scanner->skip_remaining != 0u)
    {
        --scanner->skip_remaining;
        return false;
    }
    if (scanner->index == 0u)
    {
        if (byte == scanner->address)
        {
            scanner->frame[0] = byte;
            scanner->index = 1u;
        }
        return false;
    }
    if (scanner->index == 1u)
    {
        if (byte != scanner->expected_function)
        {
            unrelated_size = response_size_for_function(byte);
            scanner->index = 0u;
            scanner->skip_remaining = (uint8_t)(unrelated_size - 2u);
            return false;
        }
        scanner->frame[1] = byte;
        scanner->index = 2u;
        return false;
    }
    scanner->frame[scanner->index++] = byte;
    if (scanner->index < scanner->expected_size) return false;
    scanner->index = 0u;
    if (scanner->frame[scanner->expected_size - 1u] != EMM42_CHECKSUM)
        return false;
    memcpy(frame, scanner->frame, scanner->expected_size);
    return true;
}

bool Emm42_RunReadRetries(Emm42ReadAttemptFn attempt, void *context,
                          uint8_t maximum_attempts, uint8_t *attempts_used)
{
    uint8_t attempt_number;
    if (attempts_used != NULL) *attempts_used = 0u;
    if ((attempt == NULL) || (maximum_attempts == 0u)) return false;
    for (attempt_number = 1u; attempt_number <= maximum_attempts;
         ++attempt_number)
    {
        if (attempts_used != NULL) *attempts_used = attempt_number;
        if (attempt(context)) return true;
    }
    return false;
}

size_t Emm42_BuildEnable(uint8_t address, bool enable, bool synchronized,
                         uint8_t *packet)
{
    if (packet == NULL) return 0u;
    packet[0] = address;
    packet[1] = 0xF3u;
    packet[2] = 0xABu;
    packet[3] = enable ? 0x01u : 0x00u;
    packet[4] = synchronized ? 0x01u : 0x00u;
    packet[5] = EMM42_CHECKSUM;
    return 6u;
}

size_t Emm42_BuildPosition(uint8_t address, bool counter_clockwise,
                           uint16_t speed_rpm, uint8_t acceleration,
                           uint32_t pulses, bool absolute,
                           bool synchronized, uint8_t *packet)
{
    if (packet == NULL)
    {
        return 0u;
    }

    packet[0] = address;
    packet[1] = 0xFDu;
    packet[2] = counter_clockwise ? 0x01u : 0x00u;
    packet[3] = (uint8_t)(speed_rpm >> 8);
    packet[4] = (uint8_t)speed_rpm;
    packet[5] = acceleration;
    packet[6] = (uint8_t)(pulses >> 24);
    packet[7] = (uint8_t)(pulses >> 16);
    packet[8] = (uint8_t)(pulses >> 8);
    packet[9] = (uint8_t)pulses;
    packet[10] = absolute ? 0x01u : 0x00u;
    packet[11] = synchronized ? 0x01u : 0x00u;
    packet[12] = EMM42_CHECKSUM;
    return 13u;
}

size_t Emm42_BuildStop(uint8_t address, bool synchronized, uint8_t *packet)
{
    if (packet == NULL)
    {
        return 0u;
    }
    packet[0] = address;
    packet[1] = 0xFEu;
    packet[2] = 0x98u;
    packet[3] = synchronized ? 0x01u : 0x00u;
    packet[4] = EMM42_CHECKSUM;
    return 5u;
}

size_t Emm42_BuildSpeed(uint8_t address, bool counter_clockwise,
                        uint16_t speed_rpm, uint8_t acceleration,
                        bool synchronized, uint8_t *packet)
{
    if (packet == NULL)
    {
        return 0u;
    }
    packet[0] = address;
    packet[1] = 0xF6u;
    packet[2] = counter_clockwise ? 0x01u : 0x00u;
    packet[3] = (uint8_t)(speed_rpm >> 8);
    packet[4] = (uint8_t)speed_rpm;
    packet[5] = acceleration;
    packet[6] = synchronized ? 0x01u : 0x00u;
    packet[7] = EMM42_CHECKSUM;
    return 8u;
}

size_t Emm42_BuildReadPosition(uint8_t address, uint8_t *packet)
{
    if (packet == NULL)
    {
        return 0u;
    }
    packet[0] = address;
    packet[1] = 0x36u;
    packet[2] = EMM42_CHECKSUM;
    return 3u;
}

size_t Emm42_BuildReadStatus(uint8_t address, uint8_t *packet)
{
    if (packet == NULL) return 0u;
    packet[0] = address;
    packet[1] = 0x3Au;
    packet[2] = EMM42_CHECKSUM;
    return 3u;
}

size_t Emm42_BuildZeroPosition(uint8_t address, uint8_t *packet)
{
    if (packet == NULL)
    {
        return 0u;
    }
    packet[0] = address;
    packet[1] = 0x0Au;
    packet[2] = 0x6Du;
    packet[3] = EMM42_CHECKSUM;
    return 4u;
}

size_t Emm42_BuildReturnToZero(uint8_t address, uint8_t mode, bool synchronized,
                               uint8_t *packet)
{
    /* Trigger physical return-to-zero (0x9A).
       mode: 0=单圈就近回零 1=单圈方向回零 2=多圈无限位碰撞 3=多圈有限位开关 */
    if (packet == NULL)
    {
        return 0u;
    }
    packet[0] = address;
    packet[1] = 0x9Au;
    packet[2] = mode;
    packet[3] = synchronized ? 0x01u : 0x00u;
    packet[4] = EMM42_CHECKSUM;
    return 5u;
}

size_t Emm42_BuildReadHomingState(uint8_t address, uint8_t *packet)
{
    if (packet == NULL) return 0u;
    packet[0] = address;
    packet[1] = 0x3Bu;
    packet[2] = EMM42_CHECKSUM;
    return 3u;
}

size_t Emm42_BuildSetHomingZero(uint8_t address, uint8_t store, uint8_t *packet)
{
    /* 0x93 0x88: set the single-turn homing zero to the current position.
       store: 0 = RAM only, 1 = save to flash (survives power cycle). */
    if (packet == NULL) return 0u;
    packet[0] = address;
    packet[1] = 0x93u;
    packet[2] = 0x88u;
    packet[3] = (store != 0u) ? 0x01u : 0x00u;
    packet[4] = EMM42_CHECKSUM;
    return 5u;
}

bool Emm42_ParseHomingState(uint8_t address, const uint8_t *reply,
                            size_t reply_size, uint8_t *state)
{
    if ((reply == NULL) || (state == NULL) || (reply_size != 4u)) return false;
    if ((reply[0] != address) || (reply[1] != 0x3Bu) ||
        (reply[3] != EMM42_CHECKSUM)) return false;
    *state = reply[2];
    return true;
}

bool Emm42_ParsePosition(uint8_t address, const uint8_t *reply,
                         size_t reply_size, int32_t *position)
{
    uint32_t magnitude;

    if ((reply == NULL) || (position == NULL) || (reply_size != 8u))
    {
        return false;
    }
    if ((reply[0] != address) || (reply[1] != 0x36u) ||
        (reply[7] != EMM42_CHECKSUM) || (reply[2] > 1u))
    {
        return false;
    }

    magnitude = ((uint32_t)reply[3] << 24) |
                ((uint32_t)reply[4] << 16) |
                ((uint32_t)reply[5] << 8) |
                (uint32_t)reply[6];
    if (magnitude > 0x7FFFFFFFu)
    {
        return false;
    }
    *position = (reply[2] != 0u) ? -(int32_t)magnitude : (int32_t)magnitude;
    return true;
}

bool Emm42_ParseStatus(uint8_t address, const uint8_t *reply,
                       size_t reply_size, uint8_t *status)
{
    if ((reply == NULL) || (status == NULL) || (reply_size != 4u)) return false;
    if ((reply[0] != address) || (reply[1] != 0x3Au) ||
        (reply[3] != EMM42_CHECKSUM)) return false;
    *status = reply[2];
    return true;
}
