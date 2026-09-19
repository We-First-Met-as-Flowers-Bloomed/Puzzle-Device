#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "emm42.h"

static void assert_packet(const uint8_t *actual, size_t actual_size,
                          const uint8_t *expected, size_t expected_size)
{
    assert(actual_size == expected_size);
    assert(memcmp(actual, expected, expected_size) == 0);
}

typedef struct
{
    uint8_t calls;
    uint8_t succeed_on;
} RetryFixture_t;

static bool retry_fixture_read(void *context)
{
    RetryFixture_t *fixture = (RetryFixture_t *)context;
    ++fixture->calls;
    return (fixture->succeed_on != 0u) &&
           (fixture->calls >= fixture->succeed_on);
}

int main(void)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size;
    int32_t position;
    uint8_t homing_state;
    uint8_t motor_status;
    Emm42FrameScanner_t scanner;

    {
        RetryFixture_t fixture = {0u, 3u};
        uint8_t attempts = 0u;
        assert(Emm42_RunReadRetries(retry_fixture_read, &fixture, 3u,
                                    &attempts));
        assert(fixture.calls == 3u);
        assert(attempts == 3u);

        fixture.calls = 0u;
        fixture.succeed_on = 0u;
        attempts = 0u;
        assert(!Emm42_RunReadRetries(retry_fixture_read, &fixture, 3u,
                                     &attempts));
        assert(fixture.calls == 3u);
        assert(attempts == 3u);
    }

    {
        const uint8_t expected_enable[] = {0x01, 0xF3, 0xAB, 0x01, 0x00, 0x6B};
        size = Emm42_BuildEnable(1u, true, false, packet);
        assert_packet(packet, size, expected_enable, sizeof(expected_enable));
    }
    {
        const uint8_t expected_status_read[] = {0x01, 0x3A, 0x6B};
        const uint8_t arrived[] = {0x01, 0x3A, 0x03, 0x6B};
        const uint8_t moving[] = {0x01, 0x3A, 0x01, 0x6B};
        size = Emm42_BuildReadStatus(1u, packet);
        assert_packet(packet, size, expected_status_read, sizeof(expected_status_read));
        assert(Emm42_ParseStatus(1u, arrived, sizeof(arrived), &motor_status));
        assert((motor_status & EMM42_STATUS_IN_POSITION) != 0u);
        assert(Emm42_ParseStatus(1u, moving, sizeof(moving), &motor_status));
        assert((motor_status & EMM42_STATUS_IN_POSITION) == 0u);
    }
    {
        const uint8_t expected_homing_read[] = {0x01, 0x3B, 0x6B};
        const uint8_t homing_ready[] = {0x01, 0x3B, 0x03, 0x6B};
        uint8_t invalid[] = {0x01, 0x3B, 0x0Bu, 0x00};
        size = Emm42_BuildReadHomingState(1u, packet);
        assert_packet(packet, size, expected_homing_read, sizeof(expected_homing_read));
        assert(Emm42_ParseHomingState(1u, homing_ready, sizeof(homing_ready),
                                      &homing_state));
        assert(homing_state == 0x03u);
        assert(!Emm42_ParseHomingState(1u, invalid, sizeof(invalid), &homing_state));
    }

    const uint8_t expected_position[] = {
        0x01, 0xFD, 0x00, 0x01, 0x2C, 0x0A,
        0x00, 0x00, 0x0C, 0x80, 0x00, 0x00, 0x6B
    };
    size = Emm42_BuildPosition(1u, false, 300u, 10u, 3200u,
                               false, false, packet);
    assert_packet(packet, size, expected_position, sizeof(expected_position));

    {
        const uint8_t expected_speed[] = {0x01, 0xF6, 0x01, 0x00, 0x3C, 0x05, 0x00, 0x6B};
        size = Emm42_BuildSpeed(1u, true, 60u, 5u, false, packet);
        assert_packet(packet, size, expected_speed, sizeof(expected_speed));
    }
    {
        const uint8_t expected_stop[] = {0x01, 0xFE, 0x98, 0x00, 0x6B};
        size = Emm42_BuildStop(1u, false, packet);
        assert_packet(packet, size, expected_stop, sizeof(expected_stop));
    }
    {
        const uint8_t expected_read[] = {0x01, 0x36, 0x6B};
        size = Emm42_BuildReadPosition(1u, packet);
        assert_packet(packet, size, expected_read, sizeof(expected_read));
    }
    {
        const uint8_t stream[] = {
            0x01, 0xFD, 0x9F, 0x6B,
            0x01, 0x36, 0x00, 0x00, 0x03, 0xC0, 0x20, 0x6B
        };
        uint8_t frame[8];
        size_t i;
        bool found = false;
        Emm42_FrameScannerInit(&scanner, 1u, 0x36u, 8u);
        for (i = 0u; i < sizeof(stream); ++i)
            if (Emm42_FrameScannerFeed(&scanner, stream[i], frame)) found = true;
        assert(found);
        assert(memcmp(frame, &stream[4], sizeof(frame)) == 0);
    }
    {
        const uint8_t stream[] = {
            0x55, 0x01, 0xFD, 0x02, 0x6B,
            0x01, 0x3A, 0x03, 0x6B
        };
        uint8_t frame[4];
        size_t i;
        bool found = false;
        Emm42_FrameScannerInit(&scanner, 1u, 0x3Au, 4u);
        for (i = 0u; i < sizeof(stream); ++i)
            if (Emm42_FrameScannerFeed(&scanner, stream[i], frame)) found = true;
        assert(found);
        assert(memcmp(frame, &stream[5], sizeof(frame)) == 0);
    }
    {
        const uint8_t expected_zero[] = {0x01, 0x0A, 0x6D, 0x6B};
        size = Emm42_BuildZeroPosition(1u, packet);
        assert_packet(packet, size, expected_zero, sizeof(expected_zero));
    }
    {
        const uint8_t positive[] = {0x01, 0x36, 0x00, 0x00, 0x00, 0x0C, 0x80, 0x6B};
        const uint8_t negative[] = {0x01, 0x36, 0x01, 0x00, 0x00, 0x0C, 0x80, 0x6B};
        uint8_t invalid[] = {0x02, 0x36, 0x00, 0x00, 0x00, 0x0C, 0x80, 0x6B};

        assert(Emm42_ParsePosition(1u, positive, sizeof(positive), &position));
        assert(position == 3200);
        assert(Emm42_ParsePosition(1u, negative, sizeof(negative), &position));
        assert(position == -3200);
        assert(!Emm42_ParsePosition(1u, invalid, sizeof(invalid), &position));
        invalid[0] = 0x01;
        invalid[7] = 0x00;
        assert(!Emm42_ParsePosition(1u, invalid, sizeof(invalid), &position));
    }

    puts("emm42: PASS");
    return 0;
}
