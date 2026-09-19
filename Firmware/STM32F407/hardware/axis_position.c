#include "axis_position.h"

#define COMMAND_PULSES_PER_REVOLUTION 3200u
#define ENCODER_UNITS_PER_REVOLUTION 65536u

uint32_t AxisPosition_CommandPulsesToEncoder(uint32_t commanded_pulses)
{
    return (uint32_t)((((uint64_t)commanded_pulses *
                        ENCODER_UNITS_PER_REVOLUTION) +
                       (COMMAND_PULSES_PER_REVOLUTION / 2u)) /
                      COMMAND_PULSES_PER_REVOLUTION);
}

uint8_t AxisPosition_HasReached(int32_t start_position,
                                int32_t current_position,
                                uint32_t commanded_pulses,
                                uint32_t tolerance_pulses)
{
    int64_t travel = (int64_t)current_position - (int64_t)start_position;
    uint64_t required;
    uint64_t expected = AxisPosition_CommandPulsesToEncoder(commanded_pulses);
    uint64_t tolerance = (((uint64_t)tolerance_pulses *
                           ENCODER_UNITS_PER_REVOLUTION) +
                          COMMAND_PULSES_PER_REVOLUTION - 1u) /
                         COMMAND_PULSES_PER_REVOLUTION;

    if (travel < 0) travel = -travel;
    required = (expected > tolerance) ? (expected - tolerance) : 0u;
    return ((uint64_t)travel >= required) ? 1u : 0u;
}

uint8_t AxisPosition_HasMovedInDirection(int32_t start_position,
                                         int32_t current_position,
                                         uint8_t logical_direction,
                                         uint8_t logical_positive_is_negative)
{
    int64_t travel = (int64_t)current_position - (int64_t)start_position;
    uint8_t expect_negative = (uint8_t)((logical_direction ? 0u : 1u) ^
                                        logical_positive_is_negative);
    if (travel == 0) return 0u;
    return expect_negative ? ((travel < 0) ? 1u : 0u) :
                             ((travel > 0) ? 1u : 0u);
}

uint8_t AxisPosition_SelectCorrection(int32_t encoder_error,
                                      uint32_t deadband_pulses,
                                      uint32_t maximum_pulses,
                                      uint8_t logical_positive_is_negative,
                                      uint8_t *direction,
                                      uint32_t *pulses)
{
    uint64_t magnitude;
    uint64_t deadband;
    uint64_t converted;

    if ((direction == 0) || (pulses == 0) || (maximum_pulses == 0u)) return 0u;
    magnitude = (encoder_error < 0) ? (uint64_t)(-(int64_t)encoder_error) :
                                      (uint64_t)encoder_error;
    deadband = AxisPosition_CommandPulsesToEncoder(deadband_pulses);
    if (magnitude <= deadband)
    {
        *pulses = 0u;
        return 0u;
    }
    converted = ((magnitude * COMMAND_PULSES_PER_REVOLUTION) +
                 (ENCODER_UNITS_PER_REVOLUTION / 2u)) /
                ENCODER_UNITS_PER_REVOLUTION;
    if (converted == 0u) converted = 1u;
    if (converted > maximum_pulses) converted = maximum_pulses;
    *pulses = (uint32_t)converted;
    if (encoder_error > 0)
        *direction = logical_positive_is_negative ? 1u : 0u;
    else
        *direction = logical_positive_is_negative ? 0u : 1u;
    return 1u;
}
