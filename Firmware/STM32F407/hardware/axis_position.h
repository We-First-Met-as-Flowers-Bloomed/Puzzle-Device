#ifndef AXIS_POSITION_H
#define AXIS_POSITION_H

#include <stdint.h>

uint32_t AxisPosition_CommandPulsesToEncoder(uint32_t commanded_pulses);
uint8_t AxisPosition_HasReached(int32_t start_position,
                                int32_t current_position,
                                uint32_t commanded_pulses,
                                uint32_t tolerance_pulses);
uint8_t AxisPosition_HasMovedInDirection(int32_t start_position,
                                         int32_t current_position,
                                         uint8_t logical_direction,
                                         uint8_t logical_positive_is_negative);
uint8_t AxisPosition_SelectCorrection(int32_t encoder_error,
                                      uint32_t deadband_pulses,
                                      uint32_t maximum_pulses,
                                      uint8_t logical_positive_is_negative,
                                      uint8_t *direction,
                                      uint32_t *pulses);

#endif
