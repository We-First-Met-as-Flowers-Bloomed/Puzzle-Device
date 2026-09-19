#include "servo.h"

#define SERVO_MIN_ANGLE 0u
#define SERVO_MAX_ANGLE 180u
#define SERVO_MIN_PULSE_US 500u
#define SERVO_MAX_PULSE_US 2500u

uint16_t Servo_AngleToPulseUs(uint16_t angle_degrees)
{
    uint32_t pulse;
    if (angle_degrees > SERVO_MAX_ANGLE) angle_degrees = SERVO_MAX_ANGLE;
    pulse = SERVO_MIN_PULSE_US +
            ((uint32_t)(angle_degrees - SERVO_MIN_ANGLE) *
             (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)) /
            (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE);
    return (uint16_t)pulse;
}

uint16_t Servo_ClampPulseUs(uint16_t pulse_us)
{
    if (pulse_us < SERVO_MIN_PULSE_US) return SERVO_MIN_PULSE_US;
    if (pulse_us > SERVO_MAX_PULSE_US) return SERVO_MAX_PULSE_US;
    return pulse_us;
}
