#ifndef SERVO_H
#define SERVO_H

#include <stdint.h>

uint16_t Servo_AngleToPulseUs(uint16_t angle_degrees);
uint16_t Servo_ClampPulseUs(uint16_t pulse_us);
void Servo_HardwareInit(void);
void Servo_SetAngle(uint8_t angle_degrees);
void Servo_SetPulseUs(uint16_t pulse_us);
void Servo_Off(void);

#endif
