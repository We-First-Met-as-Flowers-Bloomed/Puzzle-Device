#include "servo.h"

#include "tim.h"

void Servo_HardwareInit(void)
{
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);
    Servo_SetAngle(0u);
}

void Servo_SetAngle(uint8_t angle_degrees)
{
    Servo_SetPulseUs(Servo_AngleToPulseUs(angle_degrees));
}

void Servo_SetPulseUs(uint16_t pulse_us)
{
    uint32_t compare = Servo_ClampPulseUs(pulse_us);
    uint32_t period_counts = __HAL_TIM_GET_AUTORELOAD(&htim1) + 1u;
    if (compare > period_counts) compare = period_counts;
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, compare);
}

void Servo_Off(void)
{
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, 0u);
}
