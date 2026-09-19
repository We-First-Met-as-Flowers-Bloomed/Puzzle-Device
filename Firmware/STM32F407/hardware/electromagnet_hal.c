#include "electromagnet.h"

#include "main.h"
#include "tim.h"

static void electromagnet_write(uint8_t channel, uint8_t forward,
                                uint8_t reverse, uint16_t duty_permyriad)
{
    uint32_t period_counts = __HAL_TIM_GET_AUTORELOAD(&htim1) + 1u;
    uint32_t compare =
        Electromagnet_DutyToCompare(period_counts, duty_permyriad);
    if (channel == 0u)
    {
        __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, 0u);
        HAL_GPIO_WritePin(M1_low_GPIO_Port, M1_low_Pin, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(M1_high_GPIO_Port, M1_high_Pin, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(M1_low_GPIO_Port, M1_low_Pin,
                          forward ? GPIO_PIN_SET : GPIO_PIN_RESET);
        HAL_GPIO_WritePin(M1_high_GPIO_Port, M1_high_Pin,
                          reverse ? GPIO_PIN_SET : GPIO_PIN_RESET);
        __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, compare);
    }
    else
    {
        __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, 0u);
        HAL_GPIO_WritePin(M2_low_GPIO_Port, M2_low_Pin, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(M2_high_GPIO_Port, M2_high_Pin, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(M2_low_GPIO_Port, M2_low_Pin,
                          forward ? GPIO_PIN_SET : GPIO_PIN_RESET);
        HAL_GPIO_WritePin(M2_high_GPIO_Port, M2_high_Pin,
                          reverse ? GPIO_PIN_SET : GPIO_PIN_RESET);
        __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, compare);
    }
}

void Electromagnet_HardwareInit(void)
{
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    Electromagnet_Init(electromagnet_write);
}
