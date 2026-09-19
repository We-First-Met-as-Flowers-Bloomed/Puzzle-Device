#include "motor.h"

/* ==================== 宏定义 ==================== */
// 定时器句柄定义
#define PWM_TIM1    htim1
#define EXTI_TIM1    htim4
// PWM 最大限幅（与你的ARR匹配）
#define PWM_MAX     8000

/* ==================== 静态函数声明 ==================== */
// 内部辅助函数：设置GPIO方向
static void Motor_SetDir(GPIO_TypeDef *port, uint16_t pin1, uint16_t pin2, GPIO_PinState state1, GPIO_PinState state2);

/* ==================== 对外接口函数 ==================== */

/**
  * @brief  电机驱动初始化
  * @param  无
  * @retval 无
  */
void Motor_Init(void)
{
    /* 开启定时器更新中断 */
    __HAL_TIM_ENABLE_IT(&EXTI_TIM1, TIM_IT_UPDATE);
    /* 启动PWM输出通道 */
    HAL_TIM_PWM_Start(&PWM_TIM1, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&PWM_TIM1, TIM_CHANNEL_4);

    /* 初始化所有电机方向引脚为复位状态 */
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_2 | GPIO_PIN_3  , GPIO_PIN_RESET);
	  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4 | GPIO_PIN_5  , GPIO_PIN_RESET);
}

/**
  * @brief  电机1控制（右）
  * @param  Compare: PWM占空比（正负表示方向）
  * @retval 无
  */
void M1_left(int32_t Compare)
{
    // 限幅处理
    if(Compare > PWM_MAX) Compare = PWM_MAX;
    if(Compare < -PWM_MAX) Compare = -PWM_MAX;

    if(Compare >= 0)
    {
        __HAL_TIM_SET_COMPARE(&PWM_TIM1, TIM_CHANNEL_3, Compare);
        Motor_SetDir(GPIOC, GPIO_PIN_2, GPIO_PIN_3, GPIO_PIN_SET, GPIO_PIN_RESET);
    }
    else
    {
        __HAL_TIM_SET_COMPARE(&PWM_TIM1, TIM_CHANNEL_3, -Compare);
        Motor_SetDir(GPIOC, GPIO_PIN_2, GPIO_PIN_3, GPIO_PIN_RESET, GPIO_PIN_SET);
    }
}

/**
  * @brief  电机2控制（左）
  * @param  Compare: PWM占空比（正负表示方向）
  * @retval 无
  */
void M2_right(int32_t Compare)
{
    if(Compare > PWM_MAX) Compare = PWM_MAX;
    if(Compare < -PWM_MAX) Compare = -PWM_MAX;

    if(Compare >= 0)
    {
        __HAL_TIM_SET_COMPARE(&PWM_TIM1, TIM_CHANNEL_4, Compare);
        Motor_SetDir(GPIOA, GPIO_PIN_4, GPIO_PIN_5, GPIO_PIN_SET, GPIO_PIN_RESET);
    }
    else
    {
        __HAL_TIM_SET_COMPARE(&PWM_TIM1, TIM_CHANNEL_4, -Compare);
        Motor_SetDir(GPIOA, GPIO_PIN_4, GPIO_PIN_5, GPIO_PIN_RESET, GPIO_PIN_SET);
    }
}



/* ==================== 静态内部函数 ==================== */

/**
  * @brief  电机方向引脚设置（内部复用函数）
  */
static void Motor_SetDir(GPIO_TypeDef *port, uint16_t pin1, uint16_t pin2, GPIO_PinState state1, GPIO_PinState state2)
{
    HAL_GPIO_WritePin(port, pin1, state1);
    HAL_GPIO_WritePin(port, pin2, state2);
}






