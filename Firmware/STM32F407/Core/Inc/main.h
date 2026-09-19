/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define M1_low_Pin GPIO_PIN_2
#define M1_low_GPIO_Port GPIOC
#define M1_high_Pin GPIO_PIN_3
#define M1_high_GPIO_Port GPIOC
#define M1_encoder_b_Pin GPIO_PIN_1
#define M1_encoder_b_GPIO_Port GPIOA
#define m0_tx_Pin GPIO_PIN_2
#define m0_tx_GPIO_Port GPIOA
#define m0_rx_Pin GPIO_PIN_3
#define m0_rx_GPIO_Port GPIOA
#define M2_low_Pin GPIO_PIN_4
#define M2_low_GPIO_Port GPIOA
#define M2_high_Pin GPIO_PIN_5
#define M2_high_GPIO_Port GPIOA
#define M2_encoder_a_Pin GPIO_PIN_6
#define M2_encoder_a_GPIO_Port GPIOA
#define M2_encoder_b_Pin GPIO_PIN_7
#define M2_encoder_b_GPIO_Port GPIOA
#define beep_Pin GPIO_PIN_0
#define beep_GPIO_Port GPIOB
#define light_Pin GPIO_PIN_1
#define light_GPIO_Port GPIOB
#define M1_left_Pin GPIO_PIN_13
#define M1_left_GPIO_Port GPIOE
#define M2_right_Pin GPIO_PIN_14
#define M2_right_GPIO_Port GPIOE
#define jg_scl_Pin GPIO_PIN_10
#define jg_scl_GPIO_Port GPIOB
#define jg_sda_Pin GPIO_PIN_11
#define jg_sda_GPIO_Port GPIOB
#define key1_Pin GPIO_PIN_12
#define key1_GPIO_Port GPIOB
#define key1_EXTI_IRQn EXTI15_10_IRQn
#define key2_Pin GPIO_PIN_13
#define key2_GPIO_Port GPIOB
#define key2_EXTI_IRQn EXTI15_10_IRQn
#define key3_Pin GPIO_PIN_14
#define key3_GPIO_Port GPIOB
#define key3_EXTI_IRQn EXTI15_10_IRQn
#define key4_Pin GPIO_PIN_15
#define key4_GPIO_Port GPIOB
#define key4_EXTI_IRQn EXTI15_10_IRQn
#define x_tx_Pin GPIO_PIN_8
#define x_tx_GPIO_Port GPIOD
#define x_rx_Pin GPIO_PIN_9
#define x_rx_GPIO_Port GPIOD
#define y_tx_Pin GPIO_PIN_6
#define y_tx_GPIO_Port GPIOC
#define y_rx_Pin GPIO_PIN_7
#define y_rx_GPIO_Port GPIOC
#define imu_sda_Pin GPIO_PIN_9
#define imu_sda_GPIO_Port GPIOC
#define imu_scl_Pin GPIO_PIN_8
#define imu_scl_GPIO_Port GPIOA
#define DAP_tx_Pin GPIO_PIN_9
#define DAP_tx_GPIO_Port GPIOA
#define DAP_rx_Pin GPIO_PIN_10
#define DAP_rx_GPIO_Port GPIOA
#define M1_encoder_a_Pin GPIO_PIN_15
#define M1_encoder_a_GPIO_Port GPIOA
#define x_dir_Pin GPIO_PIN_10
#define x_dir_GPIO_Port GPIOC
#define g1_Pin GPIO_PIN_0
#define g1_GPIO_Port GPIOD
#define g2_Pin GPIO_PIN_1
#define g2_GPIO_Port GPIOD
#define g5_Pin GPIO_PIN_4
#define g5_GPIO_Port GPIOD
#define g6_Pin GPIO_PIN_5
#define g6_GPIO_Port GPIOD
#define g7_Pin GPIO_PIN_6
#define g7_GPIO_Port GPIOD
#define g8_Pin GPIO_PIN_7
#define g8_GPIO_Port GPIOD
#define oled_scl_Pin GPIO_PIN_6
#define oled_scl_GPIO_Port GPIOB
#define oled_sda_Pin GPIO_PIN_7
#define oled_sda_GPIO_Port GPIOB
#define x_stp_Pin GPIO_PIN_8
#define x_stp_GPIO_Port GPIOB
#define y_stp_Pin GPIO_PIN_9
#define y_stp_GPIO_Port GPIOB
#define x_en_Pin GPIO_PIN_0
#define x_en_GPIO_Port GPIOE
#define y_en_Pin GPIO_PIN_1
#define y_en_GPIO_Port GPIOE

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
