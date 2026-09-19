/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "oled.h"
#include "key_input.h"
#include "four_axis.h"
#include "status_ui.h"
#include "electromagnet.h"
#include "task.h"
#include "vision_jsonl.h"
#include "debug_uart.h"
#include "dma_rx_cursor.h"
#include "stdio.h"
#include "string.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define KEY_PIN_TO_ID(pin) \
    ((pin) == key1_Pin ? KEY1_PIN_ID : \
    ((pin) == key2_Pin ? KEY2_PIN_ID : \
    ((pin) == key3_Pin ? KEY3_PIN_ID : KEY4_PIN_ID)))
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
static uint8_t vofa_rx_byte;
static char vision_rx_log[1024];
static char host_rx_log[256];
static volatile uint16_t host_rx_size;
static volatile uint8_t host_rx_ready;
#define USART2_DMA_RX_SIZE 1024u
static uint8_t usart2_dma_rx[USART2_DMA_RX_SIZE];
static DmaRxCursor_t usart2_cursor;
static volatile uint16_t usart2_dma_producer;
static volatile uint8_t usart2_dma_event_pending;
static volatile uint8_t usart2_recovery_pending;
static volatile uint32_t usart2_pending_error;
static uint32_t usart2_recovery_count;
static uint32_t usart2_last_byte_tick;
static uint16_t usart2_partial_bytes;
static uint8_t usart2_partial_reported;
static uint8_t task1_auto_started;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* printf 重定向到 USART1（DAPLink 虚拟串口），防止 semihosting 导致卡死 */
#ifdef __GNUC__
  #define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
  #define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif

PUTCHAR_PROTOTYPE
{
    DebugUart_PutChar(ch);
    return ch;
}

static void Task_SetElectromagnets(uint8_t mode, uint16_t duty_permyriad)
{
    Electromagnet_Set(0u, (ElectromagnetMode_t)mode, duty_permyriad);
}

static uint8_t Task_HomingProcess(void)
{
    return (uint8_t)FourAxis_HomingProcess();
}

static uint8_t Task_RecordedHomeProcess(void)
{
    return (uint8_t)FourAxis_RecordedHomeProcess();
}

static void Task_SendResponse(const char *text)
{
    /* USART1 is the interactive host port. USART2 is receive-only JSONL. */
    DebugUart_Printf("%s\r\n", text);
}

static void Task_SetBeeper(uint8_t on)
{
    /* PB0 = buzzer, active high. */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void Task_SetLight(uint8_t on)
{
    HAL_GPIO_WritePin(light_GPIO_Port, light_Pin, on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static const TaskMotionOps_t task_motion_ops = {
    FourAxis_StartAll360,
    FourAxis_PauseAll,
    FourAxis_ResumeAll,
    FourAxis_AbortAll,
    FourAxis_GetProgress,
    FourAxis_AllComplete,
    Task_SetElectromagnets,
    FourAxis_StartMove,
    FourAxis_MoveComplete,
    FourAxis_MoveFailed,
    HAL_GetTick,
    FourAxis_StartCalibration,
    FourAxis_PauseCalibration,
    FourAxis_ResumeCalibration,
    FourAxis_StopCalibration,
    FourAxis_GetCalibrationPulses,
    FourAxis_StartMovePulses,
    FourAxis_StartMovePulsesAtSpeed,
    FourAxis_HomingStart,
    Task_HomingProcess,
    Task_SendResponse,
    Task_SetBeeper,
    Task_SetLight,
    FourAxis_SetHomingZeroAll,
    FourAxis_CaptureHomeReference,
    FourAxis_RecordedHomeStart,
    Task_RecordedHomeProcess,
    FourAxis_SetTask1OpenLoop
};

static void USART2_StartDmaRx(void)
{
    if (HAL_UARTEx_ReceiveToIdle_DMA(&huart2, usart2_dma_rx,
                                     sizeof(usart2_dma_rx)) == HAL_OK)
        __HAL_DMA_DISABLE_IT(huart2.hdmarx, DMA_IT_HT);
}

static void USART2_ProcessDma(void)
{
    DmaRxSpan_t spans[2];
    uint16_t producer;
    uint8_t count;
    uint8_t span;
    uint32_t primask;
    if (usart2_recovery_pending)
    {
        uint32_t error;
        primask = __get_PRIMASK();
        __disable_irq();
        error = usart2_pending_error;
        usart2_recovery_pending = 0u;
        usart2_dma_event_pending = 0u;
        if (primask == 0u) __enable_irq();
        (void)HAL_UART_AbortReceive(&huart2);
        DmaRxCursor_Reset(&usart2_cursor);
        usart2_dma_producer = 0u;
        USART2_StartDmaRx();
        ++usart2_recovery_count;
        DebugUart_Printf("[USART2 ERROR] code=%08lX RECOVER=%lu\r\n",
                         (unsigned long)error,
                         (unsigned long)usart2_recovery_count);
    }
    if (usart2_dma_event_pending)
    {
        primask = __get_PRIMASK();
        __disable_irq();
        producer = usart2_dma_producer;
        usart2_dma_event_pending = 0u;
        if (primask == 0u) __enable_irq();
        count = DmaRxCursor_Update(&usart2_cursor, producer, spans);
        for (span = 0u; span < count; ++span)
        {
            uint16_t i;
            for (i = 0u; i < spans[span].size; ++i)
            {
                uint8_t byte = usart2_dma_rx[spans[span].offset + i];
                VisionJsonl_RxByte(byte);
                usart2_last_byte_tick = HAL_GetTick();
                if (byte == '\n')
                {
                    usart2_partial_bytes = 0u;
                    usart2_partial_reported = 0u;
                }
                else if (usart2_partial_bytes < 0xFFFFu)
                    ++usart2_partial_bytes;
            }
        }
    }
    if ((usart2_partial_bytes != 0u) && !usart2_partial_reported &&
        ((uint32_t)(HAL_GetTick() - usart2_last_byte_tick) >= 500u))
    {
        DebugUart_Printf("[USART2 PARTIAL] bytes=%u NO NEWLINE\r\n",
                         usart2_partial_bytes);
        usart2_partial_reported = 1u;
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C1_Init();
  MX_I2C2_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_USART1_UART_Init();
  MX_USART2_UART_Init();
  MX_USART3_UART_Init();
  MX_USART6_UART_Init();
  MX_I2C3_Init();
  MX_TIM10_Init();
  MX_TIM11_Init();
  MX_UART4_Init();
  MX_UART5_Init();
  /* USER CODE BEGIN 2 */
  DebugUart_Init();
  KeyInput_Init();
  FourAxis_Init();
  Electromagnet_HardwareInit();
  Task_Init(&task_motion_ops);
  VisionJsonl_Init();
  DmaRxCursor_Init(&usart2_cursor, sizeof(usart2_dma_rx));
  StatusUi_Init();
  USART2_StartDmaRx();
  HAL_UART_Receive_IT(&huart1, &vofa_rx_byte, 1u);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    uint32_t key_tick = HAL_GetTick();
    USART2_ProcessDma();
    if (host_rx_ready)
    {
      char host_line[sizeof(host_rx_log)];
      uint32_t primask = __get_PRIMASK();
      __disable_irq();
      strcpy(host_line, host_rx_log);
      host_rx_ready = 0u;
      host_rx_size = 0u;
      if (primask == 0u) __enable_irq();
      DebugUart_LogTextRx("USART1", "HOST", host_line);
    }
    FourAxisHomingState_t homing;
    KeyInput_UpdateLevel(KEY1_PIN_ID,
      HAL_GPIO_ReadPin(key1_GPIO_Port, key1_Pin) == GPIO_PIN_RESET, key_tick);
    KeyInput_UpdateLevel(KEY2_PIN_ID,
      HAL_GPIO_ReadPin(key2_GPIO_Port, key2_Pin) == GPIO_PIN_RESET, key_tick);
    KeyInput_UpdateLevel(KEY3_PIN_ID,
      HAL_GPIO_ReadPin(key3_GPIO_Port, key3_Pin) == GPIO_PIN_RESET, key_tick);
    KeyInput_UpdateLevel(KEY4_PIN_ID,
      HAL_GPIO_ReadPin(key4_GPIO_Port, key4_Pin) == GPIO_PIN_RESET, key_tick);
    KeyEvent_t event = KeyInput_TakeEvent();

    if (event == KEY_EVENT_RESET)
    {
      StatusUi_LogKey("RESET");
      NVIC_SystemReset();
    }
    FourAxis_Process();
    homing = FourAxis_GetHomingState();
    if (FourAxis_GetStartupState() == FOUR_AXIS_STARTUP_READY)
    {
      if (!task1_auto_started)
      {
        Task_AutoStartTask1();
        task1_auto_started = 1u;
      }
      VisionPlan_t vision_plan;
      char vision_status[32];
      VisionJsonlResult_t vision_result =
        VisionJsonl_ProcessWithRaw(&vision_plan, vision_status,
                                   sizeof(vision_status), vision_rx_log,
                                   sizeof(vision_rx_log));
      if (vision_result != VISION_JSONL_NONE)
        StatusUi_ShowTransient("UART2 RX");
      if (vision_result != VISION_JSONL_NONE)
        DebugUart_LogTextRx("USART2", "CAM", vision_rx_log);
      if (vision_result == VISION_JSONL_PLAN)
      {
        if (Task_SubmitVisionPlan(&vision_plan))
        {
          StatusUi_ShowTransient("JSON ACCEPT");
          printf("[JSONL] ACCEPT count=%u elapsed=%lu\r\n",
                 vision_plan.count, (unsigned long)vision_plan.elapsed_ms);
        }
        else
        {
          StatusUi_ShowTransient("TASK BUSY");
          printf("[JSONL] REJECT MACHINE BUSY/NOT TASK1\r\n");
        }
      }
      else if (vision_result == VISION_JSONL_REPORT)
      {
        StatusUi_ShowTransient("UART2 REPORT");
        printf("[JSONL] REPORT status=%s NO MOTION\r\n", vision_status);
      }
      else if (vision_result == VISION_JSONL_REJECT)
      {
        StatusUi_ShowTransient("JSON INVALID");
        printf("[JSONL] REJECT INVALID PLAN\r\n");
      }
      if (event == KEY_EVENT_NEXT)
      {
        StatusUi_LogKey("NEXT");
        Task_SelectNext();
      }
      else if (event == KEY_EVENT_TOGGLE)
      {
        StatusUi_LogKey("START/PAUSE");
        Task_StartOrTogglePause();
      }
      else if (event == KEY_EVENT_EXIT)
      {
        StatusUi_LogKey("EXIT");
        Task_Exit();
      }
      Task_Process();
    }
    StatusUi_Process(homing);
		
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if ((GPIO_Pin == key1_Pin) || (GPIO_Pin == key2_Pin) ||
      (GPIO_Pin == key3_Pin) || (GPIO_Pin == key4_Pin))
  {
    KeyInput_RecordIsr(KEY_PIN_TO_ID(GPIO_Pin), HAL_GetTick());
  }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART1)
  {
    /* VOFA bridge: USART1 command bytes enter the same line parser. */
    Task_CameraRxByte(vofa_rx_byte);
    if (!host_rx_ready)
    {
      if ((vofa_rx_byte == '\n') || (vofa_rx_byte == '\r'))
      {
        if (host_rx_size != 0u)
        {
          host_rx_log[host_rx_size] = '\0';
          host_rx_ready = 1u;
        }
      }
      else if (host_rx_size < (sizeof(host_rx_log) - 1u))
      {
        host_rx_log[host_rx_size++] = (char)vofa_rx_byte;
      }
      else host_rx_size = 0u;
    }
    HAL_UART_Receive_IT(&huart1, &vofa_rx_byte, 1u);
  }
}

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
  if (huart->Instance == USART2)
  {
    usart2_dma_producer = Size;
    usart2_dma_event_pending = 1u;
  }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    usart2_pending_error = huart->ErrorCode;
    usart2_recovery_pending = 1u;
  }
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
  DebugUart_TxComplete(huart);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
