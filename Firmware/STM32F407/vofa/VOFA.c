#include <main.h>
#include "vofa.h"
#include "stm32f4xx_hal.h"
#include "motor.h"
#include "stdio.h"
#include "jy901s_i2c.h"

uint8_t USART1_RXbuff;


extern uint8_t g1, g2, g3 , g4 , g5 , g6 , g7, g8 ;
extern int32_t  err_encoder1 ,err_encoder2;
//extern MPU6050_t MPU6050;
void Float_to_Byte(float f, unsigned char byte[])
{
	FloatLongType fl;
	fl.fdata = f;
	byte[0] = (unsigned char)fl.ldata;
	byte[1] = (unsigned char)(fl.ldata >> 8);
	byte[2] = (unsigned char)(fl.ldata >> 16);
	byte[3] = (unsigned char)(fl.ldata >> 24);
}

void Send_Data(UART_HandleTypeDef *huart, float f)
{
	unsigned char byte[4] = {0};

	Float_to_Byte(f, byte);
	HAL_UART_Transmit(huart, (uint8_t *)&byte[0], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
	HAL_UART_Transmit(huart, (uint8_t *)&byte[1], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
	HAL_UART_Transmit(huart, (uint8_t *)&byte[2], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
	HAL_UART_Transmit(huart, (uint8_t *)&byte[3], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
}

void Send_Tail(UART_HandleTypeDef *huart)
{
	unsigned char byte[4] = {0x00, 0x00, 0x80, 0x7f};

	HAL_UART_Transmit(huart, (uint8_t *)&byte[0], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
	HAL_UART_Transmit(huart, (uint8_t *)&byte[1], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
	HAL_UART_Transmit(huart, (uint8_t *)&byte[2], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
	HAL_UART_Transmit(huart, (uint8_t *)&byte[3], 1, 0xffff);
	while (HAL_UART_GetState(huart) == HAL_UART_STATE_BUSY_TX)
		;
}




// Add Send_Date

unsigned char shujv[12];
unsigned char m0shuju[12];

void vodka_JustFloat_send(UART_HandleTypeDef *huart)
{
	const jy901s_data_t* jy = JY901S_Get_Data();

	/* 原有通道：编码器误差和状态变量 a */
	Send_Data(huart, (float)err_encoder1);
	Send_Data(huart, (float)err_encoder2);

	/* 新增通道：JY901S 姿态角 roll / pitch / yaw（离线时填 0） */
	if (jy->online) {
		Send_Data(huart, jy->angle.roll);
		Send_Data(huart, jy->angle.pitch);
		Send_Data(huart, jy->angle.yaw);
	} else {
		Send_Data(huart, 0.0f);
		Send_Data(huart, 0.0f);
		Send_Data(huart, 0.0f);
	}

			Send_Data(huart, g1);
			Send_Data(huart, g2);
				Send_Data(huart, g3);
				Send_Data(huart, g4);
				Send_Data(huart, g5);
				Send_Data(huart, g6);
				Send_Data(huart, g7);
				Send_Data(huart, g8);
	Send_Tail(huart);
}

/**example**/
// while(1)
// {
// 	vodka_JustFloat_send(&huart6);
// }




int test_flag = 1;
int vofa_i = 0, vofa_I = 0;
float vofa_float;


volatile int encoder=0,encoderh8=0,encoderl8;


void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{

	
}


