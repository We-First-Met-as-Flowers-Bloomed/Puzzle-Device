#ifndef __MOTOR_H
#define __MOTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* 包含头文件 */
#include "main.h"
#include "tim.h"

/* 对外函数声明 */
void Motor_Init(void);
void M1_left(int32_t Compare);
void M2_right(int32_t Compare);


#ifdef __cplusplus
}
#endif

#endif

