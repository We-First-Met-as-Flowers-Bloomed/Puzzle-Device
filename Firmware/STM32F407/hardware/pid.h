#ifndef __PID_H
#define __PID_H

#include <math.h>
// #include "zf_common_headfile.h"
typedef struct
{
  float kp;                       //比例系数Proportional
  float ki;                       //积分系数Integral
  float kd;                       //微分系数Derivative
//  float ti;                       //积分时间常数
//  float td;                       //微分时间常数
//  float period;                                       //采样周期
  float ek;                       //当前误差
  float ek1;                      //前一次误差e(k-1)
  float ek2;                      //再前一次误差e(k-2)
  float location_sum;             //累计积分位置
    float out;                                          //PID输出值
}PID_LocTypeDef;

typedef  PID_LocTypeDef *PSpeedPIDControl_Struct;
float PID_location(float setvalue, float actualvalue, PSpeedPIDControl_Struct PID);
float PID_increment(float setvalue, float actualvalue, PSpeedPIDControl_Struct PID);
void cascadeControl(
                   float targetYaw,
                   float baseSpeed,
                   float currentYaw,
                   float EncoderLeft,
                   float EncoderRight,
                   int* MotorL,
                   int* MotorR);
extern PID_LocTypeDef Turn_Pid;
extern PID_LocTypeDef speedLPID;
extern PID_LocTypeDef speedRPID;
int Velocity(float Target,float actual,PID_LocTypeDef *pid);
#endif

