#include <math.h>
#include "pid.h"
#include "stdint.h"
#include <stddef.h>
#define PID_LIMIT_MIN -800     //PID输出最低值
#define PID_LIMIT_MAX 800 //PID输出最大值
//注意：PID结构体必须定义为全局变量或静态变量，然后在函数中给KP,KI,KD赋值
/************************采样周期未知且不变************************************/
//位置式PID
//pwm=Kp*e(k)+Ki*∑e(k)+Kd[e（k）-e(k-1)]
//setvalue : 设置值（期望值）
//actualvalue: 实际值
//由于全量输出，每次输出均与过去状态有关，计算时要对ek累加，计算量大
//PID_LocTypeDef Speed1Pid = {
//        .kp = 60,
//        .ki = 50,
//        .kd = 0,
//        .ek = 0,
//        .ek1 = 0,
//        .ek2 = 0,
//        .location_sum = 0,
//        .out = 0,

//};
//PID_LocTypeDef Speed2Pid = {
//        .kp = 60,
//        .ki = 50,
//        .kd = 0,
//        .ek = 0,
//        .ek1 = 0,
//        .ek2 = 0,
//        .location_sum = 0,
//        .out = 0,

//};

//PID_LocTypeDef TurnPid = {
//        .kp = 30,
//        .ki = 0,
//        .kd = 0,
//        .ek = 0,
//        .ek1 = 0,
//        .ek2 = 0,
//        .location_sum = 0,
//        .out = 0,

//};


float PID_location(float setvalue, float actualvalue, PSpeedPIDControl_Struct PID)
{
    PID->ek =setvalue-actualvalue;
    PID->location_sum += PID->ek;                         //计算累计误差值
    if((PID->ki!=0)&&(PID->location_sum>(PID_LIMIT_MAX/PID->ki))) PID->location_sum=PID_LIMIT_MAX/PID->ki;
    if((PID->ki!=0)&&(PID->location_sum<(PID_LIMIT_MIN/PID->ki))) PID->location_sum=PID_LIMIT_MIN/PID->ki;//积分限幅

  PID->out=PID->kp*PID->ek+(PID->ki*PID->location_sum)+PID->kd*(PID->ek-PID->ek1);
  PID->ek1 = PID->ek;
    if(PID->out<PID_LIMIT_MIN)  PID->out=PID_LIMIT_MIN;
    if(PID->out>PID_LIMIT_MAX)  PID->out=PID_LIMIT_MAX;//PID->out限幅

    return PID->out;
}
//增量式PID
//pidout+=Kp[e（k）-e(k-1)]+Ki*e(k)+Kd[e(k)-2e(k-1)+e(k-2)]
//setvalue : 设置值（期望值）
//actualvalue: 实际值
float PID_increment(float setvalue, float actualvalue, PSpeedPIDControl_Struct PID)
{
    PID->ek =setvalue-actualvalue;
  PID->out+=PID->kp*(PID->ek-PID->ek1)+PID->ki*PID->ek+PID->kd*(PID->ek-2*PID->ek1+PID->ek2);
//  PID->out+=PID->kp*PID->ek-PID->ki*PID->ek1+PID->kd*PID->ek2;
  PID->ek2 = PID->ek1;
  PID->ek1 = PID->ek;

    if(PID->out<PID_LIMIT_MIN)  PID->out=PID_LIMIT_MIN;
    if(PID->out>PID_LIMIT_MAX)  PID->out=PID_LIMIT_MAX;//限幅

    return PID->out;
}
PID_LocTypeDef Turn_Pid = {

    .kp = 3.25f,

    .ki = 0.0f,

    .kd = 0.0f,

    .ek = 0.0f,

    .ek1 = 0.0f,

    .ek2 = 0.0f,

    .location_sum = 0.0f,

    .out = 0.0f,

};





PID_LocTypeDef anglePID;

PID_LocTypeDef speedLPID = {

    .kp = 35.0f,

    .ki = 0.0f,

    .kd = 1.0f,

    .ek = 0.0f,

    .ek1 = 0.0f,

    .ek2 = 0.0f,

    .location_sum = 0.0f,

    .out = 0.0f,

};

PID_LocTypeDef speedRPID = {

    .kp = 35.0f,

    .ki = 0.0f,

    .kd = 1.0f,

    .ek = 0.0f,

    .ek1 = 0.0f,

    .ek2 = 0.0f,

    .location_sum = 0.0f,

    .out = 0.0f,

};





float Velocity_Kp=35,Velocity_Ki=0.0,Velocity_Kd=1.0;		//速度环 

float Turn_Kp=3.25,Turn_Kd=0;							//转向环



static void PID_UpdateGains(PID_LocTypeDef *pid, float kp, float ki, float kd)

{

    if(pid == NULL)

    {

        return;

    }

    pid->kp = kp;

    pid->ki = ki;

    pid->kd = kd;

}



int Velocity(float Target, float actual, PID_LocTypeDef *pid)

{

    if(pid == NULL)

    {

        return 0;

    }



    PID_UpdateGains(pid, Velocity_Kp, Velocity_Ki, Velocity_Kd);

    return (int)PID_location(Target, actual, pid);

}





   // PID_UpdateGains(pid, Velocity_Kp, Velocity_Ki, Velocity_Kd);

    //int Velocity(int Target,int encoder)
//{
//	static int Err_LowOut_last,Encoder_S;
//	static float a=0.7;
//	int Err,Err_LowOut,temp;
////	Velocity_Ki=Velocity_Kp/200;
//	//1、计算偏差值
//	Err=encoder-Target;
//	//2、低通滤波
//	Err_LowOut=(1-a)*Err+a*Err_LowOut_last;
//	Err_LowOut_last=Err_LowOut;
//	//3、积分
//	Encoder_S+=Err_LowOut;
//	//4、积分限幅(-20000~20000)
//	Encoder_S=Encoder_S>10000?10000:(Encoder_S<(-10000)?(-10000):Encoder_S);
//	
//	//5、速度环计算
//	temp=Velocity_Kp*Err_LowOut+Velocity_Ki*Encoder_S;
//	return temp;
//}


int Turn(float targetYaw,float currentYaw)

{

    PID_UpdateGains(&Turn_Pid, Turn_Kp, 0.0f, Turn_Kd);

    return (int)PID_location(targetYaw, currentYaw, &Turn_Pid);

}



int angleOutput;

float targetSpeedL;

float targetSpeedR;



// void cascadeControl(

//                    float targetYaw,

//                    float baseSpeed,

//                    float currentYaw,

//                    float EncoderLeft,

//                    float EncoderRight,

//                    int* MotorL,

//                    int* MotorR) {

//     angleOutput = Turn(targetYaw, currentYaw);



//     targetSpeedL = baseSpeed - angleOutput;

//     targetSpeedR = baseSpeed + angleOutput;



//     if(MotorL != NULL) {

//         *MotorL = Velocity(targetSpeedL, EncoderLeft, &speedLPID);

//     }

//     if(MotorR != NULL) {

//         *MotorR = Velocity(targetSpeedR, EncoderRight, &speedRPID);

//     }

// }

