#ifndef FOUR_AXIS_H
#define FOUR_AXIS_H

#include <stdint.h>

typedef enum
{
    FOUR_AXIS_HOMING_IDLE = 0,
    FOUR_AXIS_HOMING_RUNNING,
    FOUR_AXIS_HOMING_DONE,
    FOUR_AXIS_HOMING_ERROR
} FourAxisHomingState_t;

typedef enum
{
    FOUR_AXIS_STARTUP_ENABLING = 0,
    FOUR_AXIS_STARTUP_RETURNING,
    FOUR_AXIS_STARTUP_WAITING,
    FOUR_AXIS_STARTUP_XY_ZEROING,
    FOUR_AXIS_STARTUP_READY,
    FOUR_AXIS_STARTUP_ERROR
} FourAxisStartupState_t;

typedef enum
{
    FOUR_AXIS_RECORDED_HOME_IDLE = 0,
    FOUR_AXIS_RECORDED_HOME_RUNNING,
    FOUR_AXIS_RECORDED_HOME_DONE,
    FOUR_AXIS_RECORDED_HOME_ERROR
} FourAxisRecordedHomeState_t;

typedef enum
{
    FOUR_AXIS_X = 0,
    FOUR_AXIS_Y,
    FOUR_AXIS_Z,
    FOUR_AXIS_R
} FourAxisId_t;

void FourAxis_Init(void);
void FourAxis_Process(void);
FourAxisStartupState_t FourAxis_GetStartupState(void);
const char *FourAxis_GetStartupText(void);

void FourAxis_StartAll360(void);
void FourAxis_PauseAll(void);
void FourAxis_ResumeAll(void);
void FourAxis_AbortAll(void);
uint8_t FourAxis_GetProgress(uint8_t axis);
uint8_t FourAxis_AllComplete(void);
void FourAxis_StartMove(uint8_t axis, uint8_t direction, uint16_t degrees);
void FourAxis_StartMovePulses(uint8_t axis, uint8_t direction, uint32_t pulses);
void FourAxis_StartMovePulsesAtSpeed(uint8_t axis, uint8_t direction,
                                    uint32_t pulses, uint16_t speed_rpm);
void FourAxis_SetTask1OpenLoop(uint8_t enabled);
uint8_t FourAxis_MoveComplete(uint8_t axis);
uint8_t FourAxis_MoveFailed(uint8_t axis);
void FourAxis_StartCalibration(uint8_t axis, uint16_t pulses_per_second);
void FourAxis_PauseCalibration(void);
void FourAxis_ResumeCalibration(void);
void FourAxis_StopCalibration(void);
uint32_t FourAxis_GetCalibrationPulses(void);

void FourAxis_HomingStart(void);
FourAxisHomingState_t FourAxis_HomingProcess(void);
FourAxisHomingState_t FourAxis_GetHomingState(void);

void FourAxis_ZeroAllSteppers(void);
void FourAxis_ZeroAxis(uint8_t axis);
uint8_t FourAxis_SetHomingZeroAll(void);
void FourAxis_CaptureHomeReference(void);
void FourAxis_RecordedHomeStart(void);
FourAxisRecordedHomeState_t FourAxis_RecordedHomeProcess(void);

#endif
