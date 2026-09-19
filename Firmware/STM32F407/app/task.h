#ifndef APP_TASK_H
#define APP_TASK_H

#include <stdint.h>
#include "vision_jsonl.h"

typedef enum
{
    TASK_STATE_IDLE = 0,
    TASK_STATE_RUNNING,
    TASK_STATE_PAUSED,
    TASK_STATE_DONE,
    TASK_STATE_ERROR
} TaskState_t;

typedef enum
{
    TASK_MAGNET_OFF = 0,
    TASK_MAGNET_ATTRACT,
    TASK_MAGNET_REPEL
} TaskMagnetMode_t;

typedef enum
{
    TASK_AXIS_X = 0,
    TASK_AXIS_Y,
    TASK_AXIS_Z,
    TASK_AXIS_R
} TaskAxis_t;

typedef struct
{
    void (*start_all_360)(void);
    void (*pause_all)(void);
    void (*resume_all)(void);
    void (*abort_all)(void);
    uint8_t (*get_axis_progress)(uint8_t axis);
    uint8_t (*all_complete)(void);
    void (*set_electromagnets)(uint8_t mode, uint16_t duty_permyriad);
    void (*start_axis_move)(uint8_t axis, uint8_t direction, uint16_t degrees);
    uint8_t (*axis_move_complete)(uint8_t axis);
    uint8_t (*axis_move_failed)(uint8_t axis);
    uint32_t (*get_tick_ms)(void);
    void (*start_calibration)(uint8_t axis, uint16_t pulses_per_second);
    void (*pause_calibration)(void);
    void (*resume_calibration)(void);
    void (*stop_calibration)(void);
    uint32_t (*get_calibration_pulses)(void);
    void (*start_axis_move_pulses)(uint8_t axis, uint8_t direction, uint32_t pulses);
    void (*start_axis_move_pulses_at_speed)(uint8_t axis, uint8_t direction,
                                            uint32_t pulses, uint16_t speed_rpm);
    void (*homing_start)(void);
    uint8_t (*homing_process)(void); /* 0=idle 1=running 2=done 3=error */
    void (*send_response)(const char *text);
    void (*set_beeper)(uint8_t on);
    void (*set_light)(uint8_t on);
    uint8_t (*set_homing_zero)(void);
    void (*capture_home_reference)(void);
    void (*recorded_home_start)(void);
    uint8_t (*recorded_home_process)(void);
    void (*set_task1_open_loop)(uint8_t enabled);
} TaskMotionOps_t;

void Task_Init(const TaskMotionOps_t *ops);
void Task_Process(void);
void Task_SelectNext(void);
void Task_AutoStartTask1(void);
void Task_StartOrTogglePause(void);
void Task_Exit(void);
void Task_SetError(void);
void Task_CameraRxByte(uint8_t byte);
uint8_t Task_SubmitVisionPlan(const VisionPlan_t *plan);
uint32_t Task_CameraPulseToMechanical(uint8_t axis, uint32_t camera_pulse);

uint8_t Task_GetCurrentId(void);
uint8_t Task_GetProgress(void);
TaskState_t Task_GetState(void);
const char *Task_GetStateText(void);
uint8_t Task_IsConfigured(void);
const char *Task_GetOperationText(void);

#endif
