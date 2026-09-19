#include "status_ui.h"

#include "oled.h"
#include "task.h"

#include "stm32f4xx_hal.h"
#include <stdio.h>

static uint32_t last_refresh;
static TaskState_t last_state;
static uint8_t last_task;
static uint8_t last_progress_bucket;
static char transient_message[24];
static uint32_t transient_started;
static uint8_t transient_active;

void StatusUi_Init(void)
{
    OLED_Init();
    OLED_Clear();
    last_refresh = 0u;
    last_state = (TaskState_t)0xFF;
    last_task = 0u;
    last_progress_bucket = 0xFFu;
    transient_message[0] = '\0';
    transient_started = 0u;
    transient_active = 0u;
    printf("[SYSTEM] Four-axis controller boot\r\n");
}

void StatusUi_LogKey(const char *key_name)
{
    printf("[KEY] %s\r\n", key_name);
}

void StatusUi_ShowTransient(const char *message)
{
    uint32_t now = HAL_GetTick();
    snprintf(transient_message, sizeof(transient_message), "%s",
             (message != NULL) ? message : "");
    transient_started = now;
    transient_active = (transient_message[0] != '\0') ? 1u : 0u;
    last_refresh = now - 100u;
}

void StatusUi_Process(FourAxisHomingState_t homing_state)
{
    uint32_t now = HAL_GetTick();
    char line[24];
    FourAxisStartupState_t startup = FourAxis_GetStartupState();
    TaskState_t state = Task_GetState();
    uint8_t task = Task_GetCurrentId();
    uint8_t progress = Task_GetProgress();
    uint8_t bucket = (uint8_t)(progress / 5u);

    if (startup != FOUR_AXIS_STARTUP_READY)
    {
        if ((uint32_t)(now - last_refresh) >= 100u)
        {
            last_refresh = now;
            OLED_ClearLine(0);
            OLED_ShowString(0, 0, (uint8_t *)"STEPPER START", 12);
            OLED_ClearLine(2);
            OLED_ShowString(0, 2, (uint8_t *)FourAxis_GetStartupText(), 12);
            OLED_ClearLine(4);
            OLED_ShowString(0, 4,
                (uint8_t *)((startup == FOUR_AXIS_STARTUP_ERROR) ?
                            "CHECK UART/MOTOR" : "PLEASE WAIT"), 12);
            OLED_ClearLine(6);
        }
        return;
    }

    if (homing_state == FOUR_AXIS_HOMING_RUNNING)
    {
        if ((uint32_t)(now - last_refresh) >= 100u)
        {
            last_refresh = now;
            OLED_ClearLine(0);
            OLED_ShowString(0, 0, (uint8_t *)"HOMING Z R", 12);
            OLED_ClearLine(2);
            OLED_ShowString(0, 2, (uint8_t *)"PLEASE WAIT", 12);
        }
        return;
    }

    if ((state != last_state) || (task != last_task))
    {
        printf("[TASK] id=%u state=%s\r\n", task, Task_GetStateText());
        last_state = state;
        last_task = task;
    }
    if (bucket != last_progress_bucket)
    {
        printf("[TASK] id=%u progress=%u%%\r\n", task, progress);
        last_progress_bucket = bucket;
    }

    if ((uint32_t)(now - last_refresh) < 100u)
    {
        return;
    }
    last_refresh = now;
    snprintf(line, sizeof(line), "TASK: %u/9", task);
    OLED_ClearLine(0);
    OLED_ShowString(0, 0, (uint8_t *)line, 12);
    snprintf(line, sizeof(line), "STATE: %s", Task_GetStateText());
    OLED_ClearLine(2);
    OLED_ShowString(0, 2, (uint8_t *)line, 12);
    if (!Task_IsConfigured())
    {
        snprintf(line, sizeof(line), "NOT CONFIGURED");
    }
    else
    {
        snprintf(line, sizeof(line), "PROGRESS: %03u%%", progress);
    }
    OLED_ClearLine(4);
    OLED_ShowString(0, 4, (uint8_t *)line, 12);
    OLED_ClearLine(6);
    if (transient_active &&
        ((uint32_t)(now - transient_started) < 1000u))
    {
        OLED_ShowString(0, 6, (uint8_t *)transient_message, 12);
    }
    else
    {
        transient_active = 0u;
        OLED_ShowString(0, 6, (uint8_t *)Task_GetOperationText(), 12);
    }
}
