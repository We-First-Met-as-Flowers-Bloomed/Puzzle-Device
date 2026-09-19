#include "task.h"

/* ============================================================================
 * TASK LOGIC LIVES IN THIS FILE ONLY.
 *
 * Every hardware capability is already exposed through TaskMotionOps_t
 * (wired up in main.c), so a new task never requires editing other files:
 *   start_all_360 / pause_all / resume_all / abort_all   - task 1 primitives
 *   get_axis_progress / all_complete                     - task 1 feedback
 *   set_electromagnets(mode, duty)                       - OFF/ATTRACT/REPEL
 *   start_axis_move(axis, dir, degrees)                  - relative move (deg)
 *   start_axis_move_pulses(axis, dir, pulses)            - relative move (pls)
 *   axis_move_complete(axis)                             - poll move done
 *   get_tick_ms()                                        - HAL millisecond tick
 *   start/pause/resume/stop_calibration + get_.._pulses  - tasks 4-7 helpers
 *   homing_start / homing_process                        - X/Y/Z saved-zero return
 *   send_response(text)                                  - reply on USART1+2
 *   set_beeper(on) / set_light(on)                       - PB0 buzzer, PB1 LED
 * Serial lines arriving on USART1/USART2 are collected by Task_CameraRxByte
 * and surfaced in Task_Process(); parse them here (see parse_piece_line).
 *
 * To add a task: extend task_table, add a start branch in
 * Task_StartOrTogglePause, route Task_Process, and (if scripted) push
 * TaskCommand_t steps like task8 does.
 *
 * Completion feedback is automatic: whenever a task enters TASK_STATE_DONE
 * the LED turns on (until the task leaves DONE back to IDLE) and the buzzer
 * beeps once for DONE_BEEP_MS.
 * ==========================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t configured; } TaskRegistration_t;
typedef enum { CMD_AXIS = 0, CMD_MAGNET, CMD_AXIS_PULSES } CommandType_t;
typedef struct
{
    CommandType_t type;
    uint8_t axis;
    uint8_t direction;
    uint16_t value;
    uint8_t placement_group;
} TaskCommand_t;

static const TaskRegistration_t task_table[] = {
    {1u}, {1u}, {1u}, {1u}, {1u}, {1u}, {1u}, {1u}, {1u}
};

#define TASK_COUNT ((uint8_t)(sizeof(task_table) / sizeof(task_table[0])))
#define TASK2_ATTRACT_DUTY 10000u
#define COMMAND_QUEUE_SIZE 32u
#define CAMERA_LINE_SIZE 64u
#define MAGNET_SETTLE_MS 500u
#define MAGNET_DEMAG_MS 50u
#define CALIBRATION_RATE_PPS 100u
#define DONE_BEEP_MS 500u
#define TRANSFER_XY_SPEED_RPM 2000u
#define TRANSFER_Z_SPEED_RPM 1000u
#define TRANSFER_R_SPEED_RPM 1500u
#define TASK1_XY_SPEED_RPM 3000u
#define TASK1_Z_SPEED_RPM 1500u
#define TRANSFER_Z_TRAVEL_PULSES 300u
#define TASK_CAMERA_X_MAX_PULSES 24000u
#define TASK_CAMERA_Y_MAX_PULSES 16700u
#define TRANSFER_MAX_X_PULSES 29500u
#define TRANSFER_MAX_Y_PULSES 16800u
#define TRANSFER_MAX_ROT_DEG 360u

typedef enum
{
    TASK9_HOMING = 0,
    TASK9_READY,
    TASK9_RUNNING,
    TASK9_REHOMING
} Task9Phase_t;

static const TaskMotionOps_t *motion;
static uint8_t current_task;
static uint8_t progress;
static TaskState_t state;
static TaskCommand_t command_queue[COMMAND_QUEUE_SIZE];
static uint8_t queue_head;
static uint8_t queue_tail;
static uint8_t command_active;
static TaskCommand_t active_command;
static uint8_t parallel_command_count;
static TaskCommand_t parallel_commands[2];
static uint16_t total_commands;
static uint16_t completed_commands;
static uint32_t command_started;
static uint32_t pause_started;
static uint32_t demag_started;
static uint8_t magnet_attracting;
static uint8_t demag_pending;
static uint32_t calibration_last_log;
static char operation_text[24];
static volatile char camera_line[CAMERA_LINE_SIZE];
static volatile uint8_t camera_length;
static volatile uint8_t camera_line_ready;
static uint8_t task9_phase;
static uint8_t task9_piece_id;
static int32_t task9_x_pulses;
static int32_t task9_y_pulses;
static uint16_t task9_r_degrees;
static VisionPlan_t task9_vision_plan;
static uint8_t task9_vision_index;
static uint8_t task9_vision_active;
static uint8_t done_announced;
static uint8_t beep_active;
static uint32_t beep_started;
static uint8_t return_home_enabled;
static uint8_t returning_home;
static uint8_t exit_returning_home;

static uint32_t task_tick(void);

static uint16_t transfer_speed_for_axis(uint8_t axis)
{
    if (current_task == 1u)
    {
        if (axis == TASK_AXIS_Z) return TASK1_Z_SPEED_RPM;
        return TASK1_XY_SPEED_RPM;
    }
    if (axis == TASK_AXIS_R) return TRANSFER_R_SPEED_RPM;
    if (axis == TASK_AXIS_Z) return TRANSFER_Z_SPEED_RPM;
    return TRANSFER_XY_SPEED_RPM;
}

uint32_t Task_CameraPulseToMechanical(uint8_t axis, uint32_t camera_pulse)
{
    uint32_t camera_max;
    uint32_t mechanical_max;
    if (axis == TASK_AXIS_X)
    {
        camera_max = TASK_CAMERA_X_MAX_PULSES;
        mechanical_max = TRANSFER_MAX_X_PULSES;
    }
    else if (axis == TASK_AXIS_Y)
    {
        camera_max = TASK_CAMERA_Y_MAX_PULSES;
        mechanical_max = TRANSFER_MAX_Y_PULSES;
    }
    else
    {
        return 0u;
    }
    if (camera_pulse >= camera_max) return mechanical_max;
    return (uint32_t)((((uint64_t)camera_pulse * mechanical_max) +
                       (camera_max / 2u)) / camera_max);
}

static void indication_trigger(void)
{
    done_announced = 1u;
    if ((motion != NULL) && (motion->set_light != NULL)) motion->set_light(1u);
    if ((motion != NULL) && (motion->set_beeper != NULL))
    {
        motion->set_beeper(1u);
        beep_active = 1u;
        beep_started = task_tick();
    }
}

static void indication_silence(void)
{
    beep_active = 0u;
    done_announced = 0u;
    if ((motion != NULL) && (motion->set_beeper != NULL)) motion->set_beeper(0u);
    if ((motion != NULL) && (motion->set_light != NULL)) motion->set_light(0u);
}

static void indication_process(void)
{
    if ((state == TASK_STATE_DONE) && !done_announced)
    {
        indication_trigger();
    }
    if (beep_active && ((uint32_t)(task_tick() - beep_started) >= DONE_BEEP_MS))
    {
        beep_active = 0u;
        motion->set_beeper(0u);
    }
}

static uint32_t task_tick(void)
{
    return ((motion != NULL) && (motion->get_tick_ms != NULL)) ?
           motion->get_tick_ms() : 0u;
}

static void magnet_attract(void)
{
    demag_pending = 0u;
    magnet_attracting = 1u;
    motion->set_electromagnets(TASK_MAGNET_ATTRACT, TASK2_ATTRACT_DUTY);
}

static void magnet_release(void)
{
    if (demag_pending) return;
    if (!magnet_attracting)
    {
        motion->set_electromagnets(TASK_MAGNET_OFF, 0u);
        snprintf(operation_text, sizeof(operation_text), "MAGNET OFF");
        return;
    }
    magnet_attracting = 0u;
    demag_pending = 1u;
    demag_started = task_tick();
    motion->set_electromagnets(TASK_MAGNET_REPEL, TASK2_ATTRACT_DUTY);
    snprintf(operation_text, sizeof(operation_text), "MAGNET DEMAG");
}

static void magnet_process_release(void)
{
    if (!demag_pending || ((uint32_t)(task_tick() - demag_started) < MAGNET_DEMAG_MS)) return;
    demag_pending = 0u;
    motion->set_electromagnets(TASK_MAGNET_OFF, 0u);
    snprintf(operation_text, sizeof(operation_text), "MAGNET OFF");
}

static void queue_clear(void)
{
    queue_head = queue_tail = 0u;
    command_active = 0u;
    parallel_command_count = 0u;
    total_commands = completed_commands = 0u;
}

static uint8_t queue_push(TaskCommand_t command)
{
    uint8_t next = (uint8_t)((queue_head + 1u) % COMMAND_QUEUE_SIZE);
    if (next == queue_tail) return 0u;
    command_queue[queue_head] = command;
    queue_head = next;
    ++total_commands;
    return 1u;
}

static uint8_t queue_pop(TaskCommand_t *command)
{
    if (queue_tail == queue_head) return 0u;
    *command = command_queue[queue_tail];
    queue_tail = (uint8_t)((queue_tail + 1u) % COMMAND_QUEUE_SIZE);
    return 1u;
}

static uint8_t queue_peek(TaskCommand_t *command)
{
    if (queue_tail == queue_head) return 0u;
    *command = command_queue[queue_tail];
    return 1u;
}

static uint8_t commands_form_xy_pair(const TaskCommand_t *first,
                                     const TaskCommand_t *second)
{
    if ((first->type != CMD_AXIS_PULSES) ||
        (second->type != CMD_AXIS_PULSES)) return 0u;
    return ((first->axis == TASK_AXIS_X) && (second->axis == TASK_AXIS_Y)) ||
           ((first->axis == TASK_AXIS_Y) && (second->axis == TASK_AXIS_X));
}

static void task3_process(void);

static void complete_or_return_home(void)
{
    progress = 100u;
    if (return_home_enabled && (motion != NULL) &&
        (motion->homing_start != NULL) && (motion->homing_process != NULL))
    {
        returning_home = 1u;
        state = TASK_STATE_RUNNING;
        motion->homing_start();
        snprintf(operation_text, sizeof(operation_text), "RETURN HOME");
        return;
    }
    state = TASK_STATE_DONE;
    snprintf(operation_text, sizeof(operation_text), "TASK %u DONE", current_task);
}

/* Protocol: PIECE,id,pickX,pickY,targetX,targetY,rotDeg.
   X/Y coordinates are absolute pulse coordinates from the power-on origin. */
static uint8_t parse_piece_line(char *line, uint8_t *piece_id,
                                uint16_t *pick_x, uint16_t *pick_y,
                                uint16_t *place_x, uint16_t *place_y,
                                uint16_t *rot_deg)
{
    char *fields[6];
    char *token;
    char *end;
    uint8_t count = 0u;
    uint8_t i;
    unsigned long values[6];

    token = strtok(line, ",");
    if ((token == NULL) || (strcmp(token, "PIECE") != 0)) return 0u;
    while ((count < 6u) && ((token = strtok(NULL, ",")) != NULL))
    {
        fields[count++] = token;
    }
    if ((count != 6u) || (strtok(NULL, ",") != NULL)) return 0u;
    for (i = 0u; i < 6u; ++i)
    {
        values[i] = strtoul(fields[i], &end, 10);
        if ((*end != '\0') || (values[i] > 65535ul)) return 0u;
    }
    if ((values[0] > 255ul) ||
        (values[1] > TRANSFER_MAX_X_PULSES) ||
        (values[2] > TRANSFER_MAX_Y_PULSES) ||
        (values[3] > TRANSFER_MAX_X_PULSES) ||
        (values[4] > TRANSFER_MAX_Y_PULSES) ||
        (values[5] > TRANSFER_MAX_ROT_DEG)) return 0u;
    *piece_id = (uint8_t)values[0];
    *pick_x = (uint16_t)values[1];
    *pick_y = (uint16_t)values[2];
    *place_x = (uint16_t)values[3];
    *place_y = (uint16_t)values[4];
    *rot_deg = (uint16_t)values[5];
    return 1u;
}

static void queue_axis_delta(uint8_t axis, int32_t *current_pulses,
                             uint32_t target_pulses,
                             uint8_t placement_group)
{
    int32_t delta = (int32_t)target_pulses - *current_pulses;
    TaskCommand_t command;
    if (delta == 0) return;
    command.type = CMD_AXIS_PULSES;
    command.axis = axis;
    command.direction = (delta > 0) ? 1u : 0u;
    command.value = (uint16_t)((delta > 0) ? delta : -delta);
    command.placement_group = placement_group;
    (void)queue_push(command);
    *current_pulses = (int32_t)target_pulses;
}

static void queue_rotation(uint16_t rotation, uint8_t direction,
                           uint8_t placement_group)
{
    TaskCommand_t command = {CMD_AXIS, TASK_AXIS_R, direction, rotation,
                             placement_group};
    if (rotation != 0u) (void)queue_push(command);
}

static void queue_interpiece_pickup_group(uint16_t pick_x, uint16_t pick_y)
{
    if (task9_r_degrees != 0u)
    {
        queue_axis_delta(TASK_AXIS_X, &task9_x_pulses, pick_x, 1u);
        queue_axis_delta(TASK_AXIS_Y, &task9_y_pulses, pick_y, 1u);
        queue_rotation(task9_r_degrees, 0u, 1u);
        task9_r_degrees = 0u;
    }
    else
    {
        queue_axis_delta(TASK_AXIS_X, &task9_x_pulses, pick_x, 0u);
        queue_axis_delta(TASK_AXIS_Y, &task9_y_pulses, pick_y, 0u);
    }
}

static void load_transfer_script(uint16_t pick_x, uint16_t pick_y,
                                 uint16_t place_x, uint16_t place_y,
                                 uint16_t rot_deg, uint8_t interpiece)
{
    static const TaskCommand_t z_down = {CMD_AXIS_PULSES, TASK_AXIS_Z, 1u, TRANSFER_Z_TRAVEL_PULSES, 0u};
    static const TaskCommand_t z_up = {CMD_AXIS_PULSES, TASK_AXIS_Z, 0u, TRANSFER_Z_TRAVEL_PULSES, 0u};
    static const TaskCommand_t magnet_on = {CMD_MAGNET, 0u, 1u, 0u, 0u};
    static const TaskCommand_t magnet_off = {CMD_MAGNET, 0u, 0u, 0u, 0u};
    queue_clear();
    if (interpiece) queue_interpiece_pickup_group(pick_x, pick_y);
    else
    {
        queue_axis_delta(TASK_AXIS_X, &task9_x_pulses, pick_x, 0u);
        queue_axis_delta(TASK_AXIS_Y, &task9_y_pulses, pick_y, 0u);
    }
    (void)queue_push(z_down);
    (void)queue_push(magnet_on);
    (void)queue_push(z_up);
    queue_axis_delta(TASK_AXIS_X, &task9_x_pulses, place_x, 1u);
    queue_axis_delta(TASK_AXIS_Y, &task9_y_pulses, place_y, 1u);
    queue_rotation(rot_deg, 1u, 1u);
    task9_r_degrees = rot_deg;
    (void)queue_push(z_down);
    (void)queue_push(magnet_off);
    (void)queue_push(z_up);
}

static void load_vision_transfer_script(const VisionCommand_t *command,
                                        uint8_t interpiece)
{
    load_transfer_script((uint16_t)command->pickup_x_pulse,
                         (uint16_t)command->pickup_y_pulse,
                         (uint16_t)command->place_x_pulse,
                         (uint16_t)command->place_y_pulse,
                         command->rotation_deg, interpiece);
}

static void start_vision_piece(uint8_t index)
{
    const VisionCommand_t *command = &task9_vision_plan.commands[index];
    task9_piece_id = command->piece_id;
    load_vision_transfer_script(
        command, (uint8_t)((current_task == 1u) && (index != 0u)));
    task9_phase = TASK9_RUNNING;
    progress = 0u;
    state = TASK_STATE_RUNNING;
    snprintf(operation_text, sizeof(operation_text),
             "JSON PIECE %u RUN", command->piece_id);
}

static void task9_process(void)
{
    uint8_t homing;

    if (task9_phase == TASK9_RUNNING)
    {
        task3_process();
        if (state == TASK_STATE_DONE)
        {
            if ((current_task == 1u) && task9_vision_active &&
                ((uint8_t)(task9_vision_index + 1u) < task9_vision_plan.count))
            {
                ++task9_vision_index;
                progress = 0u;
                state = TASK_STATE_RUNNING;
                start_vision_piece(task9_vision_index);
                return;
            }
            state = TASK_STATE_RUNNING;
            progress = 0u;
            task9_phase = TASK9_REHOMING;
            motion->homing_start();
            snprintf(operation_text, sizeof(operation_text),
                     "TASK%u RETURN HOME", current_task);
        }
        return;
    }
    if (task9_phase != TASK9_REHOMING) return;
    homing = motion->homing_process();
    if (homing == 1u) return;
    if (homing != 2u)
    {
        Task_SetError();
        snprintf(operation_text, sizeof(operation_text),
                 "TASK%u HOME ERR", current_task);
        return;
    }
    task9_phase = TASK9_READY;
    task9_x_pulses = 0;
    task9_y_pulses = 0;
    task9_r_degrees = 0u;
    if (task9_vision_active)
    {
        ++task9_vision_index;
        if (task9_vision_index < task9_vision_plan.count)
        {
            start_vision_piece(task9_vision_index);
            return;
        }
        task9_vision_active = 0u;
        progress = 0u;
        state = TASK_STATE_RUNNING;
        indication_trigger();
        snprintf(operation_text, sizeof(operation_text),
                 (current_task == 1u) ? "TASK1 WAIT JSON" : "TASK2 WAIT PIECE");
        return;
    }
    if (current_task == 2u)
    {
        char response[24];
        snprintf(response, sizeof(response), "PIECE,%u,DONE", task9_piece_id);
        if (motion->send_response != NULL) motion->send_response(response);
        progress = 0u;
        state = TASK_STATE_RUNNING;
        indication_trigger();
        snprintf(operation_text, sizeof(operation_text), "TASK2 WAIT PIECE");
        return;
    }
    progress = 100u;
    state = TASK_STATE_DONE;
    snprintf(operation_text, sizeof(operation_text), "TASK %u DONE", current_task);
}

static void format_operation(const TaskCommand_t *command)
{
    static const char axis_names[] = {'X', 'Y', 'Z', 'R'};
    if (command->type == CMD_MAGNET)
    {
        snprintf(operation_text, sizeof(operation_text), "MAGNET %s",
                 command->direction ? "ATTRACT" : "RELEASE");
    }
    else if (command->type == CMD_AXIS_PULSES)
    {
        snprintf(operation_text, sizeof(operation_text), "%c %s %u PLS",
                 axis_names[command->axis], command->direction ? "CW" : "CCW",
                 command->value);
    }
    else if (command->axis == TASK_AXIS_R)
    {
        snprintf(operation_text, sizeof(operation_text), "R %s %u DEG",
                 command->direction ? "CW" : "CCW", command->value);
    }
    else
    {
        snprintf(operation_text, sizeof(operation_text), "%c %s %u DEG",
                 axis_names[command->axis], command->direction ? "CW" : "CCW",
                 command->value);
    }
}

static void dispatch_motion_command(const TaskCommand_t *command)
{
    if (command->type == CMD_AXIS_PULSES)
    {
        if ((current_task == 1u) || (current_task == 2u))
            motion->start_axis_move_pulses_at_speed(
                command->axis, command->direction, command->value,
                transfer_speed_for_axis(command->axis));
        else
            motion->start_axis_move_pulses(
                command->axis, command->direction, command->value);
    }
    else
    {
        motion->start_axis_move(command->axis, command->direction,
                                command->value);
    }
}

static void start_command(TaskCommand_t command)
{
    active_command = command;
    command_active = 1u;
    command_started = task_tick();
    format_operation(&command);
    printf("[CMD] START %s\r\n", operation_text);
    if (command.type == CMD_MAGNET)
    {
        if (command.direction) magnet_attract();
        else magnet_release();
    }
    else dispatch_motion_command(&command);
}

static uint8_t command_complete(void)
{
    if (active_command.type == CMD_MAGNET)
        return (uint32_t)(task_tick() - command_started) >= MAGNET_SETTLE_MS;
    uint8_t i;
    if (!motion->axis_move_complete(active_command.axis)) return 0u;
    for (i = 0u; i < parallel_command_count; ++i)
        if (!motion->axis_move_complete(parallel_commands[i].axis)) return 0u;
    return 1u;
}

static uint8_t command_failed(void)
{
    uint8_t i;
    if (motion->axis_move_failed == NULL) return 0u;
    if (motion->axis_move_failed(active_command.axis)) return 1u;
    for (i = 0u; i < parallel_command_count; ++i)
        if (motion->axis_move_failed(parallel_commands[i].axis)) return 1u;
    return 0u;
}

static void task3_process(void)
{
    uint8_t group_member;
    if (!command_active)
    {
        if (!queue_pop(&active_command))
        {
            complete_or_return_home();
            return;
        }
        start_command(active_command);
        if (((current_task == 1u) || (current_task == 2u)) &&
            active_command.placement_group)
        {
            while ((parallel_command_count < 2u) &&
                   queue_peek(&parallel_commands[parallel_command_count]) &&
                   parallel_commands[parallel_command_count].placement_group)
            {
                (void)queue_pop(&parallel_commands[parallel_command_count]);
                dispatch_motion_command(&parallel_commands[parallel_command_count]);
                ++parallel_command_count;
            }
            group_member = (uint8_t)(parallel_command_count + 1u);
            snprintf(operation_text, sizeof(operation_text),
                     "XYZR GROUP %u", group_member);
        }
        else if (((current_task == 1u) || (current_task == 2u)) &&
            queue_peek(&parallel_commands[0]) &&
            commands_form_xy_pair(&active_command, &parallel_commands[0]))
        {
            uint16_t speed_rpm = transfer_speed_for_axis(TASK_AXIS_X);
            (void)queue_pop(&parallel_commands[0]);
            parallel_command_count = 1u;
            motion->start_axis_move_pulses_at_speed(
                parallel_commands[0].axis, parallel_commands[0].direction,
                parallel_commands[0].value, speed_rpm);
            snprintf(operation_text, sizeof(operation_text), "X+Y %u RPM", speed_rpm);
        }
        return;
    }
    if ((active_command.type != CMD_MAGNET) && command_failed())
    {
        printf("[CMD] ERROR %s\r\n", operation_text);
        Task_SetError();
        snprintf(operation_text, sizeof(operation_text), "AXIS VERIFY ERROR");
        return;
    }
    if (command_complete())
    {
        printf("[CMD] DONE %s\r\n", operation_text);
        command_active = 0u;
        completed_commands = (uint16_t)(completed_commands +
                             parallel_command_count + 1u);
        parallel_command_count = 0u;
        progress = (total_commands == 0u) ? 0u :
                   (uint8_t)(((uint32_t)completed_commands * 100u) / total_commands);
    }
}

static uint8_t clamp_progress(uint8_t value) { return (value > 100u) ? 100u : value; }

static void calibration_process(void)
{
    static const char axis_names[] = {'X', 'Y', 'Z', 'R'};
    uint8_t axis = (uint8_t)(current_task - 3u);
    uint32_t pulses = motion->get_calibration_pulses();
    snprintf(operation_text, sizeof(operation_text), "%c PULSE: %lu",
             axis_names[axis], (unsigned long)pulses);
    if ((uint32_t)(task_tick() - calibration_last_log) >= 1000u)
    {
        calibration_last_log = task_tick();
        printf("[CAL] %c pulses=%lu\r\n", axis_names[axis], (unsigned long)pulses);
    }
}

void Task_Init(const TaskMotionOps_t *ops)
{
    motion = ops;
    if (motion && motion->set_task1_open_loop)
        motion->set_task1_open_loop(0u);
    current_task = 1u;
    progress = 0u;
    state = TASK_STATE_IDLE;
    camera_length = 0u;
    camera_line_ready = 0u;
    magnet_attracting = 0u;
    demag_pending = 0u;
    task9_phase = TASK9_HOMING;
    task9_piece_id = 0u;
    task9_x_pulses = 0;
    task9_y_pulses = 0;
    task9_vision_index = 0u;
    task9_vision_active = 0u;
    return_home_enabled = 0u;
    returning_home = 0u;
    exit_returning_home = 0u;
    indication_silence();
    queue_clear();
    operation_text[0] = '\0';
}

void Task_Process(void)
{
    uint16_t total;
    uint8_t axis;
    if ((motion != NULL) && (motion->set_electromagnets != NULL))
        magnet_process_release();
    indication_process();
    if (camera_line_ready)
    {
        char line[CAMERA_LINE_SIZE];
        char raw_line[CAMERA_LINE_SIZE];
        uint8_t i;
        uint8_t length = camera_length;
        for (i = 0u; i <= length; ++i)
        {
            line[i] = camera_line[i];
            raw_line[i] = camera_line[i];
        }
        camera_line_ready = 0u;
        camera_length = 0u;
        if ((current_task == 2u) && (state == TASK_STATE_RUNNING))
        {
            uint8_t piece_id;
            uint16_t pick_x, pick_y, place_x, place_y, rot_deg;
            if ((task9_phase == TASK9_READY) && (strcmp(line, "ZERO") == 0))
            {
                /* Store the current position of every axis as its homing zero. */
                uint8_t ok = (motion->set_homing_zero != NULL) ?
                             motion->set_homing_zero() : 0u;
                if (motion->send_response != NULL)
                    motion->send_response(ok ? "ZERO,OK" : "ZERO,ERR");
                printf("[CAM] %s %s\r\n", ok ? "ACCEPT" : "REJECT", raw_line);
            }
            else if ((task9_phase == TASK9_READY) &&
                parse_piece_line(line, &piece_id, &pick_x, &pick_y,
                                 &place_x, &place_y, &rot_deg))
            {
                task9_piece_id = piece_id;
                task9_vision_active = 0u;
                load_transfer_script(pick_x, pick_y, place_x, place_y, rot_deg,
                                     0u);
                task9_phase = TASK9_RUNNING;
                snprintf(operation_text, sizeof(operation_text), "PIECE %u RUN", piece_id);
                printf("[CAM] ACCEPT %s\r\n", raw_line);
            }
            else
            {
                printf("[CAM] REJECT %s\r\n", raw_line);
                if ((task9_phase == TASK9_READY) && (motion->send_response != NULL))
                    motion->send_response("PIECE,ERR");
            }
        }
        else
            printf("[CAM] REJECT %s\r\n", raw_line);
    }
    if ((state != TASK_STATE_RUNNING) || (motion == NULL)) return;
    if (returning_home)
    {
        uint8_t homing = motion->homing_process();
        if (homing == 1u) return;
        returning_home = 0u;
        if (homing != 2u)
        {
            Task_SetError();
            snprintf(operation_text, sizeof(operation_text), "RETURN HOME ERR");
            return;
        }
        if (exit_returning_home)
        {
            exit_returning_home = 0u;
            progress = 0u;
            state = TASK_STATE_IDLE;
            operation_text[0] = '\0';
            return;
        }
        state = TASK_STATE_DONE;
        progress = 100u;
        snprintf(operation_text, sizeof(operation_text), "TASK %u DONE", current_task);
        return;
    }
    if ((current_task == 1u) || (current_task == 2u))
    { task9_process(); return; }
    if (current_task == 9u)
    {
        uint8_t result = (motion->recorded_home_process != NULL) ?
                         motion->recorded_home_process() : 3u;
        if (result == 2u)
        {
            progress = 100u;
            state = TASK_STATE_DONE;
            snprintf(operation_text, sizeof(operation_text), "TASK 9 DONE");
        }
        else if (result == 3u)
        {
            Task_SetError();
            snprintf(operation_text, sizeof(operation_text), "TASK9 HOME ERR");
        }
        return;
    }
    if ((current_task >= 3u) && (current_task <= 6u))
    { calibration_process(); return; }
    if (current_task == 8u)
    {
        if (!demag_pending)
        {
            progress = 100u;
            state = TASK_STATE_DONE;
            snprintf(operation_text, sizeof(operation_text), "TASK 8 DONE");
        }
        return;
    }
    if (current_task != 1u) return;
    total = 0u;
    if (motion->get_axis_progress != NULL)
    {
        for (axis = 0u; axis < 4u; ++axis)
            total = (uint16_t)(total + clamp_progress(motion->get_axis_progress(axis)));
        progress = (uint8_t)(total / 4u);
    }
    if ((motion->all_complete != NULL) && motion->all_complete())
    { complete_or_return_home(); }
}

void Task_SelectNext(void)
{
    if ((state != TASK_STATE_IDLE) && (state != TASK_STATE_DONE)) return;
    if (motion && motion->set_task1_open_loop)
        motion->set_task1_open_loop(0u);
    indication_silence();
    current_task = (uint8_t)((current_task % TASK_COUNT) + 1u);
    progress = 0u; state = TASK_STATE_IDLE; operation_text[0] = '\0';
    Task_AutoStartTask1();
}

static void task_start_selected(void)
{
    if ((state != TASK_STATE_IDLE) && (state != TASK_STATE_DONE)) return;
    {
        progress = 0u;
        indication_silence();
        if (!Task_IsConfigured()) { progress = 100u; state = TASK_STATE_DONE; return; }
        if (motion == NULL) { state = TASK_STATE_ERROR; return; }
        if (motion->set_task1_open_loop != NULL)
            motion->set_task1_open_loop((current_task == 1u) ? 1u : 0u);
        if ((current_task != 9u) && (motion->capture_home_reference != NULL))
            motion->capture_home_reference();
        if (current_task == 1u)
        {
            if ((motion->start_axis_move == NULL) ||
                (motion->start_axis_move_pulses == NULL) ||
                (motion->start_axis_move_pulses_at_speed == NULL) ||
                (motion->axis_move_complete == NULL) ||
                (motion->axis_move_failed == NULL) ||
                (motion->set_electromagnets == NULL) ||
                (motion->homing_start == NULL) ||
                (motion->homing_process == NULL))
            { state = TASK_STATE_ERROR; return; }
            queue_clear();
            task9_x_pulses = 0;
            task9_y_pulses = 0;
            task9_r_degrees = 0u;
            task9_vision_index = 0u;
            task9_vision_active = 0u;
            task9_phase = TASK9_READY;
            snprintf(operation_text, sizeof(operation_text), "TASK1 WAIT JSON");
        }
        else if (current_task == 2u)
        {
            if ((motion->start_axis_move == NULL) ||
                (motion->start_axis_move_pulses == NULL) ||
                (motion->start_axis_move_pulses_at_speed == NULL) ||
                (motion->axis_move_complete == NULL) ||
                (motion->axis_move_failed == NULL) ||
                (motion->set_electromagnets == NULL) ||
                (motion->homing_start == NULL) ||
                (motion->homing_process == NULL))
            { state = TASK_STATE_ERROR; return; }
            queue_clear();
            task9_x_pulses = 0;
            task9_y_pulses = 0;
            task9_r_degrees = 0u;
            task9_vision_index = 0u;
            task9_vision_active = 0u;
            task9_phase = TASK9_READY;
            snprintf(operation_text, sizeof(operation_text), "TASK2 WAIT PIECE");
        }
        else if ((current_task >= 3u) && (current_task <= 6u))
        {
            uint8_t axis = (uint8_t)(current_task - 3u);
            if ((motion->start_calibration == NULL) ||
                (motion->get_calibration_pulses == NULL))
            { state = TASK_STATE_ERROR; return; }
            motion->start_calibration(axis, CALIBRATION_RATE_PPS);
            calibration_last_log = task_tick();
            snprintf(operation_text, sizeof(operation_text), "%c PULSE: 0",
                     (axis == 0u) ? 'X' : ((axis == 1u) ? 'Y' :
                     ((axis == 2u) ? 'Z' : 'R')));
        }
        else if (current_task == 7u)
        {
            if (motion->set_electromagnets == NULL)
            { state = TASK_STATE_ERROR; return; }
            magnet_attract();
        }
        else if (current_task == 8u)
        {
            if (motion->set_electromagnets == NULL)
            { state = TASK_STATE_ERROR; return; }
            /* Exercise the complete release path even when Task 8 starts idle. */
            magnet_attracting = 1u;
            magnet_release();
        }
        else if (current_task == 9u)
        {
            if ((motion->recorded_home_start == NULL) ||
                (motion->recorded_home_process == NULL))
            { state = TASK_STATE_ERROR; return; }
            motion->recorded_home_start();
            snprintf(operation_text, sizeof(operation_text), "TASK9 REC HOME");
        }
        state = TASK_STATE_RUNNING;
        return;
    }
}

void Task_AutoStartTask1(void)
{
    if ((current_task == 1u) &&
        ((state == TASK_STATE_IDLE) || (state == TASK_STATE_DONE)))
        task_start_selected();
}

void Task_StartOrTogglePause(void)
{
    if (current_task == 1u) return;
    if ((state == TASK_STATE_IDLE) || (state == TASK_STATE_DONE))
    {
        task_start_selected();
        return;
    }
    if (state == TASK_STATE_RUNNING)
    {
        if (current_task == 9u) return;
        /* Transfer-task ready/rehoming phases cannot be paused safely. */
        if (((current_task == 1u) || (current_task == 2u)) &&
            (task9_phase != TASK9_RUNNING)) return;
        if (current_task == 7u) magnet_release();
        else if ((current_task >= 3u) && (current_task <= 6u) &&
                  (motion->pause_calibration != NULL)) motion->pause_calibration();
        else if (motion->pause_all != NULL) motion->pause_all();
        pause_started = task_tick(); state = TASK_STATE_PAUSED;
    }
    else if (state == TASK_STATE_PAUSED)
    {
        if (current_task == 7u) magnet_attract();
        else if ((current_task >= 3u) && (current_task <= 6u) &&
                  (motion->resume_calibration != NULL)) motion->resume_calibration();
        else if (motion->resume_all != NULL) motion->resume_all();
        if (((current_task == 1u) || (current_task == 2u)) &&
            command_active)
            command_started += (uint32_t)(task_tick() - pause_started);
        state = TASK_STATE_RUNNING;
    }
}

void Task_Exit(void)
{
    if (motion && motion->set_task1_open_loop)
        motion->set_task1_open_loop(0u);
    if (((current_task == 1u) || (current_task == 2u) ||
         (current_task == 7u) || (current_task == 8u)) &&
        motion && motion->set_electromagnets)
        magnet_release();
    if (((current_task == 1u) || (current_task == 2u)) &&
        motion && motion->abort_all)
        motion->abort_all();
    if ((current_task >= 3u) && (current_task <= 6u) && motion && motion->stop_calibration)
        motion->stop_calibration();
    task9_phase = TASK9_HOMING;
    task9_vision_active = 0u;
    indication_silence();
    queue_clear();
    progress = 0u;
    if (motion && motion->homing_start && motion->homing_process)
    {
        returning_home = 1u;
        exit_returning_home = 1u;
        state = TASK_STATE_RUNNING;
        motion->homing_start();
        snprintf(operation_text, sizeof(operation_text), "EXIT RETURN HOME");
    }
    else
    {
        returning_home = 0u;
        exit_returning_home = 0u;
        state = TASK_STATE_IDLE;
        operation_text[0] = '\0';
    }
}

void Task_SetError(void)
{
    if (motion && motion->set_electromagnets && !magnet_attracting)
        magnet_release();
    if (motion && motion->abort_all) motion->abort_all();
    if ((current_task >= 3u) && (current_task <= 6u) && motion && motion->stop_calibration)
        motion->stop_calibration();
    returning_home = 0u;
    exit_returning_home = 0u;
    indication_silence();
    queue_clear(); state = TASK_STATE_ERROR;
}

void Task_CameraRxByte(uint8_t byte)
{
    if (byte == '\r') return;
    if (byte != '\n')
    {
        if (!camera_line_ready && (camera_length < (CAMERA_LINE_SIZE - 1u)))
            camera_line[camera_length++] = (char)byte;
        else camera_length = 0u;
        return;
    }
    if (!camera_line_ready)
    {
        camera_line[camera_length] = '\0';
        camera_line_ready = 1u;
    }
}

uint8_t Task_SubmitVisionPlan(const VisionPlan_t *plan)
{
    if ((plan == NULL) || (plan->count == 0u) ||
        (plan->count > VISION_PLAN_MAX_COMMANDS) ||
        (current_task != 1u) ||
        (state != TASK_STATE_RUNNING) ||
        (task9_phase != TASK9_READY) || task9_vision_active)
        return 0u;
    task9_vision_plan = *plan;
    task9_vision_index = 0u;
    task9_vision_active = 1u;
    task9_x_pulses = 0;
    task9_y_pulses = 0;
    task9_r_degrees = 0u;
    start_vision_piece(0u);
    return 1u;
}

uint8_t Task_GetCurrentId(void) { return current_task; }
uint8_t Task_GetProgress(void) { return progress; }
TaskState_t Task_GetState(void) { return state; }
const char *Task_GetOperationText(void) { return operation_text; }
const char *Task_GetStateText(void)
{
    static const char *const names[] = {"IDLE", "RUNNING", "PAUSED", "DONE", "ERROR"};
    return names[(uint8_t)state];
}
uint8_t Task_IsConfigured(void) { return task_table[current_task - 1u].configured; }
