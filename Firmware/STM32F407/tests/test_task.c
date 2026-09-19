#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "task.h"
#include "vision_jsonl.h"

static unsigned start_count;
static unsigned pause_count;
static unsigned resume_count;
static unsigned abort_count;
static uint8_t axis_progress[4];
static uint8_t all_complete;
static unsigned magnet_count;
static uint8_t magnet_mode;
static uint16_t magnet_duty;
static uint8_t move_axis;
static uint8_t move_direction;
static uint16_t move_degrees;
static uint32_t move_pulses;
static uint8_t move_done;
static uint8_t speed_move_tracking;
static uint8_t speed_move_done[4];
static uint8_t speed_move_failed[4];
static unsigned speed_move_count[4];
static uint32_t speed_move_pulses[4];
static uint16_t speed_move_rpm[4];
static uint32_t fake_tick;
static unsigned calibration_start_count;
static unsigned calibration_pause_count;
static unsigned calibration_resume_count;
static unsigned calibration_stop_count;
static uint8_t calibration_axis;
static uint16_t calibration_rate;
static uint32_t calibration_pulses;

static void fake_start(void) { ++start_count; }
static void fake_pause(void) { ++pause_count; }
static void fake_resume(void) { ++resume_count; }
static void fake_abort(void) { ++abort_count; }
static uint8_t fake_progress(uint8_t axis) { return axis_progress[axis]; }
static uint8_t fake_complete(void) { return all_complete; }
static void fake_magnets(uint8_t mode, uint16_t duty)
{
    ++magnet_count;
    magnet_mode = mode;
    magnet_duty = duty;
}
static void fake_move(uint8_t axis, uint8_t direction, uint16_t degrees)
{ move_axis = axis; move_direction = direction; move_degrees = degrees; move_done = 0u; }
static void fake_move_pulses(uint8_t axis, uint8_t direction, uint32_t pulses)
{ move_axis = axis; move_direction = direction; move_pulses = pulses; move_done = 0u; }
static void fake_move_pulses_at_speed(uint8_t axis, uint8_t direction,
                                      uint32_t pulses, uint16_t speed_rpm)
{
    speed_move_tracking = 1u;
    ++speed_move_count[axis];
    speed_move_pulses[axis] = pulses;
    speed_move_rpm[axis] = speed_rpm;
    speed_move_done[axis] = 0u;
    move_axis = axis;
    move_direction = direction;
    move_pulses = pulses;
}
static uint8_t fake_move_complete(uint8_t axis)
{ return (speed_move_tracking && (axis != TASK_AXIS_R)) ? speed_move_done[axis] : move_done; }
static uint8_t fake_move_failed(uint8_t axis) { return speed_move_failed[axis]; }
static uint32_t fake_get_tick(void) { return fake_tick; }
static void fake_calibration_start(uint8_t axis, uint16_t rate)
{ ++calibration_start_count; calibration_axis = axis; calibration_rate = rate; calibration_pulses = 0u; }
static void fake_calibration_pause(void) { ++calibration_pause_count; }
static void fake_calibration_resume(void) { ++calibration_resume_count; }
static void fake_calibration_stop(void) { ++calibration_stop_count; }
static uint32_t fake_calibration_pulses(void) { return calibration_pulses; }
static unsigned homing_start_count;
static uint8_t homing_result;
static char last_response[32];
static void fake_homing_start(void) { ++homing_start_count; }
static uint8_t fake_homing_process(void) { return homing_result; }
static void fake_send_response(const char *text)
{ strncpy(last_response, text, sizeof(last_response) - 1u); last_response[sizeof(last_response) - 1u] = '\0'; }
static uint8_t beeper_on;
static uint8_t light_on;
static void fake_set_beeper(uint8_t on) { beeper_on = on; }
static void fake_set_light(uint8_t on) { light_on = on; }
static unsigned homing_zero_count;
static uint8_t fake_set_homing_zero(void) { ++homing_zero_count; return 1u; }
static unsigned home_reference_capture_count;
static void fake_capture_home_reference(void) { ++home_reference_capture_count; }
static unsigned recorded_home_start_count;
static uint8_t recorded_home_result;
static uint8_t task1_open_loop_enabled;
static void fake_recorded_home_start(void) { ++recorded_home_start_count; }
static uint8_t fake_recorded_home_process(void) { return recorded_home_result; }
static void fake_set_task1_open_loop(uint8_t enabled)
{ task1_open_loop_enabled = enabled; }

int main(void)
{
    const TaskMotionOps_t ops = {
        fake_start, fake_pause, fake_resume, fake_abort,
        fake_progress, fake_complete, fake_magnets,
        fake_move, fake_move_complete, fake_move_failed, fake_get_tick,
        fake_calibration_start, fake_calibration_pause,
        fake_calibration_resume, fake_calibration_stop,
        fake_calibration_pulses, fake_move_pulses,
        fake_move_pulses_at_speed,
        fake_homing_start, fake_homing_process, fake_send_response,
        fake_set_beeper, fake_set_light, fake_set_homing_zero,
        fake_capture_home_reference,
        fake_recorded_home_start, fake_recorded_home_process,
        fake_set_task1_open_loop
    };

    Task_Init(&ops);
    assert(Task_GetCurrentId() == 1u);
    assert(Task_GetState() == TASK_STATE_IDLE);
    assert(task1_open_loop_enabled == 0u);

    Task_AutoStartTask1();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    assert(strcmp(Task_GetOperationText(), "TASK1 WAIT JSON") == 0);
    {
        unsigned pause_before = pause_count;
        unsigned resume_before = resume_count;
        Task_StartOrTogglePause();
        Task_StartOrTogglePause();
        assert(Task_GetState() == TASK_STATE_RUNNING);
        assert(pause_count == pause_before);
        assert(resume_count == resume_before);
        assert(strcmp(Task_GetOperationText(), "TASK1 WAIT JSON") == 0);
    }

    Task_Init(&ops);
    home_reference_capture_count = 0u;

    Task_SelectNext(); assert(Task_GetCurrentId() == 2u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 3u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 4u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 5u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 6u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 7u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 8u);
    Task_SelectNext(); assert(Task_GetCurrentId() == 9u);
    Task_SelectNext();
    assert(Task_GetCurrentId() == 1u);
    assert(Task_GetState() == TASK_STATE_RUNNING);
    assert(task1_open_loop_enabled == 1u);
    assert(home_reference_capture_count == 1u);
    assert(start_count == 0u);
    assert(strcmp(Task_GetOperationText(), "TASK1 WAIT JSON") == 0);
    {
        VisionPlan_t plan = {0};
        plan.count = 1u;
        plan.commands[0].piece_id = 0u;
        plan.commands[0].pickup_x_pulse = 100u;
        plan.commands[0].pickup_y_pulse = 200u;
        plan.commands[0].place_x_pulse = 300u;
        plan.commands[0].place_y_pulse = 400u;
        plan.commands[0].rotation_deg = 90u;
        assert(Task_SubmitVisionPlan(&plan) == 1u);
    }
    Task_Process();
    assert(speed_move_count[TASK_AXIS_X] == 1u);
    assert(speed_move_count[TASK_AXIS_Y] == 1u);
    assert(speed_move_pulses[TASK_AXIS_X] == 100u);
    assert(speed_move_pulses[TASK_AXIS_Y] == 200u);
    assert(speed_move_rpm[TASK_AXIS_X] == 3000u);
    assert(speed_move_rpm[TASK_AXIS_Y] == 3000u);
    assert(speed_move_count[TASK_AXIS_Z] == 0u);
    speed_move_done[TASK_AXIS_X] = 1u;
    Task_Process();
    assert(speed_move_count[TASK_AXIS_Z] == 0u);
    speed_move_done[TASK_AXIS_Y] = 1u;
    Task_Process();
    Task_Process();
    assert(speed_move_count[TASK_AXIS_Z] == 1u);
    assert(speed_move_rpm[TASK_AXIS_Z] == 1500u);
    speed_move_done[TASK_AXIS_Z] = 1u;
    Task_Process();
    Task_Process();
    assert(magnet_mode == TASK_MAGNET_ATTRACT);
    fake_tick = 100u;
    Task_StartOrTogglePause();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    assert(pause_count == 0u);
    fake_tick = 1000u;
    Task_StartOrTogglePause();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    assert(resume_count == 0u);
    Task_Process();
    Task_Process();
    assert(speed_move_count[TASK_AXIS_Z] == 2u);
    speed_move_done[TASK_AXIS_Z] = 1u;
    Task_Process();
    Task_Process();
    assert(speed_move_count[TASK_AXIS_X] == 2u);
    assert(speed_move_count[TASK_AXIS_Y] == 2u);
    assert(speed_move_pulses[TASK_AXIS_X] == 200u);
    assert(speed_move_pulses[TASK_AXIS_Y] == 200u);
    assert(speed_move_rpm[TASK_AXIS_X] == 3000u);
    assert(speed_move_rpm[TASK_AXIS_Y] == 3000u);
    assert(move_axis == TASK_AXIS_R && move_direction == 1u && move_degrees == 90u);
    assert(magnet_mode == TASK_MAGNET_ATTRACT);
    speed_move_failed[TASK_AXIS_X] = 1u;
    Task_Process();
    assert(Task_GetState() == TASK_STATE_ERROR);
    assert(magnet_mode == TASK_MAGNET_ATTRACT);
    assert(speed_move_count[TASK_AXIS_Z] == 2u);
    Task_Exit();
    assert(task1_open_loop_enabled == 0u);
    assert(Task_GetState() == TASK_STATE_RUNNING);
    assert(homing_start_count == 1u);
    assert(strcmp(Task_GetOperationText(), "EXIT RETURN HOME") == 0);
    assert(magnet_mode == TASK_MAGNET_REPEL);
    assert(abort_count == 2u);
    homing_result = 1u;
    Task_Process();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    homing_result = 2u;
    Task_Process();
    assert(Task_GetState() == TASK_STATE_IDLE);
    memset(speed_move_count, 0, sizeof(speed_move_count));
    memset(speed_move_done, 0, sizeof(speed_move_done));
    memset(speed_move_failed, 0, sizeof(speed_move_failed));
    speed_move_tracking = 0u;

    /* Task 1 transitions directly between pieces: X/Y retain placement,
       Z is already raised, and R resets with the next pickup X/Y group. */
    Task_Init(&ops);
    fake_tick = 0u;
    move_done = 0u;
    homing_result = 2u;
    memset(speed_move_count, 0, sizeof(speed_move_count));
    memset(speed_move_done, 0, sizeof(speed_move_done));
    memset(speed_move_failed, 0, sizeof(speed_move_failed));
    speed_move_tracking = 0u;
    Task_AutoStartTask1();
    {
        VisionPlan_t plan = {0};
        unsigned homing_before = homing_start_count;
        unsigned i;
        uint8_t reached_piece_two = 0u;
        uint8_t z_count_before;
        plan.count = 2u;
        plan.commands[0].piece_id = 0u;
        plan.commands[0].pickup_x_pulse = 100u;
        plan.commands[0].pickup_y_pulse = 200u;
        plan.commands[0].place_x_pulse = 300u;
        plan.commands[0].place_y_pulse = 400u;
        plan.commands[0].rotation_deg = 90u;
        plan.commands[1].piece_id = 1u;
        plan.commands[1].pickup_x_pulse = 500u;
        plan.commands[1].pickup_y_pulse = 600u;
        plan.commands[1].place_x_pulse = 700u;
        plan.commands[1].place_y_pulse = 800u;
        plan.commands[1].rotation_deg = 45u;
        assert(Task_SubmitVisionPlan(&plan) == 1u);
        for (i = 0u; i < 64u; ++i)
        {
            memset(speed_move_done, 1, sizeof(speed_move_done));
            move_done = 1u;
            fake_tick += 600u;
            Task_Process();
            if (strcmp(Task_GetOperationText(), "JSON PIECE 1 RUN") == 0)
            {
                reached_piece_two = 1u;
                break;
            }
        }
        assert(reached_piece_two == 1u);
        assert(homing_start_count == homing_before);
        z_count_before = (uint8_t)speed_move_count[TASK_AXIS_Z];
        Task_Process();
        assert(speed_move_pulses[TASK_AXIS_X] == 200u);
        assert(speed_move_pulses[TASK_AXIS_Y] == 200u);
        assert(speed_move_rpm[TASK_AXIS_X] == 3000u);
        assert(speed_move_rpm[TASK_AXIS_Y] == 3000u);
        assert(move_axis == TASK_AXIS_R && move_direction == 0u &&
               move_degrees == 90u);
        assert(speed_move_count[TASK_AXIS_Z] == z_count_before);
        for (i = 0u; i < 64u && homing_start_count == homing_before; ++i)
        {
            memset(speed_move_done, 1, sizeof(speed_move_done));
            move_done = 1u;
            fake_tick += 600u;
            Task_Process();
        }
        assert(homing_start_count == homing_before + 1u);
    }

    /* Task 2 owns the former Task 9 PIECE and ZERO protocols. */
    Task_Init(&ops);
    memset(speed_move_count, 0, sizeof(speed_move_count));
    memset(speed_move_done, 0, sizeof(speed_move_done));
    memset(speed_move_failed, 0, sizeof(speed_move_failed));
    Task_SelectNext();
    assert(Task_GetCurrentId() == 2u);
    Task_StartOrTogglePause();
    assert(task1_open_loop_enabled == 0u);
    assert(strcmp(Task_GetOperationText(), "TASK2 WAIT PIECE") == 0);
    assert(home_reference_capture_count == 3u);
    { const char *zero = "ZERO\n"; const char *p;
      for (p = zero; *p; ++p) Task_CameraRxByte((uint8_t)*p); }
    Task_Process();
    assert(homing_zero_count == 1u);
    assert(strcmp(last_response, "ZERO,OK") == 0);
    { const char *cmd = "PIECE,7,100,200,300,400,90\n"; const char *p;
      for (p = cmd; *p; ++p) Task_CameraRxByte((uint8_t)*p); }
    Task_Process();
    assert(speed_move_count[TASK_AXIS_X] == 1u);
    assert(speed_move_count[TASK_AXIS_Y] == 1u);
    assert(speed_move_rpm[TASK_AXIS_X] == 2000u);
    assert(speed_move_rpm[TASK_AXIS_Y] == 2000u);
    Task_Exit();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    homing_result = 3u;
    Task_Process();
    assert(Task_GetState() == TASK_STATE_ERROR);
    homing_result = 2u;

    /* Tasks 3-6 are X/Y/Z/R calibration tests. */
    {
        uint8_t task_id;
        for (task_id = 3u; task_id <= 6u; ++task_id)
        {
            uint8_t select;
            Task_Init(&ops);
            for (select = 1u; select < task_id; ++select) Task_SelectNext();
            Task_StartOrTogglePause();
            assert(calibration_axis == (uint8_t)(task_id - 3u));
            assert(calibration_rate == 100u);
            calibration_pulses = (uint32_t)(task_id * 10u);
            Task_Process();
            Task_StartOrTogglePause();
            assert(Task_GetState() == TASK_STATE_PAUSED);
            Task_StartOrTogglePause();
            assert(Task_GetState() == TASK_STATE_RUNNING);
            Task_Exit();
            assert(Task_GetState() == TASK_STATE_RUNNING);
            Task_Process();
            assert(Task_GetState() == TASK_STATE_IDLE);
        }
    }

    /* Task 7 holds attraction; pause and exit safely release. */
    Task_Init(&ops);
    { uint8_t i; for (i = 1u; i < 7u; ++i) Task_SelectNext(); }
    fake_tick = 0u;
    Task_StartOrTogglePause();
    assert(magnet_mode == TASK_MAGNET_ATTRACT && magnet_duty == 10000u);
    Task_StartOrTogglePause();
    assert(Task_GetState() == TASK_STATE_PAUSED);
    assert(magnet_mode == TASK_MAGNET_REPEL);
    fake_tick = 50u; Task_Process();
    assert(magnet_mode == TASK_MAGNET_OFF);
    Task_StartOrTogglePause();
    assert(magnet_mode == TASK_MAGNET_ATTRACT);
    Task_Exit();
    assert(magnet_mode == TASK_MAGNET_REPEL);

    /* Task 8 demagnetizes for 50 ms, turns off, and completes. */
    Task_Init(&ops);
    { uint8_t i; for (i = 1u; i < 8u; ++i) Task_SelectNext(); }
    fake_tick = 100u;
    Task_StartOrTogglePause();
    assert(magnet_mode == TASK_MAGNET_REPEL);
    fake_tick = 149u; Task_Process();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    fake_tick = 150u; Task_Process();
    assert(magnet_mode == TASK_MAGNET_OFF && magnet_duty == 0u);
    assert(Task_GetState() == TASK_STATE_DONE);
    assert(Task_GetProgress() == 100u);

    /* Task 9 performs one recorded four-axis return. */
    Task_Init(&ops);
    { uint8_t i; for (i = 1u; i < 9u; ++i) Task_SelectNext(); }
    recorded_home_result = 1u;
    Task_StartOrTogglePause();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    assert(recorded_home_start_count == 1u);
    assert(strcmp(Task_GetOperationText(), "TASK9 REC HOME") == 0);
    Task_Process();
    assert(Task_GetState() == TASK_STATE_RUNNING);
    recorded_home_result = 2u;
    Task_Process();
    assert(Task_GetState() == TASK_STATE_DONE);
    assert(Task_GetProgress() == 100u);

    Task_Init(&ops);
    { uint8_t i; for (i = 1u; i < 9u; ++i) Task_SelectNext(); }
    recorded_home_result = 3u;
    Task_StartOrTogglePause();
    Task_Process();
    assert(Task_GetState() == TASK_STATE_ERROR);

    puts("task: PASS");
    return 0;
}
