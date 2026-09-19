#include "four_axis.h"

#include "axis_limit.h"
#include "axis_position.h"
#include "emm42.h"
#include "debug_uart.h"
#include "main.h"
#include "usart.h"
#include <stdio.h>
#include <string.h>

#define AXIS_FULL_TURN_PULSES 3200u
#define NORMAL_XY_SPEED_RPM 2000u
#define NORMAL_Z_SPEED_RPM 1000u
#define NORMAL_R_SPEED_RPM 1500u
#define TASK1_R_SPEED_RPM 3000u
#define CONSERVATIVE_AXIS_SPEED_RPM 1000u
#define CALIBRATION_SERIAL_SPEED_RPM 300u
#define NORMAL_AXIS_ACCELERATION 30u
#define TASK1_AXIS_ACCELERATION 50u
#define CONSERVATIVE_AXIS_ACCELERATION 10u
#define POSITION_POLL_MS 25u
#define ARRIVAL_CONFIRM_POLLS 3u
#define MOVE_TIMEOUT_MARGIN_MS 5000u
#define EMM42_ADDRESS 1u
#define EMM42_HOMING_MODE_SINGLE_TURN_NEAREST 0x00u
#define EMM42_HOMING_MODE_SINGLE_TURN_DIRECTIONAL 0x01u
#define EMM42_HOMING_MODE_MULTI_TURN_NO_LIMIT 0x02u
#define EMM42_HOMING_RUNNING 0x04u
#define EMM42_HOMING_FAILED 0x08u
#define HOMING_TIMEOUT_MS 20000u
#define STARTUP_HOMING_DELAY_MS 1000u
#define STARTUP_X_ZERO_PULSES 160u
#define STARTUP_Y_ZERO_PULSES 640u
#define STARTUP_XY_ZERO_TIMEOUT_MS 5000u
#define HOMING_COMMAND_ATTEMPTS 3u
#define X_SERIAL_DIRECTION_INVERT 0u
#define Y_SERIAL_DIRECTION_INVERT 0u
#define Z_SERIAL_DIRECTION_INVERT 0u
#define R_SERIAL_DIRECTION_INVERT 1u
#define Z_MAX_PULSES 300u
#define Y_MAX_PULSES 16800u
#define X_MAX_PULSES 29500u
#define CALIBRATION_SERIAL_INTERVAL_MS 1000u
#define HOME_ENCODER_DEADBAND_PULSES 20u
#define HOME_ENCODER_MAX_CORRECTION_PULSES 500u
#define HOME_ENCODER_CORRECTION_TIMEOUT_MS 2000u
#define RECORDED_HOME_TOLERANCE_PULSES 20u
#define RECORDED_HOME_MAX_CORRECTION_PULSES 500u
#define RECORDED_HOME_MAX_CORRECTION_ROUNDS 2u
#define RECORDED_HOME_CORRECTION_TIMEOUT_MS 5000u
#define MOVE_ACK_RECOVERY_TIMEOUT_MS 300u
#define MOVE_POSITION_SNAPSHOT_ATTEMPTS 3u
#define ENCODER_UNITS_PER_REVOLUTION 65536u

typedef struct
{
    UART_HandleTypeDef *uart;
    int32_t current_position;
    uint8_t running;
    uint8_t complete;
} SerialAxis_t;

typedef enum
{
    HOMING_KIND_DRIVER = 0,
    HOMING_KIND_SOFTWARE_RETURN
} HomingKind_t;

typedef enum
{
    RECORDED_HOME_PHASE_IDLE = 0,
    RECORDED_HOME_PHASE_COARSE,
    RECORDED_HOME_PHASE_MEASURE,
    RECORDED_HOME_PHASE_CORRECT
} RecordedHomePhase_t;

typedef enum
{
    SERIAL_SEND_ACK_OK = 0,
    SERIAL_SEND_ACK_TIMEOUT,
    SERIAL_SEND_TX_ERROR,
    SERIAL_SEND_REJECT
} SerialSendResult_t;

/* X/Y/Z/R are Emm42 serial-bus steppers. */
static SerialAxis_t x_serial_axis;
static SerialAxis_t y_serial_axis;
static SerialAxis_t z_serial_axis;
static SerialAxis_t r_serial_axis;
static uint32_t last_position_poll;
static FourAxisHomingState_t homing_state;
static uint32_t startup_started;
static uint32_t homing_started;
static uint32_t homing_last_poll;
static uint8_t x_homing_complete;
static uint8_t y_homing_complete;
static uint8_t z_homing_complete;
static uint8_t r_homing_complete;
static uint8_t x_homing_confirm_count;
static uint8_t y_homing_confirm_count;
static uint8_t z_homing_confirm_count;
static uint8_t r_homing_confirm_count;
static HomingKind_t homing_kind;
static int32_t r_position_pulses;
static uint32_t sync_move_started;
static uint32_t sync_move_duration;
static uint32_t sync_remaining_pulses;
static uint8_t sync_running;
static uint8_t generic_move_active[4];
static uint8_t generic_move_complete[4];
static uint8_t generic_move_failed[4];
static uint8_t generic_move_open_loop[4];
static uint8_t generic_arrival_confirm_count[4];
static uint32_t generic_move_started[4];
static uint32_t generic_move_duration[4];
static uint32_t generic_move_timeout[4];
static uint8_t generic_direction[4];
static uint32_t generic_pulses[4];
static uint16_t generic_speed_rpm[4];
static uint8_t calibration_axis = 0xFFu;
static uint8_t calibration_running;
static uint8_t calibration_paused;
static uint16_t calibration_pulses_per_second;
static uint32_t calibration_commanded_pulses;
static uint32_t calibration_last_batch;
static AxisLimit_t x_position_limit;
static AxisLimit_t y_position_limit;
static AxisLimit_t z_position_limit;
static FourAxisStartupState_t startup_state;
static char startup_text[24];
static char homing_error_text[24];
static uint32_t startup_xy_zero_started;
static uint8_t startup_x_confirm_count;
static uint8_t startup_y_confirm_count;
static uint8_t home_encoder_correction_phase;
static uint8_t home_encoder_correction_active[4];
static uint8_t home_encoder_correction_confirm[4];
static uint32_t home_encoder_correction_started;
static int32_t home_encoder_reference[4];
static uint8_t home_encoder_reference_valid[4];
static int32_t recorded_home_reference[4];
static uint8_t recorded_home_reference_valid[4];
static FourAxisRecordedHomeState_t recorded_home_state;
static RecordedHomePhase_t recorded_home_phase;
static uint8_t recorded_home_correction_round;
static uint8_t recorded_home_correction_active[4];
static uint8_t recorded_home_correction_confirm[4];
static uint32_t recorded_home_correction_started;
static uint32_t recorded_home_last_poll;
static uint8_t homing_skip_encoder_correction;
static uint8_t task1_open_loop_mode;
static uint8_t software_homing_open_loop;
static uint32_t software_homing_open_loop_duration[4];

static const char *serial_uart_name(const SerialAxis_t *axis);
static void driver_homing_start(void);
static void home_encoder_correction_finish(void);
static uint8_t software_homing_poll(SerialAxis_t *axis,
                                    uint8_t *confirm_count);
static void startup_xy_zero_start(void);
static void startup_xy_zero_process(void);
static uint8_t startup_capture_recorded_home(void);
static void home_encoder_correction_start(void);
static uint8_t home_encoder_correction_poll(void);

static const char *serial_axis_name(const SerialAxis_t *axis)
{
    if (axis == &x_serial_axis) return "X";
    if (axis == &y_serial_axis) return "Y";
    if (axis == &z_serial_axis) return "Z";
    if (axis == &r_serial_axis) return "R";
    return "?";
}


static SerialAxis_t *serial_axis_by_id(uint8_t axis)
{
    if (axis == FOUR_AXIS_X) return &x_serial_axis;
    if (axis == FOUR_AXIS_Y) return &y_serial_axis;
    if (axis == FOUR_AXIS_Z) return &z_serial_axis;
    if (axis == FOUR_AXIS_R) return &r_serial_axis;
    return NULL;
}

static uint32_t motion_duration_ms(uint16_t degrees)
{
    return 100u + (((uint32_t)degrees * 60000u) /
                   (360u * NORMAL_XY_SPEED_RPM));
}

static void serial_drain_rx(SerialAxis_t *axis)
{
    volatile uint32_t discard;
    while (__HAL_UART_GET_FLAG(axis->uart, UART_FLAG_RXNE) != RESET)
    {
        discard = axis->uart->Instance->DR;
        (void)discard;
    }
    if (__HAL_UART_GET_FLAG(axis->uart, UART_FLAG_ORE) != RESET)
        __HAL_UART_CLEAR_OREFLAG(axis->uart);
}

static uint8_t serial_receive_target_frame(SerialAxis_t *axis,
                                           uint8_t expected_function,
                                           uint8_t expected_size,
                                           uint8_t *reply)
{
    Emm42FrameScanner_t scanner;
    uint32_t started = HAL_GetTick();
    uint8_t byte;
    Emm42_FrameScannerInit(&scanner, EMM42_ADDRESS, expected_function,
                           expected_size);
    while ((uint32_t)(HAL_GetTick() - started) < 50u)
    {
        if (HAL_UART_Receive(axis->uart, &byte, 1u, 5u) != HAL_OK) continue;
        if (Emm42_FrameScannerFeed(&scanner, byte, reply))
        {
            DebugUart_LogFrameRx(serial_uart_name(axis),
                                 serial_axis_name(axis), reply,
                                 expected_size);
            return 1u;
        }
    }
    return 0u;
}

static SerialSendResult_t serial_send_result(SerialAxis_t *axis,
                                              const uint8_t *packet,
                                              uint16_t size,
                                              uint8_t expected_function)
{
    uint8_t reply[4];
    serial_drain_rx(axis);
    if (HAL_UART_Transmit(axis->uart, (uint8_t *)packet, size, 50u) != HAL_OK)
    {
        printf("[EMM42] fn=%02X TX ERROR\r\n", expected_function);
        return SERIAL_SEND_TX_ERROR;
    }
    if (!serial_receive_target_frame(axis, expected_function,
                                     sizeof(reply), reply))
    {
        printf("[EMM42] fn=%02X RX TIMEOUT/ERROR\r\n", expected_function);
        return SERIAL_SEND_ACK_TIMEOUT;
    }
    if ((reply[0] != EMM42_ADDRESS) || (reply[1] != expected_function) ||
        (reply[2] != 0x02u) || (reply[3] != 0x6Bu))
    {
        printf("[EMM42] fn=%02X BAD %02X %02X %02X %02X\r\n",
               expected_function, reply[0], reply[1], reply[2], reply[3]);
        return SERIAL_SEND_REJECT;
    }
    return SERIAL_SEND_ACK_OK;
}

static uint8_t serial_send(SerialAxis_t *axis, const uint8_t *packet,
                           uint16_t size, uint8_t expected_function)
{
    return (serial_send_result(axis, packet, size, expected_function) ==
            SERIAL_SEND_ACK_OK) ? 1u : 0u;
}

static uint8_t serial_read_position(SerialAxis_t *axis)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    uint8_t reply[8];
    size_t size = Emm42_BuildReadPosition(EMM42_ADDRESS, packet);

    serial_drain_rx(axis);
    if (HAL_UART_Transmit(axis->uart, packet, (uint16_t)size, 50u) != HAL_OK)
    {
        return 0u;
    }
    if (!serial_receive_target_frame(axis, 0x36u, sizeof(reply), reply))
    {
        return 0u;
    }
    return Emm42_ParsePosition(EMM42_ADDRESS, reply, sizeof(reply), &axis->current_position) ? 1u : 0u;
}

static void serial_clear_receive_errors(SerialAxis_t *axis)
{
    if ((__HAL_UART_GET_FLAG(axis->uart, UART_FLAG_ORE) != RESET) ||
        (__HAL_UART_GET_FLAG(axis->uart, UART_FLAG_FE) != RESET) ||
        (__HAL_UART_GET_FLAG(axis->uart, UART_FLAG_NE) != RESET))
    {
        __HAL_UART_CLEAR_OREFLAG(axis->uart);
    }
    serial_drain_rx(axis);
}

static uint32_t encoder_travel_to_command_pulses(int32_t start_position,
                                                  int32_t current_position)
{
    int64_t travel = (int64_t)current_position - (int64_t)start_position;
    uint64_t magnitude = (travel < 0) ? (uint64_t)(-travel) :
                                        (uint64_t)travel;
    return (uint32_t)(((magnitude * AXIS_FULL_TURN_PULSES) +
                       (ENCODER_UNITS_PER_REVOLUTION / 2u)) /
                      ENCODER_UNITS_PER_REVOLUTION);
}

static uint8_t axis_encoder_logical_positive_is_negative(
    const SerialAxis_t *axis)
{
    return (axis == &r_serial_axis) ? 1u : 0u;
}

static uint8_t serial_capture_move_start(SerialAxis_t *axis,
                                         int32_t *start_position)
{
    uint8_t attempt;
    for (attempt = 1u; attempt <= MOVE_POSITION_SNAPSHOT_ATTEMPTS; ++attempt)
    {
        if (serial_read_position(axis))
        {
            *start_position = axis->current_position;
            return 1u;
        }
        printf("[ACK RECOVER] axis=%s START POSITION ERROR %u/%u\r\n",
               serial_axis_name(axis), attempt,
               MOVE_POSITION_SNAPSHOT_ATTEMPTS);
    }
    return 0u;
}

static uint8_t axis_recover_lost_move_ack(SerialAxis_t *axis,
                                          int32_t start_position,
                                          uint8_t logical_direction,
                                          uint32_t expected_pulses)
{
    uint32_t started = HAL_GetTick();
    uint8_t received_position = 0u;
    const char *axis_name = serial_axis_name(axis);

    printf("[ACK RECOVER] axis=%s FD ACK LOST\r\n", axis_name);
    serial_clear_receive_errors(axis);
    while ((uint32_t)(HAL_GetTick() - started) <
           MOVE_ACK_RECOVERY_TIMEOUT_MS)
    {
        if (serial_read_position(axis))
        {
            uint32_t moved_pulses;
            received_position = 1u;
            moved_pulses = encoder_travel_to_command_pulses(
                start_position, axis->current_position);
            if (AxisPosition_HasMovedInDirection(
                    start_position, axis->current_position,
                    logical_direction,
                    axis_encoder_logical_positive_is_negative(axis)))
            {
                printf("[ACK RECOVER] axis=%s moved=%lu expected=%lu ACCEPT\r\n",
                       axis_name,
                       (unsigned long)moved_pulses,
                       (unsigned long)expected_pulses);
                return 1u;
            }
        }
    }
    if (received_position)
        printf("[ACK RECOVER] axis=%s NO FORWARD MOVEMENT REJECT\r\n",
               axis_name);
    else
        printf("[ACK RECOVER] axis=%s POSITION TIMEOUT REJECT\r\n",
               axis_name);
    return 0u;
}

static uint8_t serial_send_position_with_recovery(
    SerialAxis_t *axis, const uint8_t *packet, uint16_t size,
    uint8_t start_position_valid, int32_t start_position,
    uint8_t logical_direction,
    uint32_t expected_pulses)
{
    SerialSendResult_t send_result = serial_send_result(
        axis, packet, size, 0xFDu);
    if (send_result == SERIAL_SEND_ACK_OK) return 1u;
    if ((send_result == SERIAL_SEND_ACK_TIMEOUT) && start_position_valid &&
        axis_recover_lost_move_ack(
            axis, start_position, logical_direction, expected_pulses))
        return 1u;
    if (task1_open_loop_mode)
    {
        if (send_result == SERIAL_SEND_ACK_TIMEOUT)
        {
            printf("[TASK1 OPEN LOOP] axis=%s FD ACK MISSING, CONTINUE\r\n",
                   serial_axis_name(axis));
            return 1u;
        }
        if (send_result == SERIAL_SEND_TX_ERROR)
        {
            printf("[TASK1 OPEN LOOP] axis=%s TX ERROR, CONTINUE\r\n",
                   serial_axis_name(axis));
            return 1u;
        }
    }
    return 0u;
}

static const char *serial_uart_name(const SerialAxis_t *axis)
{
    if (axis->uart == &huart6) return "USART6";
    if (axis->uart == &huart3) return "USART3";
    if (axis->uart == &huart4) return "UART4";
    if (axis->uart == &huart5) return "UART5";
    return "UNKNOWN";
}

static uint8_t serial_read_status(SerialAxis_t *axis, uint8_t *status)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    uint8_t reply[4];
    size_t size = Emm42_BuildReadStatus(EMM42_ADDRESS, packet);

    serial_drain_rx(axis);
    if (HAL_UART_Transmit(axis->uart, packet, (uint16_t)size, 50u) != HAL_OK)
        return 0u;
    if (!serial_receive_target_frame(axis, 0x3Au, sizeof(reply), reply))
        return 0u;
    return Emm42_ParseStatus(EMM42_ADDRESS, reply, sizeof(reply), status) ? 1u : 0u;
}

static uint8_t serial_read_homing_state(SerialAxis_t *axis, uint8_t *state)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    uint8_t reply[4];
    size_t size = Emm42_BuildReadHomingState(EMM42_ADDRESS, packet);

    serial_drain_rx(axis);
    if (HAL_UART_Transmit(axis->uart, packet, (uint16_t)size, 50u) != HAL_OK)
        return 0u;
    if (!serial_receive_target_frame(axis, 0x3Bu, sizeof(reply), reply))
        return 0u;
    return Emm42_ParseHomingState(EMM42_ADDRESS, reply, sizeof(reply), state) ?
           1u : 0u;
}

static uint8_t serial_start_homing(SerialAxis_t *axis, uint8_t mode)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size = Emm42_BuildReturnToZero(EMM42_ADDRESS, mode, false, packet);
    return serial_send(axis, packet, (uint16_t)size, 0x9Au);
}

static void serial_start_homing_fire_and_forget(SerialAxis_t *axis,
                                                uint8_t mode)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size = Emm42_BuildReturnToZero(EMM42_ADDRESS, mode, false, packet);
    serial_drain_rx(axis);
    if (HAL_UART_Transmit(axis->uart, packet, (uint16_t)size, 50u) != HAL_OK)
        printf("[HOME] axis=%s fire-and-forget TX ERROR IGNORED\r\n",
               serial_axis_name(axis));
}

static uint8_t serial_start_homing_with_retries(SerialAxis_t *axis,
                                                uint8_t mode)
{
    uint8_t attempt;
    for (attempt = 1u; attempt <= HOMING_COMMAND_ATTEMPTS; ++attempt)
    {
        if (serial_start_homing(axis, mode)) return 1u;
        printf("[HOME] axis=%s command retry %u/%u\r\n",
               serial_axis_name(axis), attempt, HOMING_COMMAND_ATTEMPTS);
        if (attempt < HOMING_COMMAND_ATTEMPTS) HAL_Delay(50u);
    }
    return 0u;
}

static void generic_move_prepare(uint8_t axis)
{
    generic_move_active[axis] = 1u;
    generic_move_complete[axis] = 0u;
    generic_move_failed[axis] = 0u;
    generic_move_open_loop[axis] = task1_open_loop_mode;
    generic_arrival_confirm_count[axis] = 0u;
}

static uint32_t serial_command_relative_at_profile(
    SerialAxis_t *axis, uint32_t pulses, uint8_t direction,
    uint16_t speed_rpm, uint8_t acceleration)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size;
    uint8_t logical_direction = direction;
    uint8_t start_position_valid;
    int32_t start_position = 0;
    AxisLimit_t *limit = NULL;
    if (axis == &x_serial_axis) limit = &x_position_limit;
    if (axis == &y_serial_axis) limit = &y_position_limit;
    if (axis == &z_serial_axis) limit = &z_position_limit;
    if (limit != NULL) pulses = AxisLimit_Allow(limit, logical_direction, pulses);
    if (pulses == 0u)
    {
        axis->running = 0u;
        axis->complete = 1u;
        return 0u;
    }
    if (axis == &x_serial_axis) direction ^= X_SERIAL_DIRECTION_INVERT;
    if (axis == &y_serial_axis) direction ^= Y_SERIAL_DIRECTION_INVERT;
    if (axis == &z_serial_axis) direction ^= Z_SERIAL_DIRECTION_INVERT;
    if (axis == &r_serial_axis) direction ^= R_SERIAL_DIRECTION_INVERT;

    start_position_valid = serial_capture_move_start(axis, &start_position);
    if (!start_position_valid && !task1_open_loop_mode)
    {
        axis->running = 0u;
        axis->complete = 0u;
        return 0u;
    }
    if (!start_position_valid)
        printf("[TASK1 OPEN LOOP] axis=%s START POSITION MISSING, SEND\r\n",
               serial_axis_name(axis));
    size = Emm42_BuildPosition(EMM42_ADDRESS, direction ? false : true, speed_rpm,
                               acceleration, pulses, false, false, packet);

    if (serial_send_position_with_recovery(
            axis, packet, (uint16_t)size, start_position_valid, start_position,
            logical_direction, pulses))
    {
        if (limit != NULL) AxisLimit_Commit(limit, logical_direction, pulses);
        if (axis == &r_serial_axis)
        {
            if (logical_direction) r_position_pulses += (int32_t)pulses;
            else r_position_pulses -= (int32_t)pulses;
        }
        axis->running = 1u;
        axis->complete = 0u;
        return pulses;
    }
    else
    {
        axis->running = 0u;
        axis->complete = 0u;
        return 0u;
    }
}

static uint32_t serial_command_relative(SerialAxis_t *axis, uint32_t pulses,
                                        uint8_t direction, uint16_t speed_rpm)
{
    return serial_command_relative_at_profile(
        axis, pulses, direction, speed_rpm,
        task1_open_loop_mode ? TASK1_AXIS_ACCELERATION :
                               NORMAL_AXIS_ACCELERATION);
}

static uint8_t serial_command_correction(SerialAxis_t *axis, uint32_t pulses,
                                         uint8_t logical_direction,
                                         uint16_t speed_rpm)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    uint8_t direction = logical_direction;
    uint8_t start_position_valid;
    int32_t start_position;
    size_t size;

    start_position_valid = serial_capture_move_start(axis, &start_position);
    if (!start_position_valid && !task1_open_loop_mode) return 0u;
    if (!start_position_valid)
        printf("[TASK1 OPEN LOOP] axis=%s START POSITION MISSING, SEND\r\n",
               serial_axis_name(axis));

    if (axis == &x_serial_axis) direction ^= X_SERIAL_DIRECTION_INVERT;
    if (axis == &y_serial_axis) direction ^= Y_SERIAL_DIRECTION_INVERT;
    if (axis == &z_serial_axis) direction ^= Z_SERIAL_DIRECTION_INVERT;
    if (axis == &r_serial_axis) direction ^= R_SERIAL_DIRECTION_INVERT;
    size = Emm42_BuildPosition(EMM42_ADDRESS, direction ? false : true,
                               speed_rpm, CONSERVATIVE_AXIS_ACCELERATION, pulses,
                               false, false, packet);
    return serial_send_position_with_recovery(
        axis, packet, (uint16_t)size, start_position_valid, start_position,
        logical_direction, pulses);
}

static void serial_stop(SerialAxis_t *axis)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size = Emm42_BuildStop(EMM42_ADDRESS, false, packet);
    (void)serial_send(axis, packet, (uint16_t)size, 0xFEu);
    axis->running = 0u;
    (void)serial_read_position(axis);
}

void FourAxis_Init(void)
{
    x_serial_axis.uart = &huart6;
    y_serial_axis.uart = &huart3;
    z_serial_axis.uart = &huart4;
    r_serial_axis.uart = &huart5;
    x_serial_axis.running = y_serial_axis.running = z_serial_axis.running =
        r_serial_axis.running = 0u;
    x_serial_axis.complete = y_serial_axis.complete = z_serial_axis.complete =
        r_serial_axis.complete = 0u;
    last_position_poll = 0u;
    homing_state = FOUR_AXIS_HOMING_IDLE;
    sync_running = 0u;
    sync_remaining_pulses = 0u;
    memset(generic_move_active, 0, sizeof(generic_move_active));
    memset(generic_move_complete, 0, sizeof(generic_move_complete));
    memset(generic_move_failed, 0, sizeof(generic_move_failed));
    memset(generic_move_open_loop, 0, sizeof(generic_move_open_loop));
    memset(generic_arrival_confirm_count, 0,
           sizeof(generic_arrival_confirm_count));
    calibration_axis = 0xFFu;
    calibration_running = 0u;
    calibration_paused = 0u;
    AxisLimit_Init(&x_position_limit, X_MAX_PULSES);
    AxisLimit_Init(&y_position_limit, Y_MAX_PULSES);
    AxisLimit_Init(&z_position_limit, Z_MAX_PULSES);
    r_position_pulses = 0;
    home_encoder_correction_phase = 0u;
    memset(home_encoder_correction_active, 0,
           sizeof(home_encoder_correction_active));
    memset(home_encoder_correction_confirm, 0,
           sizeof(home_encoder_correction_confirm));
    memset(home_encoder_reference_valid, 0,
           sizeof(home_encoder_reference_valid));
    memset(recorded_home_reference_valid, 0,
           sizeof(recorded_home_reference_valid));
    memset(recorded_home_correction_active, 0,
           sizeof(recorded_home_correction_active));
    memset(recorded_home_correction_confirm, 0,
           sizeof(recorded_home_correction_confirm));
    recorded_home_state = FOUR_AXIS_RECORDED_HOME_IDLE;
    recorded_home_phase = RECORDED_HOME_PHASE_IDLE;
    recorded_home_correction_round = 0u;
    homing_skip_encoder_correction = 0u;
    task1_open_loop_mode = 0u;
    software_homing_open_loop = 0u;
    memset(software_homing_open_loop_duration, 0,
           sizeof(software_homing_open_loop_duration));
    startup_started = HAL_GetTick();
    startup_state = FOUR_AXIS_STARTUP_WAITING;
    strcpy(startup_text, "WAIT MOTOR READY");
    homing_error_text[0] = '\0';
}

void FourAxis_ZeroAxis(uint8_t axis)
{
    /* Multi-turn absolute-position return without a limit switch. The motor
       returns to the zero stored via O_Set -> Set O or command 0x93 0x88. */
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size = Emm42_BuildReturnToZero(
        EMM42_ADDRESS, EMM42_HOMING_MODE_MULTI_TURN_NO_LIMIT, false, packet);
    if (axis <= FOUR_AXIS_R)
        (void)serial_send(serial_axis_by_id(axis), packet, (uint16_t)size, 0x9Au);
}

void FourAxis_ZeroAllSteppers(void)
{
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size = Emm42_BuildReturnToZero(
        EMM42_ADDRESS, EMM42_HOMING_MODE_MULTI_TURN_NO_LIMIT, false, packet);
    (void)serial_send(&x_serial_axis, packet, (uint16_t)size, 0x9Au);
    (void)serial_send(&y_serial_axis, packet, (uint16_t)size, 0x9Au);
    (void)serial_send(&z_serial_axis, packet, (uint16_t)size, 0x9Au);
    (void)serial_send(&r_serial_axis, packet, (uint16_t)size, 0x9Au);
}

uint8_t FourAxis_SetHomingZeroAll(void)
{
    /* Store the current position of every axis as its homing zero (flash). */
    uint8_t packet[EMM42_MAX_PACKET_SIZE];
    size_t size = Emm42_BuildSetHomingZero(EMM42_ADDRESS, 1u, packet);
    uint8_t ok = serial_send(&x_serial_axis, packet, (uint16_t)size, 0x93u);
    ok &= serial_send(&y_serial_axis, packet, (uint16_t)size, 0x93u);
    ok &= serial_send(&z_serial_axis, packet, (uint16_t)size, 0x93u);
    ok &= serial_send(&r_serial_axis, packet, (uint16_t)size, 0x93u);
    if (ok)
    {
        AxisLimit_Reset(&x_position_limit);
        AxisLimit_Reset(&y_position_limit);
        AxisLimit_Reset(&z_position_limit);
        r_position_pulses = 0;
    }
    return ok;
}

void FourAxis_CaptureHomeReference(void)
{
    const uint8_t ids[3] = {FOUR_AXIS_X, FOUR_AXIS_Y, FOUR_AXIS_R};
    uint8_t i;

    memset(home_encoder_reference_valid, 0,
           sizeof(home_encoder_reference_valid));
    for (i = 0u; i < 3u; ++i)
    {
        uint8_t id = ids[i];
        SerialAxis_t *axis = serial_axis_by_id(id);
        if (!serial_read_position(axis))
        {
            printf("[HOME ENC WARN] axis=%s reference unavailable\r\n",
                   serial_axis_name(axis));
            continue;
        }
        home_encoder_reference[id] = axis->current_position;
        home_encoder_reference_valid[id] = 1u;
        printf("[HOME ENC] axis=%s reference=%ld\r\n",
               serial_axis_name(axis), (long)axis->current_position);
    }
}

static void startup_xy_zero_start(void)
{
    startup_x_confirm_count = 0u;
    startup_y_confirm_count = 0u;
    startup_xy_zero_started = HAL_GetTick();
    startup_state = FOUR_AXIS_STARTUP_XY_ZEROING;
    strcpy(startup_text, "XY AUTO ZERO");
    if (serial_command_relative_at_profile(
            &x_serial_axis, STARTUP_X_ZERO_PULSES, 1u,
            CONSERVATIVE_AXIS_SPEED_RPM,
            CONSERVATIVE_AXIS_ACCELERATION) !=
        STARTUP_X_ZERO_PULSES)
    {
        startup_state = FOUR_AXIS_STARTUP_ERROR;
        strcpy(startup_text, "X ZERO CMD ERR");
        return;
    }
    if (serial_command_relative_at_profile(
            &y_serial_axis, STARTUP_Y_ZERO_PULSES, 1u,
            CONSERVATIVE_AXIS_SPEED_RPM,
            CONSERVATIVE_AXIS_ACCELERATION) !=
        STARTUP_Y_ZERO_PULSES)
    {
        startup_state = FOUR_AXIS_STARTUP_ERROR;
        strcpy(startup_text, "Y ZERO CMD ERR");
    }
}

static void startup_xy_zero_process(void)
{
    uint8_t x_result;
    uint8_t y_result;

    if ((uint32_t)(HAL_GetTick() - startup_xy_zero_started) >=
        STARTUP_XY_ZERO_TIMEOUT_MS)
    {
        startup_state = FOUR_AXIS_STARTUP_ERROR;
        strcpy(startup_text, "XY ZERO TIMEOUT");
        return;
    }
    x_result = software_homing_poll(&x_serial_axis,
                                    &startup_x_confirm_count);
    if (x_result == 2u)
    {
        startup_state = FOUR_AXIS_STARTUP_ERROR;
        strcpy(startup_text, "X ZERO STATE ERR");
        return;
    }
    y_result = software_homing_poll(&y_serial_axis,
                                    &startup_y_confirm_count);
    if (y_result == 2u)
    {
        startup_state = FOUR_AXIS_STARTUP_ERROR;
        strcpy(startup_text, "Y ZERO STATE ERR");
        return;
    }
    if ((x_result != 1u) || (y_result != 1u)) return;

    x_serial_axis.running = y_serial_axis.running = 0u;
    x_serial_axis.complete = y_serial_axis.complete = 1u;
    AxisLimit_Reset(&x_position_limit);
    AxisLimit_Reset(&y_position_limit);
    startup_state = FOUR_AXIS_STARTUP_RETURNING;
    strcpy(startup_text, "HOMING Z R");
    driver_homing_start();
}

static uint8_t startup_capture_recorded_home(void)
{
    SerialAxis_t *axes[4] = {
        &x_serial_axis, &y_serial_axis, &z_serial_axis, &r_serial_axis
    };
    uint8_t i;

    memset(recorded_home_reference_valid, 0,
           sizeof(recorded_home_reference_valid));
    for (i = 0u; i < 4u; ++i)
    {
        if (!serial_read_position(axes[i]))
        {
            printf("[ZERO REF] axis=%s read error\r\n",
                   serial_axis_name(axes[i]));
            return 0u;
        }
        recorded_home_reference[i] = axes[i]->current_position;
        recorded_home_reference_valid[i] = 1u;
        printf("[ZERO REF] axis=%s enc=%ld pulse=0\r\n",
               serial_axis_name(axes[i]),
               (long)recorded_home_reference[i]);
    }
    return 1u;
}

FourAxisStartupState_t FourAxis_GetStartupState(void)
{
    return startup_state;
}

const char *FourAxis_GetStartupText(void)
{
    return startup_text;
}

void FourAxis_Process(void)
{
    uint32_t now = HAL_GetTick();
    uint8_t axis;
    if (startup_state == FOUR_AXIS_STARTUP_WAITING)
    {
        if ((uint32_t)(now - startup_started) < STARTUP_HOMING_DELAY_MS) return;
        startup_xy_zero_start();
    }
    if (startup_state == FOUR_AXIS_STARTUP_XY_ZEROING)
    {
        if ((uint32_t)(now - last_position_poll) < POSITION_POLL_MS) return;
        last_position_poll = now;
        startup_xy_zero_process();
        return;
    }
    if (startup_state == FOUR_AXIS_STARTUP_RETURNING)
    {
        FourAxisHomingState_t result = FourAxis_HomingProcess();
        if (result == FOUR_AXIS_HOMING_DONE)
        {
            if (startup_capture_recorded_home())
            {
                startup_state = FOUR_AXIS_STARTUP_READY;
                strcpy(startup_text, "READY ZERO REC");
            }
            else
            {
                startup_state = FOUR_AXIS_STARTUP_ERROR;
                strcpy(startup_text, "ZERO RECORD ERR");
            }
        }
        else if (result == FOUR_AXIS_HOMING_ERROR)
        {
            startup_state = FOUR_AXIS_STARTUP_ERROR;
            snprintf(startup_text, sizeof(startup_text), "%s",
                     (homing_error_text[0] != '\0') ? homing_error_text :
                     "Z R HOME ERROR");
        }
    }
    if (startup_state != FOUR_AXIS_STARTUP_READY) return;
    if ((uint32_t)(now - last_position_poll) < POSITION_POLL_MS)
    {
        return;
    }
    last_position_poll = now;
    if (calibration_running && !calibration_paused &&
        (calibration_axis <= FOUR_AXIS_R) &&
        ((uint32_t)(now - calibration_last_batch) >= CALIBRATION_SERIAL_INTERVAL_MS))
    {
        /* Z axis is limited to Z_MAX_PULSES total; stop calibration when reached. */
        if (calibration_axis == FOUR_AXIS_Z &&
            calibration_commanded_pulses >= Z_MAX_PULSES)
        {
            calibration_running = 0u;
            calibration_paused = 0u;
            calibration_axis = 0xFFu;
        }
        else
        {
            SerialAxis_t *cal_axis = serial_axis_by_id(calibration_axis);
            uint32_t actual = serial_command_relative_at_profile(
                cal_axis, calibration_pulses_per_second, 1u,
                CALIBRATION_SERIAL_SPEED_RPM,
                CONSERVATIVE_AXIS_ACCELERATION);
            calibration_commanded_pulses += actual;
            calibration_last_batch = now;
            if (actual == 0u)
            {
                calibration_running = 0u;
                calibration_paused = 0u;
                calibration_axis = 0xFFu;
                return;
            }
            /* Diagnostic: prove whether the driver answers on the serial bus. */
            if (serial_read_position(cal_axis))
            {
                printf("[CAL] axis=%u ack pos=%ld\r\n", calibration_axis,
                       (long)cal_axis->current_position);
            }
            else
            {
                printf("[CAL] axis=%u NO REPLY\r\n", calibration_axis);
            }
        }
    }
    for (axis = 0u; axis <= FOUR_AXIS_R; ++axis)
    {
        uint32_t elapsed;
        SerialAxis_t *serial_axis;
        uint8_t motor_status;
        if (!generic_move_active[axis]) continue;
        elapsed = (uint32_t)(now - generic_move_started[axis]);
        if (elapsed >= generic_move_timeout[axis])
        {
            generic_move_active[axis] = 0u;
            if (generic_move_open_loop[axis])
            {
                generic_move_complete[axis] = 1u;
                printf("[TASK1 OPEN LOOP] axis=%s ARRIVAL TIMEOUT, CONTINUE\r\n",
                       serial_axis_name(serial_axis_by_id(axis)));
            }
            else
            {
                generic_move_failed[axis] = 1u;
                printf("[MOVE] axis=%u TIMEOUT\r\n", axis);
            }
            continue;
        }
        if (elapsed < generic_move_duration[axis]) continue;
        serial_axis = serial_axis_by_id(axis);
        if (!serial_read_status(serial_axis, &motor_status))
        {
            generic_arrival_confirm_count[axis] = 0u;
            if (generic_move_open_loop[axis])
            {
                generic_move_active[axis] = 0u;
                generic_move_complete[axis] = 1u;
                printf("[TASK1 OPEN LOOP] axis=%s STATUS MISSING, TIME COMPLETE\r\n",
                       serial_axis_name(serial_axis));
            }
            continue;
        }
        if ((motor_status & (EMM42_STATUS_STALLED |
                             EMM42_STATUS_STALL_PROTECTION)) != 0u)
        {
            generic_move_active[axis] = 0u;
            generic_move_failed[axis] = 1u;
            printf("[MOVE] axis=%u STALL status=%02X\r\n",
                   axis, motor_status);
            continue;
        }
        if ((motor_status & EMM42_STATUS_IN_POSITION) != 0u)
        {
            if (generic_arrival_confirm_count[axis] < ARRIVAL_CONFIRM_POLLS)
                ++generic_arrival_confirm_count[axis];
        }
        else generic_arrival_confirm_count[axis] = 0u;
        if (generic_arrival_confirm_count[axis] >= ARRIVAL_CONFIRM_POLLS)
        {
            generic_move_active[axis] = 0u;
            generic_move_complete[axis] = 1u;
            printf("[MOVE] axis=%u DRIVER ARRIVED\r\n", axis);
        }
    }

    /* Task 1: all three serial steppers run at the same configured speed, so a
       shared time budget is the non-blocking completion reference. */
    if (sync_running && ((uint32_t)(now - sync_move_started) >= sync_move_duration))
    {
        sync_running = 0u;
        if (x_serial_axis.running) { x_serial_axis.running = 0u; x_serial_axis.complete = 1u; }
        if (y_serial_axis.running) { y_serial_axis.running = 0u; y_serial_axis.complete = 1u; }
        if (z_serial_axis.running) { z_serial_axis.running = 0u; z_serial_axis.complete = 1u; }
        if (r_serial_axis.running) { r_serial_axis.running = 0u; r_serial_axis.complete = 1u; }
    }
}

void FourAxis_StartAll360(void)
{
    uint32_t r_duration;
    serial_command_relative(&x_serial_axis, AXIS_FULL_TURN_PULSES, 1u,
                            NORMAL_XY_SPEED_RPM);
    serial_command_relative(&y_serial_axis, AXIS_FULL_TURN_PULSES, 1u,
                            NORMAL_XY_SPEED_RPM);
    serial_command_relative(&z_serial_axis, Z_MAX_PULSES, 1u,
                            NORMAL_Z_SPEED_RPM);
    serial_command_relative(&r_serial_axis, AXIS_FULL_TURN_PULSES, 1u,
                            NORMAL_R_SPEED_RPM);
    sync_move_started = HAL_GetTick();
    /* X/Y/R = 360 degrees; Z is limited to 300 pulses. */
    sync_move_duration = motion_duration_ms(360u);
    r_duration = 100u + (60000u / NORMAL_R_SPEED_RPM);
    if (sync_move_duration < r_duration) sync_move_duration = r_duration;
    sync_remaining_pulses = AXIS_FULL_TURN_PULSES;
    sync_running = 1u;
}

void FourAxis_PauseAll(void)
{
    uint8_t axis_id;
    uint8_t generic_running = 0u;
    for (axis_id = FOUR_AXIS_X; axis_id <= FOUR_AXIS_R; ++axis_id)
    {
        if (generic_move_active[axis_id])
        {
            uint32_t elapsed = HAL_GetTick() - generic_move_started[axis_id];
            SerialAxis_t *axis = serial_axis_by_id(axis_id);
            generic_running = 1u;
            serial_stop(axis);
            if (elapsed < generic_move_duration[axis_id])
            {
                generic_pulses[axis_id] = (generic_pulses[axis_id] *
                    (generic_move_duration[axis_id] - elapsed)) /
                    generic_move_duration[axis_id];
                generic_move_duration[axis_id] -= elapsed;
            }
            else generic_pulses[axis_id] = 0u;
        }
    }
    if (generic_running) return;
    if (sync_running)
    {
        uint32_t elapsed = HAL_GetTick() - sync_move_started;
        if (elapsed < sync_move_duration)
        {
            sync_remaining_pulses = (AXIS_FULL_TURN_PULSES *
                (sync_move_duration - elapsed)) / sync_move_duration;
            sync_move_duration -= elapsed;
        }
        else sync_remaining_pulses = 0u;
        sync_running = 0u;
    }
    serial_stop(&x_serial_axis);
    serial_stop(&y_serial_axis);
    serial_stop(&z_serial_axis);
    serial_stop(&r_serial_axis);
}

void FourAxis_ResumeAll(void)
{
    uint8_t axis_id;
    uint8_t generic_running = 0u;
    for (axis_id = FOUR_AXIS_X; axis_id <= FOUR_AXIS_R; ++axis_id)
    {
        if (generic_move_active[axis_id])
        {
            SerialAxis_t *axis = serial_axis_by_id(axis_id);
            generic_running = 1u;
            generic_move_started[axis_id] = HAL_GetTick();
            if (generic_pulses[axis_id] != 0u)
                (void)serial_command_relative(
                    axis, generic_pulses[axis_id], generic_direction[axis_id],
                    generic_speed_rpm[axis_id]);
        }
    }
    if (generic_running) return;
    if (sync_remaining_pulses != 0u)
    {
        if (!x_serial_axis.complete)
            serial_command_relative(&x_serial_axis, sync_remaining_pulses, 1u,
                                    NORMAL_XY_SPEED_RPM);
        if (!y_serial_axis.complete)
            serial_command_relative(&y_serial_axis, sync_remaining_pulses, 1u,
                                    NORMAL_XY_SPEED_RPM);
        if (!z_serial_axis.complete)
        {
            uint32_t z_resume = sync_remaining_pulses;
            if (z_resume > Z_MAX_PULSES) z_resume = Z_MAX_PULSES;
            serial_command_relative(&z_serial_axis, z_resume, 1u,
                                    NORMAL_Z_SPEED_RPM);
        }
        if (!r_serial_axis.complete)
            serial_command_relative(&r_serial_axis, sync_remaining_pulses, 1u,
                                    NORMAL_R_SPEED_RPM);
        sync_move_started = HAL_GetTick();
        sync_running = 1u;
    }
}

void FourAxis_AbortAll(void)
{
    serial_stop(&x_serial_axis);
    serial_stop(&y_serial_axis);
    serial_stop(&z_serial_axis);
    serial_stop(&r_serial_axis);
    sync_running = 0u;
    sync_remaining_pulses = 0u;
    x_serial_axis.complete = 0u;
    y_serial_axis.complete = 0u;
    z_serial_axis.complete = 0u;
    r_serial_axis.complete = 0u;
    memset(generic_move_active, 0, sizeof(generic_move_active));
    memset(generic_move_complete, 0, sizeof(generic_move_complete));
    memset(generic_move_failed, 0, sizeof(generic_move_failed));
    memset(generic_arrival_confirm_count, 0,
           sizeof(generic_arrival_confirm_count));
}

uint8_t FourAxis_GetProgress(uint8_t axis)
{
    if (axis <= FOUR_AXIS_R)
    {
        uint32_t elapsed;
        const SerialAxis_t *serial_axis = serial_axis_by_id(axis);
        if (serial_axis->complete) return 100u;
        if (!sync_running || (sync_move_duration == 0u)) return 0u;
        elapsed = HAL_GetTick() - sync_move_started;
        return (elapsed >= sync_move_duration) ? 100u :
               (uint8_t)((elapsed * 100u) / sync_move_duration);
    }
    return 0u;
}

uint8_t FourAxis_AllComplete(void)
{
    return x_serial_axis.complete && y_serial_axis.complete &&
           z_serial_axis.complete && r_serial_axis.complete;
}

void FourAxis_StartMove(uint8_t axis, uint8_t direction, uint16_t degrees)
{
    uint32_t pulses;
    uint32_t actual;
    uint16_t speed_rpm;
    if (axis > FOUR_AXIS_R) return;
    pulses = (((uint32_t)degrees * AXIS_FULL_TURN_PULSES) + 180u) / 360u;
    generic_move_prepare(axis);
    generic_move_started[axis] = HAL_GetTick();
    generic_direction[axis] = direction;
    generic_pulses[axis] = pulses;
    speed_rpm = (axis == FOUR_AXIS_R) ?
                (task1_open_loop_mode ? TASK1_R_SPEED_RPM :
                                        NORMAL_R_SPEED_RPM) :
                ((axis == FOUR_AXIS_Z) ? NORMAL_Z_SPEED_RPM :
                                         NORMAL_XY_SPEED_RPM);
    generic_speed_rpm[axis] = speed_rpm;

    {
        SerialAxis_t *serial_axis = serial_axis_by_id(axis);
        actual = serial_command_relative(serial_axis, pulses, direction,
                                         speed_rpm);
        generic_pulses[axis] = actual;
        if (actual == 0u)
        {
            generic_move_active[axis] = 0u;
            generic_move_complete[axis] = serial_axis->complete;
            generic_move_failed[axis] = serial_axis->complete ? 0u : 1u;
            generic_move_duration[axis] = 0u;
            generic_move_timeout[axis] = 0u;
            return;
        }
        generic_move_duration[axis] = 100u +
            ((actual * 60000u) / (AXIS_FULL_TURN_PULSES * speed_rpm));
        generic_move_timeout[axis] = generic_move_duration[axis] +
                                     MOVE_TIMEOUT_MARGIN_MS;
    }
}

void FourAxis_StartMovePulses(uint8_t axis, uint8_t direction, uint32_t pulses)
{
    uint16_t speed = (axis == FOUR_AXIS_R) ? NORMAL_R_SPEED_RPM :
                     ((axis == FOUR_AXIS_Z) ? NORMAL_Z_SPEED_RPM :
                                              NORMAL_XY_SPEED_RPM);
    FourAxis_StartMovePulsesAtSpeed(axis, direction, pulses, speed);
}

void FourAxis_SetTask1OpenLoop(uint8_t enabled)
{
    task1_open_loop_mode = enabled ? 1u : 0u;
}

void FourAxis_StartMovePulsesAtSpeed(uint8_t axis, uint8_t direction,
                                    uint32_t pulses, uint16_t speed_rpm)
{
    uint32_t actual;
    SerialAxis_t *serial_axis;
    if (axis > FOUR_AXIS_Z) return;
    generic_move_prepare(axis);
    generic_move_started[axis] = HAL_GetTick();
    generic_direction[axis] = direction;
    generic_speed_rpm[axis] = speed_rpm;
    serial_axis = serial_axis_by_id(axis);
    actual = serial_command_relative(serial_axis, pulses, direction,
                                     speed_rpm);
    generic_pulses[axis] = actual;
    if (actual == 0u)
    {
        generic_move_active[axis] = 0u;
        generic_move_complete[axis] = serial_axis->complete;
        generic_move_failed[axis] = serial_axis->complete ? 0u : 1u;
        generic_move_duration[axis] = 0u;
        generic_move_timeout[axis] = 0u;
        return;
    }
    /* ms per pulse = 60000 / (pulses_per_rev * rpm); valid while
       pulses * 60000 stays below 2^32 (~71k pulses at 150 RPM). */
    generic_move_duration[axis] = 100u +
        ((actual * 60000u) / (AXIS_FULL_TURN_PULSES * speed_rpm));
    generic_move_timeout[axis] = generic_move_duration[axis] +
                                 MOVE_TIMEOUT_MARGIN_MS;
}

uint8_t FourAxis_MoveComplete(uint8_t axis)
{
    if ((axis <= FOUR_AXIS_R) && generic_move_complete[axis])
        return 1u;
    return 0u;
}

uint8_t FourAxis_MoveFailed(uint8_t axis)
{
    return ((axis <= FOUR_AXIS_R) && generic_move_failed[axis]) ? 1u : 0u;
}

void FourAxis_StartCalibration(uint8_t axis, uint16_t pulses_per_second)
{
    calibration_axis = axis;
    calibration_running = 1u;
    calibration_paused = 0u;
    calibration_pulses_per_second = pulses_per_second;
    calibration_commanded_pulses = 0u;
    calibration_last_batch = HAL_GetTick();
}

void FourAxis_PauseCalibration(void)
{
    if (!calibration_running || calibration_paused) return;
    if (calibration_axis <= FOUR_AXIS_R) serial_stop(serial_axis_by_id(calibration_axis));
    calibration_paused = 1u;
}

void FourAxis_ResumeCalibration(void)
{
    if (!calibration_running || !calibration_paused) return;
    if (calibration_axis <= FOUR_AXIS_R) calibration_last_batch = HAL_GetTick();
    calibration_paused = 0u;
}

void FourAxis_StopCalibration(void)
{
    if (!calibration_running) return;
    if (calibration_axis <= FOUR_AXIS_R) serial_stop(serial_axis_by_id(calibration_axis));
    calibration_running = 0u;
    calibration_paused = 0u;
    calibration_axis = 0xFFu;
}

uint32_t FourAxis_GetCalibrationPulses(void)
{
    if (calibration_axis <= FOUR_AXIS_R) return calibration_commanded_pulses;
    return 0u;
}

static void driver_homing_start(void)
{
    uint8_t r_ok;
    homing_started = HAL_GetTick();
    homing_last_poll = homing_started - POSITION_POLL_MS;
    homing_state = FOUR_AXIS_HOMING_RUNNING;
    homing_kind = HOMING_KIND_DRIVER;
    x_homing_complete = 1u;
    y_homing_complete = 1u;
    z_homing_complete = 1u;
    r_homing_complete = 0u;
    x_homing_confirm_count = 0u;
    y_homing_confirm_count = 0u;
    z_homing_confirm_count = 0u;
    r_homing_confirm_count = 0u;
    homing_error_text[0] = '\0';
    serial_start_homing_fire_and_forget(
        &z_serial_axis, EMM42_HOMING_MODE_SINGLE_TURN_NEAREST);
    r_ok = serial_start_homing_with_retries(
        &r_serial_axis, EMM42_HOMING_MODE_SINGLE_TURN_NEAREST);
    if (!r_ok)
    {
        strcpy(homing_error_text, "R HOME CMD ERR");
        homing_state = FOUR_AXIS_HOMING_ERROR;
    }
}

static void software_homing_start(uint8_t skip_encoder_correction)
{
    uint32_t x_pulses = AxisLimit_GetPosition(&x_position_limit);
    uint32_t y_pulses = AxisLimit_GetPosition(&y_position_limit);
    uint32_t z_pulses = AxisLimit_GetPosition(&z_position_limit);
    int32_t r_pulses = r_position_pulses;
    uint32_t r_magnitude = (r_pulses < 0) ? (uint32_t)(-r_pulses) :
                                            (uint32_t)r_pulses;
    uint8_t ok = 1u;

    homing_started = HAL_GetTick();
    homing_last_poll = homing_started - POSITION_POLL_MS;
    homing_state = FOUR_AXIS_HOMING_RUNNING;
    homing_kind = HOMING_KIND_SOFTWARE_RETURN;
    software_homing_open_loop = task1_open_loop_mode;
    software_homing_open_loop_duration[FOUR_AXIS_X] = 100u +
        ((x_pulses * 60000u) /
         (AXIS_FULL_TURN_PULSES * CONSERVATIVE_AXIS_SPEED_RPM));
    software_homing_open_loop_duration[FOUR_AXIS_Y] = 100u +
        ((y_pulses * 60000u) /
         (AXIS_FULL_TURN_PULSES * CONSERVATIVE_AXIS_SPEED_RPM));
    software_homing_open_loop_duration[FOUR_AXIS_Z] = 100u +
        ((z_pulses * 60000u) /
         (AXIS_FULL_TURN_PULSES * CONSERVATIVE_AXIS_SPEED_RPM));
    software_homing_open_loop_duration[FOUR_AXIS_R] = 100u +
        ((r_magnitude * 60000u) /
         (AXIS_FULL_TURN_PULSES * CONSERVATIVE_AXIS_SPEED_RPM));
    homing_skip_encoder_correction = skip_encoder_correction;
    home_encoder_correction_phase = 0u;
    homing_error_text[0] = '\0';
    x_homing_complete = (x_pulses == 0u) ? 1u : 0u;
    y_homing_complete = (y_pulses == 0u) ? 1u : 0u;
    z_homing_complete = (z_pulses == 0u) ? 1u : 0u;
    r_homing_complete = (r_magnitude == 0u) ? 1u : 0u;
    x_homing_confirm_count = 0u;
    y_homing_confirm_count = 0u;
    z_homing_confirm_count = 0u;
    r_homing_confirm_count = 0u;

    if (x_pulses != 0u &&
        serial_command_relative_at_profile(
            &x_serial_axis, x_pulses, 0u, CONSERVATIVE_AXIS_SPEED_RPM,
            CONSERVATIVE_AXIS_ACCELERATION) != x_pulses)
    {
        strcpy(homing_error_text, "X RETURN CMD ERR");
        ok = 0u;
    }
    if (y_pulses != 0u &&
        serial_command_relative_at_profile(
            &y_serial_axis, y_pulses, 0u, CONSERVATIVE_AXIS_SPEED_RPM,
            CONSERVATIVE_AXIS_ACCELERATION) != y_pulses)
    {
        strcpy(homing_error_text, "Y RETURN CMD ERR");
        ok = 0u;
    }
    if (z_pulses != 0u &&
        serial_command_relative_at_profile(
            &z_serial_axis, z_pulses, 0u, CONSERVATIVE_AXIS_SPEED_RPM,
            CONSERVATIVE_AXIS_ACCELERATION) != z_pulses)
    {
        strcpy(homing_error_text, "Z RETURN CMD ERR");
        ok = 0u;
    }
    if (r_magnitude != 0u &&
        serial_command_relative_at_profile(
            &r_serial_axis, r_magnitude, (r_pulses < 0) ? 1u : 0u,
            CONSERVATIVE_AXIS_SPEED_RPM,
            CONSERVATIVE_AXIS_ACCELERATION) != r_magnitude)
    {
        strcpy(homing_error_text, "R RETURN CMD ERR");
        ok = 0u;
    }
    if (!ok) homing_state = FOUR_AXIS_HOMING_ERROR;
    else if (x_homing_complete && y_homing_complete && z_homing_complete &&
             r_homing_complete)
    {
        if (homing_skip_encoder_correction) home_encoder_correction_finish();
        else home_encoder_correction_start();
    }
}

void FourAxis_HomingStart(void)
{
    software_homing_start(0u);
}

static uint8_t software_homing_poll(SerialAxis_t *axis,
                                    uint8_t *confirm_count)
{
    uint8_t status;
    if (!serial_read_status(axis, &status))
    {
        uint8_t id = (axis == &x_serial_axis) ? FOUR_AXIS_X :
                     ((axis == &y_serial_axis) ? FOUR_AXIS_Y :
                     ((axis == &z_serial_axis) ? FOUR_AXIS_Z : FOUR_AXIS_R));
        *confirm_count = 0u;
        if (software_homing_open_loop &&
            ((uint32_t)(HAL_GetTick() - homing_started) >=
             software_homing_open_loop_duration[id]))
        {
            printf("[TASK1 OPEN LOOP] axis=%s RETURN STATUS MISSING, CONTINUE\r\n",
                   serial_axis_name(axis));
            return 1u;
        }
        return 0u;
    }
    if ((status & (EMM42_STATUS_STALLED |
                   EMM42_STATUS_STALL_PROTECTION)) != 0u)
        return 2u;
    if ((status & EMM42_STATUS_IN_POSITION) == 0u)
    {
        *confirm_count = 0u;
        return 0u;
    }
    if (*confirm_count < ARRIVAL_CONFIRM_POLLS) ++(*confirm_count);
    return (*confirm_count >= ARRIVAL_CONFIRM_POLLS) ? 1u : 0u;
}

static void home_encoder_correction_finish(void)
{
    AxisLimit_Reset(&x_position_limit);
    AxisLimit_Reset(&y_position_limit);
    AxisLimit_Reset(&z_position_limit);
    r_position_pulses = 0;
    home_encoder_correction_phase = 0u;
    homing_skip_encoder_correction = 0u;
    homing_state = FOUR_AXIS_HOMING_DONE;
}

static void home_encoder_correction_start(void)
{
    SerialAxis_t *axes[3] = {&x_serial_axis, &y_serial_axis, &r_serial_axis};
    const uint8_t ids[3] = {FOUR_AXIS_X, FOUR_AXIS_Y, FOUR_AXIS_R};
    const uint8_t positive_is_negative[3] = {0u, 0u, 1u};
    uint8_t i;
    uint8_t any_active = 0u;

    home_encoder_correction_phase = 1u;
    home_encoder_correction_started = HAL_GetTick();
    memset(home_encoder_correction_active, 0,
           sizeof(home_encoder_correction_active));
    memset(home_encoder_correction_confirm, 0,
           sizeof(home_encoder_correction_confirm));
    for (i = 0u; i < 3u; ++i)
    {
        uint8_t direction;
        uint32_t pulses;
        int64_t wide_error;
        int32_t encoder_error;
        if (!home_encoder_reference_valid[ids[i]])
        {
            printf("[HOME ENC WARN] axis=%s no reference, skip\r\n",
                   serial_axis_name(axes[i]));
            continue;
        }
        if (!serial_read_position(axes[i]))
        {
            printf("[HOME ENC WARN] axis=%s position unavailable, skip\r\n",
                   serial_axis_name(axes[i]));
            continue;
        }
        wide_error = (int64_t)axes[i]->current_position -
                     (int64_t)home_encoder_reference[ids[i]];
        if (wide_error > 2147483647LL) encoder_error = 2147483647L;
        else if (wide_error < -2147483647LL) encoder_error = -2147483647L;
        else encoder_error = (int32_t)wide_error;
        if (!AxisPosition_SelectCorrection(
                encoder_error, HOME_ENCODER_DEADBAND_PULSES,
                HOME_ENCODER_MAX_CORRECTION_PULSES,
                positive_is_negative[i], &direction, &pulses))
        {
            printf("[HOME ENC] axis=%s error=%ld within deadband\r\n",
                   serial_axis_name(axes[i]), (long)encoder_error);
            continue;
        }
        printf("[HOME ENC] axis=%s error=%ld correct dir=%u pulses=%lu\r\n",
               serial_axis_name(axes[i]), (long)encoder_error,
               direction, (unsigned long)pulses);
        if (!serial_command_correction(
                axes[i], pulses, direction,
                CONSERVATIVE_AXIS_SPEED_RPM))
        {
            printf("[HOME ENC WARN] axis=%s correction command skipped\r\n",
                   serial_axis_name(axes[i]));
            continue;
        }
        home_encoder_correction_active[ids[i]] = 1u;
        any_active = 1u;
    }
    if (!any_active) home_encoder_correction_finish();
}

static uint8_t home_encoder_correction_poll(void)
{
    const uint8_t ids[3] = {FOUR_AXIS_X, FOUR_AXIS_Y, FOUR_AXIS_R};
    uint8_t i;
    uint8_t any_active = 0u;

    if ((uint32_t)(HAL_GetTick() - home_encoder_correction_started) >=
        HOME_ENCODER_CORRECTION_TIMEOUT_MS)
    {
        printf("[HOME ENC WARN] correction timeout, continue task\r\n");
        home_encoder_correction_finish();
        return 1u;
    }
    for (i = 0u; i < 3u; ++i)
    {
        uint8_t id = ids[i];
        uint8_t result;
        SerialAxis_t *axis;
        if (!home_encoder_correction_active[id]) continue;
        axis = serial_axis_by_id(id);
        result = software_homing_poll(axis,
                                      &home_encoder_correction_confirm[id]);
        if (result == 1u)
        {
            home_encoder_correction_active[id] = 0u;
            printf("[HOME ENC] axis=%s correction arrived\r\n",
                   serial_axis_name(axis));
        }
        else if (result == 2u)
        {
            home_encoder_correction_active[id] = 0u;
            printf("[HOME ENC WARN] axis=%s correction state ignored\r\n",
                   serial_axis_name(axis));
        }
        if (home_encoder_correction_active[id]) any_active = 1u;
    }
    if (!any_active)
    {
        home_encoder_correction_finish();
        return 1u;
    }
    return 0u;
}

FourAxisHomingState_t FourAxis_HomingProcess(void)
{
    uint32_t now;
    uint8_t state;
    if (homing_state != FOUR_AXIS_HOMING_RUNNING)
        return homing_state;
    now = HAL_GetTick();
    if ((uint32_t)(now - homing_started) >= HOMING_TIMEOUT_MS)
    {
        if ((homing_kind == HOMING_KIND_SOFTWARE_RETURN) &&
            home_encoder_correction_phase)
        {
            printf("[HOME ENC WARN] overall timeout, continue task\r\n");
            home_encoder_correction_finish();
            return homing_state;
        }
        if ((homing_kind == HOMING_KIND_SOFTWARE_RETURN) &&
            software_homing_open_loop)
        {
            printf("[TASK1 OPEN LOOP] RETURN TIMEOUT, CONTINUE\r\n");
            home_encoder_correction_finish();
            return homing_state;
        }
        strcpy(homing_error_text,
               (homing_kind == HOMING_KIND_DRIVER) ?
               "Z R HOME TIMEOUT" : "Z R RETURN TIMEOUT");
        homing_state = FOUR_AXIS_HOMING_ERROR;
        return homing_state;
    }
    if ((uint32_t)(now - homing_last_poll) < POSITION_POLL_MS)
        return homing_state;
    homing_last_poll = now;
    if (homing_kind == HOMING_KIND_SOFTWARE_RETURN)
    {
        uint8_t result;
        if (home_encoder_correction_phase)
        {
            (void)home_encoder_correction_poll();
            return homing_state;
        }
        if (!x_homing_complete)
        {
            result = software_homing_poll(&x_serial_axis,
                                          &x_homing_confirm_count);
            if (result == 2u)
            {
                strcpy(homing_error_text, "X RETURN STATE ERR");
                homing_state = FOUR_AXIS_HOMING_ERROR;
            }
            else if (result == 1u) x_homing_complete = 1u;
        }
        if ((homing_state == FOUR_AXIS_HOMING_RUNNING) && !y_homing_complete)
        {
            result = software_homing_poll(&y_serial_axis,
                                          &y_homing_confirm_count);
            if (result == 2u)
            {
                strcpy(homing_error_text, "Y RETURN STATE ERR");
                homing_state = FOUR_AXIS_HOMING_ERROR;
            }
            else if (result == 1u) y_homing_complete = 1u;
        }
        if (!z_homing_complete)
        {
            result = software_homing_poll(&z_serial_axis,
                                          &z_homing_confirm_count);
            if (result == 2u)
            {
                strcpy(homing_error_text, "Z RETURN STATE ERR");
                homing_state = FOUR_AXIS_HOMING_ERROR;
            }
            else if (result == 1u) z_homing_complete = 1u;
        }
        if ((homing_state == FOUR_AXIS_HOMING_RUNNING) && !r_homing_complete)
        {
            result = software_homing_poll(&r_serial_axis,
                                          &r_homing_confirm_count);
            if (result == 2u)
            {
                strcpy(homing_error_text, "R RETURN STATE ERR");
                homing_state = FOUR_AXIS_HOMING_ERROR;
            }
            else if (result == 1u) r_homing_complete = 1u;
        }
        if ((homing_state == FOUR_AXIS_HOMING_RUNNING) && x_homing_complete &&
            y_homing_complete && z_homing_complete && r_homing_complete)
        {
            if (homing_skip_encoder_correction) home_encoder_correction_finish();
            else home_encoder_correction_start();
        }
        return homing_state;
    }
    if ((homing_state == FOUR_AXIS_HOMING_RUNNING) && !r_homing_complete &&
        serial_read_homing_state(&r_serial_axis, &state))
    {
        if ((state & EMM42_HOMING_FAILED) != 0u)
        {
            strcpy(homing_error_text, "R HOME STATE ERR");
            homing_state = FOUR_AXIS_HOMING_ERROR;
        }
        else if ((state & EMM42_HOMING_RUNNING) == 0u)
            r_homing_complete = 1u;
    }
    if ((homing_state == FOUR_AXIS_HOMING_RUNNING) && z_homing_complete &&
        r_homing_complete)
    {
        z_serial_axis.running = r_serial_axis.running = 0u;
        z_serial_axis.complete = r_serial_axis.complete = 1u;
        AxisLimit_Reset(&z_position_limit);
        r_position_pulses = 0;
        homing_state = FOUR_AXIS_HOMING_DONE;
    }
    return homing_state;
}

FourAxisHomingState_t FourAxis_GetHomingState(void)
{
    return homing_state;
}

static void recorded_home_fail(const char *text)
{
    recorded_home_state = FOUR_AXIS_RECORDED_HOME_ERROR;
    recorded_home_phase = RECORDED_HOME_PHASE_IDLE;
    printf("[REC HOME ERR] %s\r\n", text);
}

static void recorded_home_measure(void)
{
    SerialAxis_t *axes[4] = {
        &x_serial_axis, &y_serial_axis, &z_serial_axis, &r_serial_axis
    };
    const uint8_t positive_is_negative[4] = {0u, 0u, 0u, 1u};
    uint8_t directions[4] = {0u, 0u, 0u, 0u};
    uint32_t pulses[4] = {0u, 0u, 0u, 0u};
    uint8_t any_correction = 0u;
    uint8_t i;

    memset(recorded_home_correction_active, 0,
           sizeof(recorded_home_correction_active));
    memset(recorded_home_correction_confirm, 0,
           sizeof(recorded_home_correction_confirm));
    for (i = 0u; i < 4u; ++i)
    {
        int64_t wide_error;
        int32_t encoder_error;
        if (!recorded_home_reference_valid[i])
        {
            recorded_home_fail("REFERENCE INVALID");
            return;
        }
        if (!serial_read_position(axes[i]))
        {
            recorded_home_fail("POSITION READ");
            return;
        }
        wide_error = (int64_t)axes[i]->current_position -
                     (int64_t)recorded_home_reference[i];
        if (wide_error > 2147483647LL) encoder_error = 2147483647L;
        else if (wide_error < -2147483647LL) encoder_error = -2147483647L;
        else encoder_error = (int32_t)wide_error;
        if (AxisPosition_SelectCorrection(
                encoder_error, RECORDED_HOME_TOLERANCE_PULSES,
                RECORDED_HOME_MAX_CORRECTION_PULSES,
                positive_is_negative[i], &directions[i], &pulses[i]))
            any_correction = 1u;
        printf("[REC HOME] axis=%s error=%ld correction=%lu\r\n",
               serial_axis_name(axes[i]), (long)encoder_error,
               (unsigned long)pulses[i]);
    }
    if (!any_correction)
    {
        AxisLimit_Reset(&x_position_limit);
        AxisLimit_Reset(&y_position_limit);
        AxisLimit_Reset(&z_position_limit);
        r_position_pulses = 0;
        recorded_home_state = FOUR_AXIS_RECORDED_HOME_DONE;
        recorded_home_phase = RECORDED_HOME_PHASE_IDLE;
        return;
    }
    if (recorded_home_correction_round >=
        RECORDED_HOME_MAX_CORRECTION_ROUNDS)
    {
        recorded_home_fail("RESIDUAL ERROR");
        return;
    }
    for (i = 0u; i < 4u; ++i)
    {
        uint16_t speed;
        if (pulses[i] == 0u) continue;
        speed = CONSERVATIVE_AXIS_SPEED_RPM;
        if (!serial_command_correction(axes[i], pulses[i], directions[i],
                                       speed))
        {
            recorded_home_fail("CORRECTION CMD");
            return;
        }
        recorded_home_correction_active[i] = 1u;
    }
    ++recorded_home_correction_round;
    recorded_home_correction_started = HAL_GetTick();
    recorded_home_last_poll = recorded_home_correction_started -
                              POSITION_POLL_MS;
    recorded_home_phase = RECORDED_HOME_PHASE_CORRECT;
}

void FourAxis_RecordedHomeStart(void)
{
    uint8_t i;
    for (i = 0u; i < 4u; ++i)
    {
        if (!recorded_home_reference_valid[i])
        {
            recorded_home_fail("REFERENCE INVALID");
            return;
        }
    }
    recorded_home_state = FOUR_AXIS_RECORDED_HOME_RUNNING;
    recorded_home_phase = RECORDED_HOME_PHASE_COARSE;
    recorded_home_correction_round = 0u;
    software_homing_start(1u);
}

FourAxisRecordedHomeState_t FourAxis_RecordedHomeProcess(void)
{
    uint32_t now;
    uint8_t i;
    uint8_t any_active;

    if (recorded_home_state != FOUR_AXIS_RECORDED_HOME_RUNNING)
        return recorded_home_state;
    if (recorded_home_phase == RECORDED_HOME_PHASE_COARSE)
    {
        FourAxisHomingState_t coarse = FourAxis_HomingProcess();
        if (coarse == FOUR_AXIS_HOMING_ERROR)
            recorded_home_fail("COARSE RETURN");
        else if (coarse == FOUR_AXIS_HOMING_DONE)
            recorded_home_phase = RECORDED_HOME_PHASE_MEASURE;
        return recorded_home_state;
    }
    if (recorded_home_phase == RECORDED_HOME_PHASE_MEASURE)
    {
        recorded_home_measure();
        return recorded_home_state;
    }
    if (recorded_home_phase != RECORDED_HOME_PHASE_CORRECT)
    {
        recorded_home_fail("PHASE");
        return recorded_home_state;
    }

    now = HAL_GetTick();
    if ((uint32_t)(now - recorded_home_correction_started) >=
        RECORDED_HOME_CORRECTION_TIMEOUT_MS)
    {
        recorded_home_fail("CORRECTION TIMEOUT");
        return recorded_home_state;
    }
    if ((uint32_t)(now - recorded_home_last_poll) < POSITION_POLL_MS)
        return recorded_home_state;
    recorded_home_last_poll = now;
    any_active = 0u;
    for (i = 0u; i < 4u; ++i)
    {
        uint8_t result;
        if (!recorded_home_correction_active[i]) continue;
        result = software_homing_poll(serial_axis_by_id(i),
                                      &recorded_home_correction_confirm[i]);
        if (result == 2u)
        {
            recorded_home_fail("CORRECTION STATE");
            return recorded_home_state;
        }
        if (result == 1u) recorded_home_correction_active[i] = 0u;
        if (recorded_home_correction_active[i]) any_active = 1u;
    }
    if (!any_active) recorded_home_phase = RECORDED_HOME_PHASE_MEASURE;
    return recorded_home_state;
}
