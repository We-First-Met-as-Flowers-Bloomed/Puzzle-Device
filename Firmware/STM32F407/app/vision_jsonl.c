#include "vision_jsonl.h"

#include <stdlib.h>
#include <string.h>

#define VISION_RX_LINE_MAX 1024u
#define VISION_RX_SLOTS 2u

typedef struct
{
    const char *cursor;
} JsonParser_t;

static char rx_lines[VISION_RX_SLOTS][VISION_RX_LINE_MAX];
static volatile uint16_t rx_lengths[VISION_RX_SLOTS];
static volatile uint8_t rx_write_slot;
static volatile uint8_t rx_read_slot;
static volatile uint8_t rx_ready_count;
static volatile uint8_t rx_overflow;

static void skip_space(JsonParser_t *parser)
{
    while ((*parser->cursor == ' ') || (*parser->cursor == '\t') ||
           (*parser->cursor == '\r') || (*parser->cursor == '\n'))
        ++parser->cursor;
}

static uint8_t take_char(JsonParser_t *parser, char expected)
{
    skip_space(parser);
    if (*parser->cursor != expected) return 0u;
    ++parser->cursor;
    return 1u;
}

static uint8_t parse_string(JsonParser_t *parser, char *output, size_t output_size)
{
    size_t length = 0u;
    if (!take_char(parser, '"')) return 0u;
    while ((*parser->cursor != '\0') && (*parser->cursor != '"'))
    {
        char value = *parser->cursor++;
        if ((value == '\\') || ((unsigned char)value < 0x20u)) return 0u;
        if ((output != NULL) && (length + 1u < output_size)) output[length] = value;
        ++length;
    }
    if (*parser->cursor != '"') return 0u;
    ++parser->cursor;
    if ((output != NULL) && (output_size != 0u))
        output[(length < output_size) ? length : (output_size - 1u)] = '\0';
    return (output == NULL) || (length < output_size);
}

static uint8_t parse_bool(JsonParser_t *parser, uint8_t *value)
{
    skip_space(parser);
    if (strncmp(parser->cursor, "true", 4u) == 0)
    {
        parser->cursor += 4;
        *value = 1u;
        return 1u;
    }
    if (strncmp(parser->cursor, "false", 5u) == 0)
    {
        parser->cursor += 5;
        *value = 0u;
        return 1u;
    }
    return 0u;
}

static uint8_t parse_uint(JsonParser_t *parser, uint32_t *value)
{
    char *end;
    unsigned long parsed;
    skip_space(parser);
    if ((*parser->cursor < '0') || (*parser->cursor > '9')) return 0u;
    parsed = strtoul(parser->cursor, &end, 10);
    if (end == parser->cursor) return 0u;
    parser->cursor = end;
    *value = (uint32_t)parsed;
    return parsed <= 0xFFFFFFFFul;
}

static uint8_t parse_rotation(JsonParser_t *parser, uint16_t *rotation)
{
    char *end;
    double parsed;
    uint16_t rounded;
    skip_space(parser);
    if ((*parser->cursor < '0') || (*parser->cursor > '9')) return 0u;
    parsed = strtod(parser->cursor, &end);
    if ((end == parser->cursor) || (parsed < 0.0) || (parsed >= 360.0))
        return 0u;
    parser->cursor = end;
    rounded = (uint16_t)(parsed + 0.5);
    *rotation = rounded;
    return 1u;
}

static uint8_t skip_value(JsonParser_t *parser);

static uint8_t skip_array(JsonParser_t *parser)
{
    if (!take_char(parser, '[')) return 0u;
    skip_space(parser);
    if (take_char(parser, ']')) return 1u;
    do
    {
        if (!skip_value(parser)) return 0u;
        skip_space(parser);
        if (take_char(parser, ']')) return 1u;
    } while (take_char(parser, ','));
    return 0u;
}

static uint8_t skip_object(JsonParser_t *parser)
{
    char key[32];
    if (!take_char(parser, '{')) return 0u;
    skip_space(parser);
    if (take_char(parser, '}')) return 1u;
    do
    {
        if (!parse_string(parser, key, sizeof(key)) ||
            !take_char(parser, ':') || !skip_value(parser)) return 0u;
        skip_space(parser);
        if (take_char(parser, '}')) return 1u;
    } while (take_char(parser, ','));
    return 0u;
}

static uint8_t skip_value(JsonParser_t *parser)
{
    char *end;
    skip_space(parser);
    if (*parser->cursor == '"') return parse_string(parser, NULL, 0u);
    if (*parser->cursor == '{') return skip_object(parser);
    if (*parser->cursor == '[') return skip_array(parser);
    if (strncmp(parser->cursor, "true", 4u) == 0) { parser->cursor += 4; return 1u; }
    if (strncmp(parser->cursor, "false", 5u) == 0) { parser->cursor += 5; return 1u; }
    if (strncmp(parser->cursor, "null", 4u) == 0) { parser->cursor += 4; return 1u; }
    (void)strtod(parser->cursor, &end);
    if (end == parser->cursor) return 0u;
    parser->cursor = end;
    return 1u;
}

static uint8_t parse_command(JsonParser_t *parser, VisionCommand_t *command)
{
    char key[32];
    uint8_t fields = 0u;
    uint32_t value;
    memset(command, 0, sizeof(*command));
    if (!take_char(parser, '{')) return 0u;
    do
    {
        if (!parse_string(parser, key, sizeof(key)) || !take_char(parser, ':'))
            return 0u;
        if (strcmp(key, "piece_id") == 0)
        {
            if ((fields & 0x01u) || !parse_uint(parser, &value) || (value > 3u)) return 0u;
            command->piece_id = (uint8_t)value; fields |= 0x01u;
        }
        else if (strcmp(key, "pickup_x_pulse") == 0)
        {
            if ((fields & 0x02u) || !parse_uint(parser, &value) || (value > 24000u)) return 0u;
            command->pickup_x_pulse = value; fields |= 0x02u;
        }
        else if (strcmp(key, "pickup_y_pulse") == 0)
        {
            if ((fields & 0x04u) || !parse_uint(parser, &value) || (value > 16700u)) return 0u;
            command->pickup_y_pulse = value; fields |= 0x04u;
        }
        else if (strcmp(key, "place_x_pulse") == 0)
        {
            if ((fields & 0x08u) || !parse_uint(parser, &value) || (value > 24000u)) return 0u;
            command->place_x_pulse = value; fields |= 0x08u;
        }
        else if (strcmp(key, "place_y_pulse") == 0)
        {
            if ((fields & 0x10u) || !parse_uint(parser, &value) || (value > 16700u)) return 0u;
            command->place_y_pulse = value; fields |= 0x10u;
        }
        else if (strcmp(key, "rotation_deg") == 0)
        {
            if ((fields & 0x20u) || !parse_rotation(parser, &command->rotation_deg)) return 0u;
            fields |= 0x20u;
        }
        else if (!skip_value(parser)) return 0u;
        skip_space(parser);
        if (take_char(parser, '}')) break;
        if (!take_char(parser, ',')) return 0u;
    } while (1);
    return fields == 0x3Fu;
}

static uint8_t parse_commands(JsonParser_t *parser, VisionPlan_t *plan)
{
    uint8_t ids = 0u;
    if (!take_char(parser, '[')) return 0u;
    skip_space(parser);
    if (take_char(parser, ']')) { plan->count = 0u; return 1u; }
    plan->count = 0u;
    do
    {
        VisionCommand_t *command;
        if (plan->count >= VISION_PLAN_MAX_COMMANDS) return 0u;
        command = &plan->commands[plan->count];
        if (!parse_command(parser, command) ||
            ((ids & (uint8_t)(1u << command->piece_id)) != 0u)) return 0u;
        ids |= (uint8_t)(1u << command->piece_id);
        ++plan->count;
        skip_space(parser);
        if (take_char(parser, ']')) return 1u;
    } while (take_char(parser, ','));
    return 0u;
}

static VisionJsonlResult_t parse_line(const char *line, VisionPlan_t *plan,
                                      char *status, size_t status_size)
{
    JsonParser_t parser;
    char key[32];
    uint8_t fields = 0u;
    uint8_t calibrated = 0u;
    uint32_t elapsed = 0u;
    parser.cursor = line;
    memset(plan, 0, sizeof(*plan));
    if ((status != NULL) && (status_size != 0u)) status[0] = '\0';
    if (!take_char(&parser, '{')) return VISION_JSONL_REJECT;
    do
    {
        if (!parse_string(&parser, key, sizeof(key)) || !take_char(&parser, ':'))
            return VISION_JSONL_REJECT;
        if (strcmp(key, "status") == 0)
        {
            if ((fields & 0x01u) || !parse_string(&parser, status, status_size))
                return VISION_JSONL_REJECT;
            fields |= 0x01u;
        }
        else if (strcmp(key, "robot_calibrated") == 0)
        {
            if ((fields & 0x02u) || !parse_bool(&parser, &calibrated))
                return VISION_JSONL_REJECT;
            fields |= 0x02u;
        }
        else if (strcmp(key, "elapsed_ms") == 0)
        {
            if ((fields & 0x04u) || !parse_uint(&parser, &elapsed))
                return VISION_JSONL_REJECT;
            fields |= 0x04u;
        }
        else if (strcmp(key, "commands") == 0)
        {
            if ((fields & 0x08u) || !parse_commands(&parser, plan))
                return VISION_JSONL_REJECT;
            fields |= 0x08u;
        }
        else if (!skip_value(&parser)) return VISION_JSONL_REJECT;
        skip_space(&parser);
        if (take_char(&parser, '}')) break;
        if (!take_char(&parser, ',')) return VISION_JSONL_REJECT;
    } while (1);
    skip_space(&parser);
    if (*parser.cursor != '\0') return VISION_JSONL_REJECT;
    if (!(fields & 0x01u)) return VISION_JSONL_REJECT;
    if (strcmp(status, "OK") != 0) return VISION_JSONL_REPORT;
    if ((fields != 0x0Fu) || !calibrated ||
        (plan->count == 0u) || (plan->count > VISION_PLAN_MAX_COMMANDS))
        return VISION_JSONL_REJECT;
    plan->elapsed_ms = elapsed;
    return VISION_JSONL_PLAN;
}

void VisionJsonl_Init(void)
{
    memset(rx_lines, 0, sizeof(rx_lines));
    memset((void *)rx_lengths, 0, sizeof(rx_lengths));
    rx_write_slot = 0u;
    rx_read_slot = 0u;
    rx_ready_count = 0u;
    rx_overflow = 0u;
}

void VisionJsonl_RxByte(uint8_t byte)
{
    uint16_t length;
    if (byte == '\r') return;
    if (byte == '\n')
    {
        if ((rx_ready_count < VISION_RX_SLOTS) &&
            (rx_overflow || (rx_lengths[rx_write_slot] != 0u)))
        {
            if (!rx_overflow)
                rx_lines[rx_write_slot][rx_lengths[rx_write_slot]] = '\0';
            else
                rx_lines[rx_write_slot][0] = '\0';
            ++rx_ready_count;
            rx_write_slot = (uint8_t)((rx_write_slot + 1u) % VISION_RX_SLOTS);
            rx_lengths[rx_write_slot] = 0u;
        }
        rx_overflow = 0u;
        return;
    }
    if (rx_overflow || (rx_ready_count >= VISION_RX_SLOTS)) return;
    length = rx_lengths[rx_write_slot];
    if (length + 1u < VISION_RX_LINE_MAX)
        rx_lines[rx_write_slot][rx_lengths[rx_write_slot]++] = (char)byte;
    else
        rx_overflow = 1u;
}

VisionJsonlResult_t VisionJsonl_ProcessWithRaw(VisionPlan_t *plan,
                                              char *status,
                                              size_t status_size,
                                              char *raw,
                                              size_t raw_size)
{
    VisionJsonlResult_t result;
    if ((plan == NULL) || (rx_ready_count == 0u)) return VISION_JSONL_NONE;
    if ((raw != NULL) && (raw_size != 0u))
    {
        strncpy(raw, rx_lines[rx_read_slot], raw_size - 1u);
        raw[raw_size - 1u] = '\0';
    }
    result = (rx_lines[rx_read_slot][0] == '\0') ?
             VISION_JSONL_REJECT :
             parse_line(rx_lines[rx_read_slot], plan, status, status_size);
    rx_lengths[rx_read_slot] = 0u;
    rx_read_slot = (uint8_t)((rx_read_slot + 1u) % VISION_RX_SLOTS);
    --rx_ready_count;
    return result;
}

VisionJsonlResult_t VisionJsonl_Process(VisionPlan_t *plan,
                                       char *status,
                                       size_t status_size)
{
    return VisionJsonl_ProcessWithRaw(plan, status, status_size, NULL, 0u);
}
