#ifndef APP_VISION_JSONL_H
#define APP_VISION_JSONL_H

#include <stddef.h>
#include <stdint.h>

#define VISION_PLAN_MAX_COMMANDS 4u

typedef struct
{
    uint8_t piece_id;
    uint32_t pickup_x_pulse;
    uint32_t pickup_y_pulse;
    uint32_t place_x_pulse;
    uint32_t place_y_pulse;
    uint16_t rotation_deg; /* Positive camera angle, rounded to 0..360. */
} VisionCommand_t;

typedef struct
{
    uint32_t elapsed_ms;
    uint8_t count;
    VisionCommand_t commands[VISION_PLAN_MAX_COMMANDS];
} VisionPlan_t;

typedef enum
{
    VISION_JSONL_NONE = 0,
    VISION_JSONL_PLAN,
    VISION_JSONL_REPORT,
    VISION_JSONL_REJECT
} VisionJsonlResult_t;

void VisionJsonl_Init(void);
void VisionJsonl_RxByte(uint8_t byte);
VisionJsonlResult_t VisionJsonl_Process(VisionPlan_t *plan,
                                       char *status,
                                       size_t status_size);
VisionJsonlResult_t VisionJsonl_ProcessWithRaw(VisionPlan_t *plan,
                                              char *status,
                                              size_t status_size,
                                              char *raw,
                                              size_t raw_size);

#endif
