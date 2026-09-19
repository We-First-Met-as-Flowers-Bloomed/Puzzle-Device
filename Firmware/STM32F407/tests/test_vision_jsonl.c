#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vision_jsonl.h"

static VisionJsonlResult_t feed(const char *line, VisionPlan_t *plan)
{
    const char *p;
    char status[32];
    for (p = line; *p != '\0'; ++p) VisionJsonl_RxByte((uint8_t)*p);
    return VisionJsonl_Process(plan, status, sizeof(status));
}

int main(void)
{
    VisionPlan_t plan;
    char status[32];
    char raw[1024];
    const char *valid =
        "{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":6000,"
        "\"commands\":[{\"piece_id\":0,\"pickup_x_pulse\":24000,"
        "\"pickup_y_pulse\":16700,\"place_x_pulse\":0,\"place_y_pulse\":0,"
        "\"rotation_deg\":359.4829862640287}]}\n";
    const char *two =
        "{\"commands\":["
        "{\"rotation_deg\":30.0,\"place_y_pulse\":12500,\"piece_id\":0,"
        "\"pickup_y_pulse\":4200,\"place_x_pulse\":10600,\"pickup_x_pulse\":8000},"
        "{\"piece_id\":1,\"pickup_x_pulse\":15000,\"pickup_y_pulse\":5000,"
        "\"place_x_pulse\":16800,\"place_y_pulse\":13200,\"rotation_deg\":179.9}],"
        "\"elapsed_ms\":12000,\"robot_calibrated\":true,\"status\":\"OK\"}\n";

    VisionJsonl_Init();
    {
        const char *p;
        for (p = valid; *p != '\0'; ++p) VisionJsonl_RxByte((uint8_t)*p);
        assert(VisionJsonl_ProcessWithRaw(&plan, status, sizeof(status),
                                         raw, sizeof(raw)) == VISION_JSONL_PLAN);
        assert(strcmp(raw,
          "{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":6000,"
          "\"commands\":[{\"piece_id\":0,\"pickup_x_pulse\":24000,"
          "\"pickup_y_pulse\":16700,\"place_x_pulse\":0,\"place_y_pulse\":0,"
          "\"rotation_deg\":359.4829862640287}]}") == 0);
    }
    assert(plan.count == 1u && plan.elapsed_ms == 6000u);
    assert(plan.commands[0].piece_id == 0u);
    assert(plan.commands[0].pickup_x_pulse == 24000u);
    assert(plan.commands[0].pickup_y_pulse == 16700u);
    assert(plan.commands[0].rotation_deg == 359);
    assert(feed("{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":0,"
                "\"commands\":[{\"piece_id\":3,\"pickup_x_pulse\":0,"
                "\"pickup_y_pulse\":0,\"place_x_pulse\":0,\"place_y_pulse\":0,"
                "\"rotation_deg\":272.32294496804366}]}\n", &plan) == VISION_JSONL_PLAN);
    assert(plan.commands[0].rotation_deg == 272);
    assert(feed("{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":0,"
                "\"commands\":[{\"piece_id\":3,\"pickup_x_pulse\":8518,"
                "\"pickup_y_pulse\":7205,\"place_x_pulse\":16939,\"place_y_pulse\":8894,"
                "\"rotation_deg\":345.87339701406665}]}\n", &plan) == VISION_JSONL_PLAN);
    assert(plan.commands[0].rotation_deg == 346u);
    assert(feed("{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":0,"
                "\"commands\":[{\"piece_id\":2,\"pickup_x_pulse\":0,"
                "\"pickup_y_pulse\":0,\"place_x_pulse\":0,\"place_y_pulse\":0,"
                "\"rotation_deg\":-1}]}\n", &plan) == VISION_JSONL_REJECT);

    assert(feed(two, &plan) == VISION_JSONL_PLAN);
    assert(plan.count == 2u);
    assert(plan.commands[1].place_x_pulse == 16800u);
    assert(plan.commands[1].rotation_deg == 180);

    assert(feed("{\"status\":\"UART_TEST\",\"robot_calibrated\":false,"
                "\"elapsed_ms\":0,\"commands\":[]}\n", &plan) ==
           VISION_JSONL_REPORT);
    assert(feed("{\"status\":\"LOW_MARGIN\",\"robot_calibrated\":true,"
                "\"elapsed_ms\":1,\"commands\":[]}\n", &plan) ==
           VISION_JSONL_REPORT);
    assert(feed("{\"status\":\"ERROR\",\"type\":\"DetectionError\","
                "\"message\":\"camera failed\"}\n", &plan) ==
           VISION_JSONL_REPORT);
    assert(feed("{\"status\":\"OK\",\"robot_calibrated\":false,"
                "\"elapsed_ms\":0,\"commands\":[]}\n", &plan) ==
           VISION_JSONL_REJECT);
    assert(feed("{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":0,"
                "\"commands\":[{\"piece_id\":0,\"pickup_x_pulse\":1,"
                "\"pickup_y_pulse\":1,\"place_x_pulse\":1,\"place_y_pulse\":1,"
                "\"rotation_deg\":0},{\"piece_id\":0,\"pickup_x_pulse\":2,"
                "\"pickup_y_pulse\":2,\"place_x_pulse\":2,\"place_y_pulse\":2,"
                "\"rotation_deg\":0}]}\n", &plan) == VISION_JSONL_REJECT);
    assert(feed("{\"status\":\"OK\",\"robot_calibrated\":true,\"elapsed_ms\":0,"
                "\"commands\":[{\"piece_id\":0,\"pickup_x_pulse\":24001,"
                "\"pickup_y_pulse\":1,\"place_x_pulse\":1,\"place_y_pulse\":1,"
                "\"rotation_deg\":0}]}\n", &plan) == VISION_JSONL_REJECT);
    assert(feed("{bad json}\n", &plan) == VISION_JSONL_REJECT);

    VisionJsonl_Init();
    {
        unsigned i;
        for (i = 0u; i < 1100u; ++i) VisionJsonl_RxByte('x');
        VisionJsonl_RxByte('\n');
    }
    assert(VisionJsonl_Process(&plan, status, sizeof(status)) == VISION_JSONL_REJECT);
    assert(feed(valid, &plan) == VISION_JSONL_PLAN);

    puts("vision_jsonl: PASS");
    return 0;
}
