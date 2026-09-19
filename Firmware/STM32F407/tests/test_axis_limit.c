#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "axis_limit.h"

int main(void)
{
    AxisLimit_t z;
    AxisLimit_t y;
    AxisLimit_t x;

    AxisLimit_Init(&x, 29500u);
    assert(AxisLimit_Allow(&x, 1u, 30000u) == 29500u);
    AxisLimit_Commit(&x, 1u, 29500u);
    assert(AxisLimit_GetPosition(&x) == 29500u);
    assert(AxisLimit_Allow(&x, 1u, 1u) == 0u);
    assert(AxisLimit_Allow(&x, 0u, 30000u) == 29500u);

    AxisLimit_Init(&z, 300u);
    assert(AxisLimit_GetPosition(&z) == 0u);
    assert(AxisLimit_Allow(&z, 1u, 250u) == 250u);
    AxisLimit_Commit(&z, 1u, 250u);
    assert(AxisLimit_GetPosition(&z) == 250u);
    assert(AxisLimit_Allow(&z, 1u, 300u) == 50u);
    AxisLimit_Commit(&z, 1u, 50u);
    assert(AxisLimit_GetPosition(&z) == 300u);
    assert(AxisLimit_Allow(&z, 1u, 1u) == 0u);
    assert(AxisLimit_Allow(&z, 0u, 400u) == 300u);
    AxisLimit_Commit(&z, 0u, 300u);
    assert(AxisLimit_GetPosition(&z) == 0u);
    assert(AxisLimit_Allow(&z, 0u, 1u) == 0u);

    AxisLimit_Init(&y, 16800u);
    assert(AxisLimit_Allow(&y, 1u, 16000u) == 16000u);
    AxisLimit_Commit(&y, 1u, 16000u);
    assert(AxisLimit_Allow(&y, 1u, 3200u) == 800u);
    AxisLimit_Commit(&y, 1u, 800u);
    assert(AxisLimit_GetPosition(&y) == 16800u);
    assert(AxisLimit_Allow(&y, 0u, 30000u) == 16800u);

    puts("axis_limit: PASS");
    return 0;
}
