#include <assert.h>
#include <stdio.h>

#include "servo.h"

int main(void)
{
    assert(Servo_AngleToPulseUs(0u) == 500u);
    assert(Servo_AngleToPulseUs(90u) == 1500u);
    assert(Servo_AngleToPulseUs(180u) == 2500u);
    assert(Servo_AngleToPulseUs(255u) == 2500u);
    assert(Servo_ClampPulseUs(100u) == 500u);
    assert(Servo_ClampPulseUs(1700u) == 1700u);
    assert(Servo_ClampPulseUs(3000u) == 2500u);
    puts("servo: PASS");
    return 0;
}
