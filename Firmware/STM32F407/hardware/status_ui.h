#ifndef STATUS_UI_H
#define STATUS_UI_H

#include "four_axis.h"

void StatusUi_Init(void);
void StatusUi_Process(FourAxisHomingState_t homing_state);
void StatusUi_LogKey(const char *key_name);
void StatusUi_ShowTransient(const char *message);

#endif
