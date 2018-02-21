/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CSleep.h>

#include <core/WindowsSafe.h>


namespace ml
{
namespace core
{


// Default processing delay is 100 milliseconds
const uint32_t CSleep::DEFAULT_PROCESSING_DELAY(100);


void CSleep::sleep(uint32_t milliseconds)
{
    if (milliseconds > 0)
    {
        Sleep(milliseconds);
    }
}

void CSleep::delayProcessing(void)
{
    // 0.1 seconds is a good length of time to delay processing.
    CSleep::sleep(DEFAULT_PROCESSING_DELAY);
}


}
}

