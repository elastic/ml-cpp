/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <ctime>
#include <iostream>
#include <vector>

int main(int, char**) {
    const std::time_t startTime = 1346968800;
    const std::time_t bucketLength = 3600;

    std::vector<std::time_t> eventTimes;
    eventTimes.push_back(1346968927);
    eventTimes.push_back(1346969039);
    eventTimes.push_back(1347019162);

    std::time_t endTime = (eventTimes.back() / bucketLength + 1) * bucketLength;
    std::cout << "startTime = " << startTime << ", endTime = " << endTime
              << ", # events = " << eventTimes.size() << std::endl;

    {
        std::time_t offset = endTime - startTime;
        unsigned long i = 0;
        for (std::time_t bucketStartTime = startTime; bucketStartTime < endTime;
             bucketStartTime += bucketLength) {
            std::time_t bucketEndTime = bucketStartTime + bucketLength;
            for (; i < eventTimes.size() && eventTimes[i] < bucketEndTime; ++i) {
                std::vector<std::time_t> temp;
                // Comment out the next line to get the correct result
                temp.push_back(eventTimes[i] + offset);
            }
        }
    }

    std::time_t offset = 2 * (endTime - startTime);

    // Both lines of output should read 100800
    // With "Microsoft (R) C/C++ Optimizing Compiler Version 18.00.21005.1 for x64" the first prints 4295068096
    std::cout << offset << std::endl;
    std::cout << (offset & 0xFFFFFFFF) << std::endl;

    return 0;
}
