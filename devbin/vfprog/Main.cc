/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CMonotonicTime.h>

#include <vflib/CIncrementer.h>
#include <vflib/CLooper.h>

#include "CIncrementer.h"
#include "CLooper.h"

#include <iostream>
#include <utility>

#include <stdlib.h>
#include <string.h>


namespace
{

const size_t WARMUP_COUNT(100);
const size_t TEST_COUNT(1000000000);

typedef std::pair<size_t, uint64_t> TSizeUInt64Pr;

size_t benchmark(char testId,
                 ml::vfprog::CIncrementer &incrementer,
                 size_t count)
{
    size_t val(0);
    switch (testId)
    {
        case '1':
            val = ml::vfprog::CLooper::inlinedProgramCallLoop(incrementer,
                                                                   count,
                                                                   val);
            break;
        case '2':
            val = ml::vfprog::CLooper::nonVirtualProgramCallLoop(incrementer,
                                                                      count,
                                                                      val);
            break;
        case '3':
            val = ml::vfprog::CLooper::virtualProgramCallLoop(incrementer,
                                                                   count,
                                                                   val);
            break;
    }

    return val;
}

size_t benchmark(char testId,
                 ml::vflib::CIncrementer &incrementer,
                 size_t count)
{
    size_t val(0);
    switch (testId)
    {
        case '4':
            val = ml::vfprog::CLooper::inlinedLibraryCallLoop(incrementer,
                                                                   count,
                                                                   val);
            break;
        case '5':
            val = ml::vfprog::CLooper::nonVirtualLibraryCallLoop(incrementer,
                                                                      count,
                                                                      val);
            break;
        case '6':
            val = ml::vfprog::CLooper::virtualLibraryCallLoop(incrementer,
                                                                   count,
                                                                   val);
            break;
        case '7':
            val = ml::vflib::CLooper::inlinedLibraryCallLoop(incrementer,
                                                                  count,
                                                                  val);
            break;
        case '8':
            val = ml::vflib::CLooper::nonVirtualLibraryCallLoop(incrementer,
                                                                     count,
                                                                     val);
            break;
        case '9':
            val = ml::vflib::CLooper::virtualLibraryCallLoop(incrementer,
                                                                  count,
                                                                  val);
            break;
    }

    return val;
}

template <typename INCREMENTER>
TSizeUInt64Pr benchmark(char testId,
                        INCREMENTER &incrementer)
{
    ml::core::CMonotonicTime clock;
    TSizeUInt64Pr result;

    // Do a few calls without measurement to warm up the CPU cache
    benchmark(testId, incrementer, WARMUP_COUNT);

    // Now time the full quota of calls
    uint64_t startTime(clock.nanoseconds());
    result.first = benchmark(testId, incrementer, TEST_COUNT);
    uint64_t endTime(clock.nanoseconds());
    result.second = endTime - startTime;

    return result;
}

TSizeUInt64Pr benchmark(char testId)
{
    if (testId < '4')
    {
        ml::vfprog::CIncrementer incrementer;
        return benchmark(testId, incrementer);
    }

    ml::vflib::CIncrementer incrementer;
    return benchmark(testId, incrementer);
}

}


int main(int argc, char **argv)
{
    if (argc != 2 ||
        ::strlen(argv[1]) != 1 ||
        argv[1][0] < '1' ||
        argv[1][0] > '9')
    {
        std::cerr << "Usage: " << argv[0] << " <1-9>" << std::endl;
        return EXIT_FAILURE;
    }

    TSizeUInt64Pr result(benchmark(argv[1][0]));

    std::cout << "Final value: " << result.first << std::endl;
    std::cout << "Run time: " << result.second << " nanoseconds" << std::endl;

    return EXIT_SUCCESS;
}

