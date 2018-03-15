/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifdef NO_STD_CHRONO
#include <core/CMonotonicTime.h>
#else
#include <chrono>
#endif
#include <iostream>
#include <string>
#include <utility>

#include <stdlib.h>

#ifdef _MSC_VER
#define DONT_INLINE_THIS_FUNCTION __declspec(noinline)
#else
#define DONT_INLINE_THIS_FUNCTION __attribute__((__noinline__))
#endif

std::string s0;
std::string s1;
std::string s2;
std::string s3;
size_t totalLength = 0;

template<char OP>
void transfer(std::string&& from, std::string& to) {
    if (OP == 'm') {
        to = std::move(from);
    } else if (OP == 'c') {
        to = from;
    } else if (OP == 'd') {
        to.assign(from, 0, from.length());
    } else {
        from.swap(to);
    }
}

template<char OP>
DONT_INLINE_THIS_FUNCTION void func3(std::string&& s) {
    transfer<OP>(std::move(s), s3);
    s3[0] = '3';
    totalLength += s3.length();
}

template<char OP>
DONT_INLINE_THIS_FUNCTION void func2(std::string&& s) {
    transfer<OP>(std::move(s), s2);
    s2[0] = '2';
    func3<OP>(std::move(s2));
}

template<char OP>
DONT_INLINE_THIS_FUNCTION void func1(std::string&& s) {
    transfer<OP>(std::move(s), s1);
    s1[0] = '1';
    func2<OP>(std::move(s1));
}

template<char OP>
void generate(size_t minSize, size_t iterations) {
    for (size_t count = 0; count < iterations; ++count) {
        s0.assign(minSize + count % 15, char('A' + count % 26));
        func1<OP>(std::move(s0));
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m|c|d|s> <min size> <iterations>" << std::endl
                  << "Where: m = move" << std::endl
                  << "       c = copy" << std::endl
                  << "       d = copy defeating copy-on-write" << std::endl
                  << "       s = swap" << std::endl;
        return EXIT_FAILURE;
    }

    size_t minSize = static_cast<size_t>(atoi(argv[2]));
    size_t iterations = static_cast<size_t>(atoi(argv[3]));

#ifdef NO_STD_CHRONO
    ml::core::CMonotonicTime clock;
    uint64_t startTimeNs = clock.nanoseconds();
#else
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
#endif

    if (argv[1][0] == 'm') {
        generate<'m'>(minSize, iterations);
    } else if (argv[1][0] == 'c') {
        generate<'c'>(minSize, iterations);
    } else if (argv[1][0] == 'd') {
        generate<'d'>(minSize, iterations);
    } else {
        generate<'s'>(minSize, iterations);
    }

#ifdef NO_STD_CHRONO
    uint64_t endTimeNs = clock.nanoseconds();
    uint64_t durationTenthMs = (endTimeNs - startTimeNs) / 100000;
    uint64_t durationMs = (durationTenthMs / 10) + ((durationTenthMs % 10 >= 5) ? 1 : 0);
#else
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    size_t durationMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
#endif

    std::cout << "Time " << durationMs
              << "ms, "
                 "Total length: "
              << totalLength << std::endl;

    return EXIT_SUCCESS;
}
