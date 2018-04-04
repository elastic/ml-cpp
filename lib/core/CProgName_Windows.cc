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
#include <core/CProgName.h>

#include <core/WindowsSafe.h>

#include <algorithm>

#include <string.h>

namespace ml {
namespace core {

std::string CProgName::progName() {
    static const size_t BUFFER_SIZE(2048);
    char buffer[BUFFER_SIZE] = {'\0'};
    if (GetModuleFileName(0, buffer, BUFFER_SIZE - 1) == FALSE) {
        return std::string();
    }

    // Always return the long file name of the program, even if it was invoked
    // using the 8.3 name
    char longPathBuffer[BUFFER_SIZE] = {'\0'};
    if (GetLongPathName(buffer, longPathBuffer, BUFFER_SIZE - 1) == FALSE) {
        return std::string();
    }

    char* progName(longPathBuffer);

    // Strip the path
    char* lastSlash(std::max(::strrchr(progName, '/'), ::strrchr(progName, '\\')));
    if (lastSlash != 0) {
        progName = lastSlash + 1;
    }

    // Strip the extension
    char* lastDot(::strrchr(progName, '.'));
    if (lastDot != 0) {
        *lastDot = '\0';
    }

    return progName;
}

std::string CProgName::progDir() {
    static const size_t BUFFER_SIZE(2048);
    std::string path(BUFFER_SIZE, '\0');
    if (GetModuleFileName(0, &path[0], BUFFER_SIZE) == FALSE) {
        return std::string();
    }
    size_t lastSlash(path.find_last_of("\\/"));
    if (lastSlash == std::string::npos) {
        return std::string();
    }
    path.resize(lastSlash);

    // We are using the ANSI versions of Windows API functions, which don't
    // support extended paths, so strip any leading extended length indicator.
    // (We have to accept that if the path is more than 260 characters long
    // after doing this then the program won't work.)
    if (path.compare(0, 4, "\\\\?\\") == 0) {
        path.erase(0, 4);
    }

    return path;
}
}
}
