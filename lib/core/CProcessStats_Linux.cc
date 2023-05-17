/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <core/CProcessStats.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <fcntl.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

namespace ml {
namespace core {

namespace {
bool readFromSystemFile(const std::string& fileName, std::string& content) {
    char buffer[128] = {'\0'};

    // Use low level functions to read rather than C++ wrappers, as these are
    // system files.
    int fd = ::open(fileName.c_str(), O_RDONLY);
    if (fd == -1) {
        LOG_DEBUG(<< "Could not open " << fileName << ": " << ::strerror(errno));
        return false;
    }

    ssize_t bytesRead = ::read(fd, buffer, sizeof(buffer));
    ::close(fd);

    if (bytesRead < 0) {
        LOG_DEBUG(<< "Error reading from " << fileName << ": " << ::strerror(errno));
        return false;
    }

    if (bytesRead == 0) {
        LOG_DEBUG(<< "Read nothing from " << fileName);
        return false;
    }

    content.assign(buffer, 0, static_cast<size_t>(bytesRead));
    ml::core::CStringUtils::trimWhitespace(content);

    return true;
}
}

std::size_t CProcessStats::residentSetSize() {
    std::string statm;
    std::size_t rss;

    if (readFromSystemFile("/proc/self/statm", statm) == true) {
        std::vector<std::string> tokens;
        std::string remainder;
        core::CStringUtils::tokenise(" ", statm, tokens, remainder);

        if (tokens.size() < 2) {
            LOG_DEBUG(<< "unexpected output from /proc/self/statm, missing rss: " << statm);
            return 0;
        }
        core::CStringUtils::stringToTypeSilent(tokens[1], rss);
    }

    return 0;
}

std::size_t CProcessStats::maxResidentSetSize() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);

    // ru_maxrss is in kilobytes
    return static_cast<std::size_t>(rusage.ru_maxrss * 1024L);
}
}
}
