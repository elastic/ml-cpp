/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CProcessPriority.h>

#include <core/CLogger.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

namespace ml {
namespace core {

namespace {

bool writeToSystemFile(const std::string& fileName, const std::string& value) {
    // Use low level functions to write rather than C++ wrappers, as these are
    // system files.
    int fd = ::open(fileName.c_str(), O_WRONLY);
    if (fd == -1) {
        return false;
    }

    if (::write(fd, value.c_str(), value.length()) < static_cast<ssize_t>(value.length())) {
        ::close(fd);
        return false;
    }

    LOG_DEBUG(<< "Successfully increased OOM killer adjustment via " << fileName);

    ::close(fd);

    return true;
}

void increaseOomKillerAdj() {
    // oom_score_adj is supported by newer kernels and oom_adj by older kernels.
    // oom_score_adj is on a scale of -1000 to 1000.
    // oom_adj is on a scale of -16 to 15.
    // In both cases higher numbers mean the process is more likely to be killed
    // in low memory situations.
    if (writeToSystemFile("/proc/self/oom_score_adj", "667\n") == false && writeToSystemFile("/proc/self/oom_adj", "10\n") == false) {
        LOG_WARN(<< "Could not increase OOM killer adjustment using "
                    "/proc/self/oom_score_adj or /proc/self/oom_adj: "
                 << ::strerror(errno));
    }
}
}

void CProcessPriority::reducePriority() {
    // Currently the only action is to increase the OOM killer adjustment, but
    // there could be others in the future.
    increaseOomKillerAdj();
}
}
}
