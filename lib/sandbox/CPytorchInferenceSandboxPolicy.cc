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
#include <sandbox/CPytorchInferenceSandboxPolicy.h>

#include <limits.h>
#include <sys/stat.h>

namespace ml {
namespace sandbox {

SArgDirExtraction extractArgDirs(const std::vector<std::string>& args) {
    SArgDirExtraction extraction;
    for (const auto& arg : args) {
        size_t eqPos = arg.find('=');
        if (eqPos == std::string::npos || eqPos + 1 >= arg.size()) {
            continue;
        }

        if (arg[eqPos + 1] != '/') {
            extraction.m_RejectedPipeArgs.push_back(arg + " (path not absolute)");
            continue;
        }

        std::string path = arg.substr(eqPos + 1);
        size_t lastSlash = path.rfind('/');
        if (lastSlash == 0) {
            extraction.m_RejectedPipeArgs.push_back(arg + " (no mountable directory)");
            continue;
        }

        std::string dir = path.substr(0, lastSlash);
        char resolved[PATH_MAX];
        std::string canonical = ::realpath(dir.c_str(), resolved) != nullptr ? resolved : dir;
        extraction.m_ArgDirs.insert(canonical);
        struct stat dirStat;
        if (dir != canonical && ::stat(dir.c_str(), &dirStat) == 0) {
            extraction.m_ArgDirs.insert(dir);
            extraction.m_PipeDirAliasMappings.push_back(dir + "->" + canonical);
        }
    }
    return extraction;
}

} // namespace sandbox
} // namespace ml
