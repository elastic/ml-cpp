/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CProgName.h>

#include <boost/filesystem.hpp>

#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdlib.h>


namespace ml
{
namespace core
{


std::string CProgName::progName(void)
{
    const char *progName(::getprogname());
    if (progName == 0)
    {
        return std::string();
    }

    return progName;
}

std::string CProgName::progDir(void)
{
    uint32_t bufferSize(2048);
    std::string path(bufferSize, '\0');
    if (_NSGetExecutablePath(&path[0], &bufferSize) != 0)
    {
        return std::string();
    }
    size_t lastSlash(path.rfind('/'));
    if (lastSlash == std::string::npos)
    {
        return std::string();
    }
    path.resize(lastSlash);

    // On Mac OS X the path returned from _NSGetExecutablePath() is not always
    // canonical, e.g. containing /./
    return boost::filesystem::canonical(boost::filesystem::path(path)).string();
}


}
}

