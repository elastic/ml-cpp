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

#include <boost/filesystem.hpp>

#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdlib.h>


namespace ml
{
namespace core
{


std::string CProgName::progName()
{
    const char *progName(::getprogname());
    if (progName == 0)
    {
        return std::string();
    }

    return progName;
}

std::string CProgName::progDir()
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

