/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CUname.h>

#include <core/CStringUtils.h>

#include <sys/utsname.h>
#include <unistd.h>

#include <string.h>

namespace ml {
namespace core {

std::string CUname::sysName() {
    struct utsname name;
    ::uname(&name);

    return name.sysname;
}

std::string CUname::nodeName() {
    struct utsname name;
    ::uname(&name);

    return name.nodename;
}

std::string CUname::release() {
    struct utsname name;
    ::uname(&name);

    return name.release;
}

std::string CUname::version() {
    struct utsname name;
    ::uname(&name);

    return name.version;
}

std::string CUname::machine() {
    struct utsname name;
    ::uname(&name);

    return name.machine;
}

std::string CUname::all() {
    struct utsname name;
    ::uname(&name);

    // This is in the format of "uname -a"
    std::string all(name.sysname);
    all += ' ';
    all += name.nodename;
    all += ' ';
    all += name.release;
    all += ' ';
    all += name.version;
    all += ' ';
    all += name.machine;

    return all;
}

std::string CUname::mlPlatform() {
    struct utsname name;
    ::uname(&name);

    // Determine the current platform name, in the format used by Kibana
    // downloads.  For *nix platforms this is more-or-less `uname -s`-`uname -m`
    // converted to lower case.  However, for consistency between different
    // operating systems on the same architecture "amd64"/"i86pc" are replaced
    // with "x86_64", "i386"/"i686" with "x86" and "arm64" with "aarch64".
    //
    // Assuming the current platform is supported this name will be one of:
    // - darwin-aarch64
    // - darwin-x86_64
    // - linux-aarch64
    // - linux-x86_64
    //
    // For an unsupported platform it will be something different, but hopefully
    // still recognisable.

    std::string os(CStringUtils::toLower(name.sysname));

    const std::string& machine = CStringUtils::toLower(name.machine);
    if (machine.length() == 4 && machine[0] == 'i' && machine[2] == '8' &&
        machine[3] == '6') {
        return os + "-x86";
    }

    if (machine == "amd64" || machine == "i86pc") {
        return os + "-x86_64";
    }

    if (machine == "arm64") {
        return os + "-aarch64";
    }

    return os + '-' + machine;
}

std::string CUname::mlOsVer() {
    return CUname::release();
}
}
}
