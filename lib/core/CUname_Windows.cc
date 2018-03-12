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
#include <core/CUname.h>

#include <core/CLogger.h>
#include <core/CWindowsError.h>
#include <core/WindowsSafe.h>

#include <boost/scoped_array.hpp>

#include <sstream>
#include <vector>


namespace ml {
namespace core {
namespace detail {


bool queryKernelVersion(uint16_t &major, uint16_t &minor, uint16_t &build) {
    // This used to be done with GetVersionEx(), but that no longer works
    // starting with Windows 8.1/Windows Server 2012r2.  Instead we get the
    // true OS version by looking at the product version for kernel32.dll, and
    // then distinguish client/server versions of Windows using
    // VerifyVersionInfo().

    static const char *KERNEL32_DLL("kernel32.dll");

    DWORD handle(0);
    DWORD size(GetFileVersionInfoSize(KERNEL32_DLL, &handle));
    if (size == 0) {
        LOG_ERROR("Error getting file version info size for " << KERNEL32_DLL <<
                  " - error code : " << CWindowsError());
        return false;
    }

    typedef boost::scoped_array<char> TScopedCharArray;
    TScopedCharArray buffer(new char[size]);
    if (GetFileVersionInfo(KERNEL32_DLL, handle, size, buffer.get()) == FALSE) {
        LOG_ERROR("Error getting file version info for " << KERNEL32_DLL <<
                  " - error code : " << CWindowsError());
        return false;
    }

    UINT len(0);
    VS_FIXEDFILEINFO *fixedFileInfo(0);
    if (VerQueryValue(buffer.get(),
                      "\\",
                      reinterpret_cast<void **>(&fixedFileInfo),
                      &len) == FALSE) {
        LOG_ERROR("Error querying fixed file info for " << KERNEL32_DLL <<
                  " - error code : " << CWindowsError());
        return false;
    }

    if (len < sizeof(VS_FIXEDFILEINFO)) {
        LOG_ERROR("Too little data returned for VS_FIXEDFILEINFO - " <<
                  "expected " << sizeof(VS_FIXEDFILEINFO) << " bytes, got " <<
                  len);
        return false;
    }

    major = HIWORD(fixedFileInfo->dwProductVersionMS);
    minor = LOWORD(fixedFileInfo->dwProductVersionMS);
    build = HIWORD(fixedFileInfo->dwProductVersionLS);

    return true;
}


}


std::string CUname::sysName(void) {
    return "Windows";
}

std::string CUname::nodeName(void) {
    // First ask with a size of zero to find the required size
    DWORD size(0);
    BOOL res(GetComputerNameEx(ComputerNameDnsHostname, 0, &size));
    if (res != FALSE || GetLastError() != ERROR_MORE_DATA) {
        LOG_ERROR("Error getting computer name length - error code : " <<
                  CWindowsError());
        return std::string();
    }

    typedef std::vector<char> TCharVec;
    TCharVec buffer(size);

    res = GetComputerNameEx(ComputerNameDnsHostname,
                            &buffer[0],
                            &size);
    if (res == FALSE) {
        LOG_ERROR("Error getting computer name - error code : " <<
                  CWindowsError());
        return std::string();
    }

    return std::string(buffer.begin(), buffer.begin() + size);
}

std::string CUname::release(void) {
    uint16_t major(0);
    uint16_t minor(0);
    uint16_t build(0);
    if (detail::queryKernelVersion(major, minor, build) == false) {
        // Error logging done in the helper function
        return std::string();
    }

    std::ostringstream strm;
    strm << major << '.' << minor;

    return strm.str();
}

std::string CUname::version(void) {
    uint16_t major(0);
    uint16_t minor(0);
    uint16_t build(0);
    if (detail::queryKernelVersion(major, minor, build) == false) {
        // Error logging done in the helper function
        return std::string();
    }

    std::ostringstream strm;
    strm << "Windows NT Version " << major << '.' << minor;

    // Client and server builds of Windows share the same version numbers, so
    // determine whether this is a client or server
    OSVERSIONINFOEX versionInfoEx = { 0 };
    versionInfoEx.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

    DWORDLONG conditionMask(0);
    versionInfoEx.wProductType = VER_NT_DOMAIN_CONTROLLER;
    if (VerifyVersionInfo(&versionInfoEx,
                          VER_PRODUCT_TYPE,
                          VerSetConditionMask(conditionMask,
                                              VER_PRODUCT_TYPE,
                                              VER_EQUAL)) != FALSE) {
        strm << " (Domain Controller)";
    } else {
        conditionMask = 0;
        versionInfoEx.wProductType = VER_NT_SERVER;
        if (VerifyVersionInfo(&versionInfoEx,
                              VER_PRODUCT_TYPE,
                              VerSetConditionMask(conditionMask,
                                                  VER_PRODUCT_TYPE,
                                                  VER_EQUAL)) != FALSE) {
            strm << " (Server)";
        } else {
            conditionMask = 0;
            versionInfoEx.wProductType = VER_NT_WORKSTATION;
            if (VerifyVersionInfo(&versionInfoEx,
                                  VER_PRODUCT_TYPE,
                                  VerSetConditionMask(conditionMask,
                                                      VER_PRODUCT_TYPE,
                                                      VER_EQUAL)) != FALSE) {
                strm << " (Workstation)";
            }
        }
    }

    strm << " Build " << build;

    return strm.str();
}

std::string CUname::machine(void) {
    SYSTEM_INFO systemInfo;
    GetNativeSystemInfo(&systemInfo);

    std::string result;

    switch (systemInfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            result = "x64";
            break;
        case PROCESSOR_ARCHITECTURE_IA64:
            result = "itanium";
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            result = "x86";
            break;
        case PROCESSOR_ARCHITECTURE_UNKNOWN:
            result = "unknown";
            break;
        default:
            LOG_ERROR("Unexpected result from GetNativeSystemInfo() : "
                      "wProcessorArchitecture = " <<
                      systemInfo.wProcessorArchitecture);
            break;
    }

    return result;
}

std::string CUname::all(void) {
    // This is in the format of "uname -a"
    std::string all(CUname::sysName());
    all += ' ';
    all += CUname::nodeName();
    all += ' ';
    all += CUname::release();
    all += ' ';
    all += CUname::version();
    all += ' ';
    all += CUname::machine();

    return all;
}

std::string CUname::mlPlatform(void) {
    // Determine the current platform name, in the format used by Kibana
    // downloads.  For Windows this is either "windows-x86" or "windows-x86_64".

    if (sizeof(void *) == 8) {
        return "windows-x86_64";
    }
    return "windows-x86";
}

std::string CUname::mlOsVer(void) {
    uint16_t major(0);
    uint16_t minor(0);
    uint16_t build(0);
    if (detail::queryKernelVersion(major, minor, build) == false) {
        // Error logging done in the helper function
        return std::string();
    }

    std::ostringstream strm;
    strm << major << '.' << minor << '.' << build;

    return strm.str();
}


}
}

