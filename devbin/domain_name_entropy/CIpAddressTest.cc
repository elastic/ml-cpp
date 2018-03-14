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
#include "CIpAddressTest.h"

#include <core/CLogger.h>

namespace ml {
namespace domain_name_entropy {


CIpAddressTest::CIpAddressTest(void) {
    // https://gist.github.com/syzdek/6086792

    const std::string ipv4ReStr = "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])";

    std::string ipv6ReStr = "([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|"; // # TEST: 1:2:3:4:5:6:7:8
    ipv6ReStr += "([0-9a-fA-F]{1,4}:){1,7}:|";                           // # TEST: 1::                              1:2:3:4:5:6:7::
    ipv6ReStr += "([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|";           // # TEST: 1::8             1:2:3:4:5:6::8  1:2:3:4:5:6::8
    ipv6ReStr += "([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|";    // # TEST: 1::7:8           1:2:3:4:5::7:8  1:2:3:4:5::8
    ipv6ReStr += "([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|";    // # TEST: 1::6:7:8         1:2:3:4::6:7:8  1:2:3:4::8
    ipv6ReStr += "([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|";    // # TEST: 1::5:6:7:8       1:2:3::5:6:7:8  1:2:3::8
    ipv6ReStr += "([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|";    // # TEST: 1::4:5:6:7:8     1:2::4:5:6:7:8  1:2::8
    ipv6ReStr += "[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|";         // # TEST: 1::3:4:5:6:7:8   1::3:4:5:6:7:8  1::8
    ipv6ReStr += ":((:[0-9a-fA-F]{1,4}){1,7}|:)|";                       // # TEST: ::2:3:4:5:6:7:8  ::2:3:4:5:6:7:8 ::8       ::
    ipv6ReStr += "fe08:(:[0-9a-fA-F]{1,4}){2,2}%[0-9a-zA-Z]{1,}|";       // # TEST: fe08::7:8%eth0      fe08::7:8%1                                      (link-local IPv6 addresses with zone index)
    ipv6ReStr +="::(ffff(:0{1,4}){0,1}:){0,1}" + ipv4ReStr + "|";        // # TEST: ::255.255.255.255   ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)
    ipv6ReStr +="([0-9a-fA-F]{1,4}:){1,4}:" + ipv4ReStr;                 // # TEST: 2001:db8:3:4::192.0.2.33  64:ff9b::192.0.2.33                        (IPv4-Embedded IPv6 Address)

    if (m_Ipv4Regex.init(ipv4ReStr) == false) {
        LOG_ERROR("Can not init regex :" << ipv4ReStr);
    }

    if (m_Ipv6Regex.init(ipv6ReStr) == false) {
        LOG_ERROR("Can not init regex :" << ipv6ReStr);
    }
}

bool CIpAddressTest::isIpAddress(const std::string &str) const {
    if (m_Ipv4Regex.matches(str)) {
        return true;
    }

    if (m_Ipv6Regex.matches(str)) {
        return true;
    }

    return false;
}


}
}
