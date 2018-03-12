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
#ifndef INCLUDED_ml_domain_name_entropy_CIpAddressTest_h
#define INCLUDED_ml_domain_name_entropy_CIpAddressTest_h

#include <core/CRegex.h>

#include <string>


namespace ml {
namespace domain_name_entropy {


//! \brief
//! Test if a string is a valid ipv4 or ipv6 address
//!
//! DESCRIPTION:\n
//! Test if a string is a valid ipv4 or ipv6 address
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses regexes. Lexing and parsing would probably be faster.
//!
class CIpAddressTest {
    public:
        CIpAddressTest(void);

        bool isIpAddress(const std::string &) const;

    private:
        core::CRegex m_Ipv4Regex;
        core::CRegex m_Ipv6Regex;
};


}
}

#endif // INCLUDED_ml_domain_name_entropy_CIpAddressTest_h

