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
#include <core/CXmlParserIntf.h>

#include <core/CStringUtils.h>

#include <ctype.h>


namespace ml {
namespace core {


const std::string CXmlParserIntf::XML_HEADER("<?xml version=\"1.0\"?>");


CXmlParserIntf::CXmlParserIntf(void) {}

CXmlParserIntf::~CXmlParserIntf(void) {}

std::string CXmlParserIntf::makeValidName(const std::string &str) {
    std::string result(str);

    if (!result.empty()) {
        // First character can't be a number
        if (!::isalpha(static_cast<unsigned char>(result[0]))) {
            result[0] = '_';
        }

        // Other characters can be numbers, but change all other punctuation to
        // underscores
        for (std::string::iterator iter = result.begin() + 1;
             iter != result.end();
             ++iter) {
            if (!::isalnum(static_cast<unsigned char>(*iter))) {
                *iter = '_';
            }
        }
    }

    return result;
}

std::string CXmlParserIntf::toOneLine(const std::string &xml) {
    std::string oneLine(xml);

    CStringUtils::replace(XML_HEADER, "", oneLine);
    CStringUtils::trimWhitespace(oneLine);
    CStringUtils::replace("\r", "&#xD;", oneLine);
    CStringUtils::replace("\n", "&#xA;", oneLine);

    return oneLine;
}


}
}

