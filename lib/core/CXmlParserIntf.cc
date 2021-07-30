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
#include <core/CXmlParserIntf.h>

#include <core/CStringUtils.h>

#include <ctype.h>

namespace ml {
namespace core {

const std::string CXmlParserIntf::XML_HEADER("<?xml version=\"1.0\"?>");

CXmlParserIntf::CXmlParserIntf() {
}

CXmlParserIntf::~CXmlParserIntf() {
}

std::string CXmlParserIntf::makeValidName(const std::string& str) {
    std::string result(str);

    if (!result.empty()) {
        // First character can't be a number
        if (!::isalpha(static_cast<unsigned char>(result[0]))) {
            result[0] = '_';
        }

        // Other characters can be numbers, but change all other punctuation to
        // underscores
        for (std::string::iterator iter = result.begin() + 1; iter != result.end(); ++iter) {
            if (!::isalnum(static_cast<unsigned char>(*iter))) {
                *iter = '_';
            }
        }
    }

    return result;
}

std::string CXmlParserIntf::toOneLine(const std::string& xml) {
    std::string oneLine(xml);

    CStringUtils::replace(XML_HEADER, "", oneLine);
    CStringUtils::trimWhitespace(oneLine);
    CStringUtils::replace("\r", "&#xD;", oneLine);
    CStringUtils::replace("\n", "&#xA;", oneLine);

    return oneLine;
}
}
}
