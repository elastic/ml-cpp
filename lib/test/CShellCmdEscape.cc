/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CShellCmdEscape.h>

#include <core/CStringUtils.h>

namespace ml {
namespace test {

void CShellCmdEscape::escapeCmd(std::string& cmd) {
    // Special characters are \ * ? < > # & | ( ) ' " ` ;
    // Escape character is \ on Unix

    core::CStringUtils::replace("\\", "\\\\", cmd);
    core::CStringUtils::replace("*", "\\*", cmd);
    core::CStringUtils::replace("?", "\\?", cmd);
    core::CStringUtils::replace("<", "\\<", cmd);
    core::CStringUtils::replace(">", "\\>", cmd);
    core::CStringUtils::replace("#", "\\#", cmd);
    core::CStringUtils::replace("&", "\\&", cmd);
    core::CStringUtils::replace("|", "\\|", cmd);
    core::CStringUtils::replace("(", "\\(", cmd);
    core::CStringUtils::replace(")", "\\)", cmd);
    core::CStringUtils::replace("'", "\\'", cmd);
    core::CStringUtils::replace("\"", "\\\"", cmd);
    core::CStringUtils::replace("`", "\\`", cmd);
    core::CStringUtils::replace(";", "\\;", cmd);
}
}
}
