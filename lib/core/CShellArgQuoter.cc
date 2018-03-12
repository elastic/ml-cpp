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
#include <core/CShellArgQuoter.h>


namespace ml {
namespace core {


std::string CShellArgQuoter::quote(const std::string &arg) {
    if (arg.empty()) {
        return "''";
    }

    std::string result;

    // Reserve space for two escaped characters - if more are needed, the string
    // will just reallocate
    result.reserve(arg.length() + 6);

    // Use single quotes on Unix, as they prevent expansion of environment
    // variables
    bool insideSingleQuote(false);

    for (std::string::const_iterator iter = arg.begin();
            iter != arg.end();
            ++iter) {
        switch (*iter) {
            case '\'':
            case '!':
                // Take single quotes and exclamation marks outside of the main
                // single quoted string and escape them individually using
                // backslashes
                if (insideSingleQuote) {
                    result += '\'';
                    insideSingleQuote = false;
                }
                result += '\\';
                result += *iter;
                break;
            default:
                if (!insideSingleQuote) {
                    result += '\'';
                    insideSingleQuote = true;
                }
                result += *iter;
                break;
        }
    }

    if (insideSingleQuote) {
        result += '\'';
    }

    return result;
}


}
}

