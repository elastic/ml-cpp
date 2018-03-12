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
#include <core/CRegex.h>

#include <core/CLogger.h>

#include <limits>


namespace {

const char *translateErrorCode(boost::regex_constants::error_type code) {
    // From boost_1_47_0/libs/regex/doc/html/boost_regex/ref/error_type.html
    // and /usr/local/include/boost-1_47/boost/regex/v4/error_type.hpp.  The
    // switch cases are in the same order as the enum definition in
    // /usr/local/include/boost-1_47/boost/regex/v4/error_type.hpp to make it
    // easier to add new cases in future versions of Boost.  The -Wswitch-enum
    // option to g++ should warn if future versions of Boost introduce new enum
    // values.
    switch (code) {
        case boost::regex_constants::error_ok:
            return "No error."; // Not used in Boost 1.47
        case boost::regex_constants::error_no_match:
            return "No match."; // Not used in Boost 1.47
        case boost::regex_constants::error_bad_pattern:
            return "Other unspecified errors.";
        case boost::regex_constants::error_collate:
            return "An invalid collating element was specified in a [[.name.]] block.";
        case boost::regex_constants::error_ctype:
            return "An invalid character class name was specified in a [[:name:]] block.";
        case boost::regex_constants::error_escape:
            return "An invalid or trailing escape was encountered.";
        case boost::regex_constants::error_backref:
            return "A back-reference to a non-existant marked sub-expression was encountered.";
        case boost::regex_constants::error_brack:
            return "An invalid character set [...] was encountered.";
        case boost::regex_constants::error_paren:
            return "Mismatched '(' and ')'.";
        case boost::regex_constants::error_brace:
            return "Mismatched '{' and '}'.";
        case boost::regex_constants::error_badbrace:
            return "Invalid contents of a {...} block.";
        case boost::regex_constants::error_range:
            return "A character range was invalid, for example [d-a].";
        case boost::regex_constants::error_space:
            return "Out of memory.";
        case boost::regex_constants::error_badrepeat:
            return "An attempt to repeat something that can not be repeated - for example a*+";
        case boost::regex_constants::error_end:
            return "Unexpected end of regular expression."; // Not used in Boost 1.47
        case boost::regex_constants::error_size:
            return "Regular expression too big.";
        case boost::regex_constants::error_right_paren:
            return "Unmatched ')'."; // Not used in Boost 1.47
        case boost::regex_constants::error_empty:
            return "Regular expression starts or ends with the alternation operator |.";
        case boost::regex_constants::error_complexity:
            return "The expression became too complex to handle.";
        case boost::regex_constants::error_stack:
            return "Out of program stack space.";
        case boost::regex_constants::error_perl_extension:
            return "An invalid Perl extension was encountered.";
        case boost::regex_constants::error_unknown:
            return "Unknown error.";
    }

    LOG_ERROR("Unexpected error code " << code);
    return "Unexpected error.";
}

} // anonymous namespace


namespace ml {
namespace core {


CRegex::CRegex(void)
    : m_Initialised(false) {
}

bool CRegex::init(const std::string &regex) {
    // Allow expression to be initialised twice
    m_Initialised = false;

    try {
        m_Regex = boost::regex(regex.c_str());
    } catch (boost::regex_error &e) {
        if (static_cast<size_t>(e.position()) <= regex.size()) {
            LOG_ERROR("Unable to compile regex: '" <<
                      regex << "' '" <<
                      regex.substr(0, e.position()) << "' '" <<
                      regex.substr(e.position()) << "': " <<
                      ::translateErrorCode(e.code()));
        } else {
            LOG_ERROR("Unable to compile regex: '" << regex << "': " <<
                      ::translateErrorCode(e.code()));
        }
        return false;
    } catch (std::exception &e) {
        LOG_ERROR("Unable to compile regex: " << e.what());
        return false;
    }

    m_Initialised = true;

    return true;
}

bool CRegex::tokenise(const std::string &str,
                      CRegex::TStrVec &tokens) const {
    tokens.clear();

    if (!m_Initialised) {
        LOG_ERROR("Regex not initialised");
        return false;
    }

    try {
        boost::smatch matches;
        if (boost::regex_match(str, matches, m_Regex) == false) {
            return false;
        }

        for (int i = 1; i < static_cast<int>(matches.size()); ++i) {
            tokens.push_back(std::string(matches[i].first, matches[i].second));
        }
    } catch (boost::regex_error &e) {
        LOG_ERROR("Unable to tokenise using regex: '" << str << "': " <<
                  ::translateErrorCode(e.code()));
        return false;
    } catch (std::exception &e) {
        LOG_ERROR("Unable to tokenise using regex: " << e.what());
        return false;
    }

    return true;
}

bool CRegex::split(const std::string &str,
                   CRegex::TStrVec &tokens) const {
    tokens.clear();

    if (!m_Initialised) {
        LOG_ERROR("Regex not initialised");
        return false;
    }

    try {
        boost::sregex_token_iterator i(str.begin(), str.end(), m_Regex, -1);
        boost::sregex_token_iterator j;

        while(i != j) {
            tokens.push_back(*i++);
        }
    } catch (boost::regex_error &e) {
        LOG_ERROR("Unable to tokenise using regex: '" << str << "': " <<
                  ::translateErrorCode(e.code()));
        return false;
    } catch (std::exception &e) {
        LOG_ERROR("Unable to tokenise using regex: " << e.what());
        return false;
    }

    return true;
}

bool CRegex::matches(const std::string &str) const {
    if (!m_Initialised) {
        LOG_ERROR("Regex not initialised");
        return false;
    }

    try {
        boost::smatch matches;
        if (boost::regex_match(str, matches, m_Regex) == false) {
            return false;
        }
    } catch (boost::regex_error &e) {
        LOG_ERROR("Unable to match using regex: '" << str << "': " <<
                  ::translateErrorCode(e.code()));
        return false;
    } catch (std::exception &e) {
        LOG_ERROR("Unable to match using regex: " << e.what());
        return false;
    }

    return true;
}

bool CRegex::search(size_t startPos,
                    const std::string &str,
                    size_t &position,
                    size_t &length) const {
    if (!m_Initialised) {
        LOG_ERROR("Regex not initialised");
        return false;
    }

    if (startPos >= str.length()) {
        return false;
    }

    try {
        boost::smatch matches;
        if (boost::regex_search(str.begin() + startPos,
                                str.begin() + str.length(),
                                matches,
                                m_Regex) == false) {
            return false;
        }

        position = matches[0].first - str.begin();
        length = matches[0].second - matches[0].first;
    } catch (boost::regex_error &e) {
        LOG_ERROR("Unable to search using regex: '" << str << "': " <<
                  ::translateErrorCode(e.code()));
        return false;
    } catch (std::exception &e) {
        LOG_ERROR("Unable to match using regex: " << e.what());
        return false;
    }

    return true;
}

bool CRegex::search(size_t startPos,
                    const std::string &str,
                    size_t &position) const {
    size_t length(0);

    return this->search(startPos, str, position, length);
}

bool CRegex::search(const std::string &str,
                    size_t &position,
                    size_t &length) const {
    return this->search(0, str, position, length);
}

bool CRegex::search(const std::string &str,
                    size_t &position) const {
    size_t length(0);

    return this->search(0, str, position, length);
}

std::string CRegex::str(void) const {
    if (!m_Initialised) {
        LOG_ERROR("Regex not initialised");
        return std::string();
    }

    return m_Regex.str();
}

size_t CRegex::literalCount(void) const {
    if (!m_Initialised) {
        LOG_ERROR("Regex not initialised");
        return 0;
    }

    // This is only approximate at the moment - there will be cases it gets
    // things wrong - good enough for now, but may need improving in the future
    // depending on what it's used for

    size_t count(0);

    std::string regexStr(m_Regex.str());

    bool inSubMatch(false);
    size_t squareBracketCount(0);
    size_t braceCount(0);
    size_t subCount(0);
    size_t minSubCount(std::numeric_limits<size_t>::max());

    for (std::string::iterator iter = regexStr.begin();
         iter != regexStr.end();
         ++iter) {
        char thisChar(*iter);

        switch (thisChar) {
            case '$':
                // Perl can expand variables, so should really skip over
                // variable names at this point
                break;
            case '.':
            case '^':
            case '*':
            case '+':
            case '?':
                break;
            case '\\':
                ++iter;
                if (iter == regexStr.end()) {
                    LOG_ERROR("Inconsistency - backslash at the end of regex");
                    return count;
                }
                thisChar = *iter;
                if (thisChar != 'd' && thisChar != 's' && thisChar != 'w' &&
                    thisChar != 'D' && thisChar != 'S' && thisChar != 'W' &&
                    (thisChar < '0' || thisChar > '9')) {
                    if (squareBracketCount == 0 && braceCount == 0) {
                        std::string::iterator nextIter(iter + 1);
                        if (nextIter == regexStr.end() ||
                            (*nextIter != '*' && *nextIter != '+' && *nextIter != '?')) {
                            if (inSubMatch) {
                                ++subCount;
                            } else {
                                ++count;
                            }
                        }
                    }
                }
                break;
            case '[':
                ++squareBracketCount;
                break;
            case ']':
                if (squareBracketCount == 0) {
                    LOG_ERROR("Inconsistency - more ] than [");
                } else {
                    --squareBracketCount;
                }
                break;
            case '{':
                ++braceCount;
                break;
            case '}':
                if (braceCount == 0) {
                    LOG_ERROR("Inconsistency - more } than {");
                } else {
                    --braceCount;
                }
                break;
            case '|':
                if (inSubMatch) {
                    if (subCount < minSubCount) {
                        minSubCount = subCount;
                    }
                    subCount = 0;
                } else {
                }
                break;
            case '(':
                inSubMatch = true;
                break;
            case ')':
                inSubMatch = false;
                if (subCount < minSubCount) {
                    minSubCount = subCount;
                }
                count += minSubCount;
                subCount = 0;
                minSubCount = std::numeric_limits<size_t>::max();
                break;
            default:
                if (squareBracketCount == 0 && braceCount == 0) {
                    std::string::iterator nextIter(iter + 1);
                    if (nextIter == regexStr.end() ||
                        (*nextIter != '*' && *nextIter != '+' && *nextIter != '?')) {
                        if (inSubMatch) {
                            ++subCount;
                        } else {
                            ++count;
                        }
                    }
                }
                break;
        }
    }

    return count;
}

std::string CRegex::escapeRegexSpecial(const std::string &literal) {
    std::string result;
    result.reserve(literal.size());

    for (std::string::const_iterator iter = literal.begin();
         iter != literal.end();
         ++iter) {
        char thisChar = *iter;

        switch (thisChar) {
            case '.':
            case '*':
            case '+':
            case '?':
            case '|':
            case '^':
            case '$':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '\\':
                result += '\\';
                result += thisChar;
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                // Carriage returns are made optional to prevent the regex
                // having a silly incompatibility between Windows text and Unix
                // text files
                result += "\\r?";
                break;
            default:
                result += thisChar;
                break;
        }
    }

    return result;
}


}
}

