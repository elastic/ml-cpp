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
#include <core/CDelimiter.h>

#include <core/CLogger.h>

#include <algorithm>
#include <ostream>

namespace ml {
namespace core {

const std::string CDelimiter::DEFAULT_DELIMITER(",");

CDelimiter::CDelimiter()
    : m_Valid(m_Delimiter.init(DEFAULT_DELIMITER)),
      m_HaveFollowingRegex(false),
      m_WaiveFollowingRegexAfterTime(false),
      m_Quote('\0'),
      m_Escape('\0') {
}

CDelimiter::CDelimiter(const std::string& delimiter)
    : m_Valid(m_Delimiter.init(delimiter)),
      m_HaveFollowingRegex(false),
      m_WaiveFollowingRegexAfterTime(false),
      m_Quote('\0'),
      m_Escape('\0') {
    if (!m_Valid) {
        LOG_ERROR("Unable to set delimiter regex to " << delimiter);
    }
}

CDelimiter::CDelimiter(const std::string& delimiter, const std::string& followingRegex, bool orTime)
    : m_Valid(m_Delimiter.init(delimiter)),
      m_HaveFollowingRegex(m_FollowingRegex.init(followingRegex)),
      m_WaiveFollowingRegexAfterTime(orTime),
      m_Quote('\0'),
      m_Escape('\0') {
    if (!m_Valid) {
        LOG_ERROR("Unable to set delimiter regex to " << delimiter);
    }

    if (!m_HaveFollowingRegex) {
        LOG_ERROR("Unable to set following regex to " << followingRegex);
    }
}

bool CDelimiter::operator==(const CDelimiter& rhs) const {
    if (m_Valid != rhs.m_Valid || m_HaveFollowingRegex != rhs.m_HaveFollowingRegex ||
        m_WaiveFollowingRegexAfterTime != rhs.m_WaiveFollowingRegexAfterTime || m_Quote != rhs.m_Quote || m_Escape != rhs.m_Escape) {
        return false;
    }

    // Only test more complex conditions if simple ones passed
    if (m_Valid) {
        if (m_Delimiter.str() != rhs.m_Delimiter.str()) {
            return false;
        }
    }

    if (m_HaveFollowingRegex) {
        if (m_FollowingRegex.str() != rhs.m_FollowingRegex.str()) {
            return false;
        }
    }

    return true;
}

bool CDelimiter::operator!=(const CDelimiter& rhs) const {
    return !this->operator==(rhs);
}

// Check whether the text that followed the primary delimiter was acceptable
bool CDelimiter::isFollowingTextAcceptable(size_t searchPos, const std::string& str, bool timePassed) const {
    bool answer(false);

    if (m_HaveFollowingRegex) {
        if (m_WaiveFollowingRegexAfterTime && timePassed && searchPos == str.length()) {
            answer = true;
        } else {
            size_t foundPos(0);
            bool found = m_FollowingRegex.search(searchPos, str, foundPos);
            if (found && foundPos == searchPos) {
                answer = true;
            }
        }
    } else {
        answer = true;
    }

    return answer;
}

bool CDelimiter::valid() const {
    return m_Valid;
}

std::string CDelimiter::delimiter() const {
    return m_Delimiter.str();
}

void CDelimiter::tokenise(const std::string& str, CStringUtils::TStrVec& tokens, std::string& remainder) const {
    std::string exampleDelimiter;
    this->tokenise(str, false, tokens, exampleDelimiter, remainder);
}

void CDelimiter::tokenise(const std::string& str, bool timePassed, CStringUtils::TStrVec& tokens, std::string& remainder) const {
    std::string exampleDelimiter;
    this->tokenise(str, timePassed, tokens, exampleDelimiter, remainder);
}

void CDelimiter::tokenise(const std::string& str,
                          CStringUtils::TStrVec& tokens,
                          std::string& exampleDelimiter,
                          std::string& remainder) const {
    this->tokenise(str, false, tokens, exampleDelimiter, remainder);
}

void CDelimiter::tokenise(const std::string& str,
                          bool timePassed,
                          CStringUtils::TStrVec& tokens,
                          std::string& exampleDelimiter,
                          std::string& remainder) const {
    tokens.clear();
    exampleDelimiter.clear();

    if (!m_Valid) {
        LOG_ERROR("Cannot tokenise using invalid delimiter");
        remainder.clear();
        return;
    }

    size_t tokenStartPos(0);
    size_t delimStartPos(0);
    size_t delimLength(0);
    size_t searchPos(0);

    bool expectingQuote(false);

    for (;;) {
        size_t quotePos(this->getNextQuote(str, searchPos));

        // Check if the very first character is a quote
        if (quotePos == 0) {
            searchPos = 1;
            tokenStartPos = 1;
            expectingQuote = true;
            continue;
        }

        // If we're expecting a quote and don't find one, the rest of the string
        // is remainder
        if (expectingQuote && quotePos == std::string::npos) {
            // Don't unescape the result, as this might be from a partial read
            // that needs to be prepended to the next read
            remainder.assign(str, tokenStartPos, std::string::npos);

            return;
        }

        // Search for the delimiter
        bool found(m_Delimiter.search(expectingQuote ? (quotePos + 1) : searchPos, str, delimStartPos, delimLength));
        if (!found) {
            if (expectingQuote && quotePos < str.length()) {
                // If we're expecting a quote and find one, treat this as
                // another token
                remainder.assign(str, tokenStartPos, quotePos - tokenStartPos);
                CStringUtils::unEscape(m_Escape, remainder);
            } else {
                // If we're not expecting a quote, don't unescape the result,
                // as this might be from a partial read that needs to be
                // prepended to the next read
                remainder.assign(str, tokenStartPos, std::string::npos);
            }

            return;
        }

        // Check for stray quotes
        if (!expectingQuote && quotePos <= delimStartPos) {
            LOG_WARN("String to be delimited does not conform to config:"
                     " quote pos "
                     << quotePos << " delim pos " << delimStartPos);
        }

        // Move the search position beyond the last, regardless of
        // whether it's acceptable as the end of the token
        searchPos = delimStartPos + delimLength;
        if (this->isFollowingTextAcceptable(searchPos, str, timePassed)) {
            if (exampleDelimiter.empty()) {
                exampleDelimiter.assign(str, delimStartPos, delimLength);
            }

            size_t tokenLength(delimStartPos - tokenStartPos);
            if (expectingQuote) {
                tokenLength = quotePos - tokenStartPos;
            }

            // Following text was acceptable, so keep what we've
            // accumulated to date.  The following text is NOT passed
            // over, so that it can become part of the next token.
            tokens.push_back(str.substr(tokenStartPos, tokenLength));
            CStringUtils::unEscape(m_Escape, tokens.back());

            if (this->getNextQuote(str, searchPos) == searchPos) {
                // Quote comes immediately after delimiter, so skip and
                // expect a quote next
                ++searchPos;
                expectingQuote = true;
            } else {
                expectingQuote = false;
            }

            tokenStartPos = searchPos;
        } else {
            if (this->getNextQuote(str, searchPos) == searchPos) {
                // Quote comes immediately after delimiter, so skip and
                // expect a quote next
                ++searchPos;
                expectingQuote = true;
            } else {
                expectingQuote = false;
            }
        }
    }
}

void CDelimiter::quote(char quote, char escape) {
    m_Quote = quote;
    m_Escape = escape;
}

char CDelimiter::quote() const {
    return m_Quote;
}

size_t CDelimiter::getNextQuote(const std::string& str, size_t startPos) const {
    size_t result(std::string::npos);

    if (m_Quote != '\0') {
        while (startPos < str.length()) {
            size_t quotePos(str.find(m_Quote, startPos));

            // If no quote found at all then give up
            if (quotePos == std::string::npos) {
                break;
            }

            // If quote is not escaped, set result and stop - different logic is
            // needed for the case where the escape character is the same as the
            // quote character
            if (m_Quote == m_Escape) {
                if (quotePos == str.length() - 1 || str[quotePos + 1] != m_Escape) {
                    result = quotePos;
                    break;
                }

                // Continue searching beyond the escaped quote
                startPos = quotePos + 2;
            } else {
                if (quotePos == startPos || str[quotePos - 1] != m_Escape) {
                    result = quotePos;
                    break;
                }

                // Continue searching beyond the escaped quote
                startPos = quotePos + 1;
            }
        }
    }

    return result;
}

std::ostream& operator<<(std::ostream& strm, const CDelimiter& delimiter) {
    strm << "Delimiter { ";

    if (delimiter.m_Valid) {
        strm << "Regex " << delimiter.m_Delimiter.str();

        if (delimiter.m_Quote != '\0') {
            strm << ", Quote " << delimiter.m_Quote;

            if (delimiter.m_Escape != '\0') {
                strm << ", Escape " << delimiter.m_Escape;
            }
        }

        if (delimiter.m_HaveFollowingRegex) {
            strm << ", Following Regex " << delimiter.m_FollowingRegex.str();
        }

        if (delimiter.m_WaiveFollowingRegexAfterTime) {
            strm << ", Following Regex Waived After Time";
        }
    } else {
        strm << "Invalid!";
    }

    strm << " }";

    return strm;
}
}
}
