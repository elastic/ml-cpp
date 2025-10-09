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
#include <core/CJsonStateRestoreTraverser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <boost/json.hpp>

// This file must be manually included when
// using basic_parser to implement a parser.
#include <boost/json/basic_parser_impl.hpp>

namespace json = boost::json;

namespace ml {
namespace core {

namespace {
const std::string EMPTY_STRING;
}

CJsonStateRestoreTraverser::CJsonStateRestoreTraverser(std::istream& inputStream)
    : m_ReadStream(inputStream), m_Reader({32, {}, false, false, true, false}),
      m_Handler(m_Reader.handler()), m_Started(false), m_DesiredLevel(0),
      m_IsArrayOfObjects(false) {
}

bool CJsonStateRestoreTraverser::isEof() const {
    // CBoostJsonUnbufferedIStreamWrapper returns \0 when it reaches EOF
    return m_ReadStream.peek() == '\0';
}

bool CJsonStateRestoreTraverser::next() {
    if (haveBadState()) {
        return false;
    }

    if (!m_Started) {
        if (this->start() == false) {
            return false;
        }
    }

    if (this->nextIsEndOfLevel()) {
        return false;
    }

    if (this->nextLevel() == m_DesiredLevel ||
        (this->currentLevel() == m_DesiredLevel && this->nextLevel() == m_DesiredLevel + 1)) {
        return this->advance();
    }

    // If we get here then we're skipping over a nested object that's not of
    // interest
    while (this->nextLevel() > m_DesiredLevel) {
        if (this->advance() == false) {
            return false;
        }
    }

    if (this->nextLevel() == m_DesiredLevel) {
        return this->advance() && !this->nextIsEndOfLevel();
    }

    return false;
}

bool CJsonStateRestoreTraverser::nextObject() {
    if (!m_IsArrayOfObjects) {
        return false;
    }

    if (haveBadState()) {
        return false;
    }

    // Advance to the next start object token
    bool ok = this->advance() && this->advance();
    ok = ok && this->next();

    return ok;
}

bool CJsonStateRestoreTraverser::hasSubLevel() const {
    if (!m_Started) {
        if (const_cast<CJsonStateRestoreTraverser*>(this)->start() == false) {
            return false;
        }
    }

    if (haveBadState()) {
        return false;
    }

    return this->currentLevel() == 1 + m_DesiredLevel;
}

const std::string& CJsonStateRestoreTraverser::name() const {
    if (haveBadState()) {
        return EMPTY_STRING;
    }

    if (!m_Started) {
        if (const_cast<CJsonStateRestoreTraverser*>(this)->start() == false) {
            return EMPTY_STRING;
        }
    }

    return this->currentName();
}

const std::string& CJsonStateRestoreTraverser::value() const {
    if (haveBadState()) {
        return EMPTY_STRING;
    }

    if (!m_Started) {
        if (const_cast<CJsonStateRestoreTraverser*>(this)->start() == false) {
            return EMPTY_STRING;
        }
    }

    return this->currentValue();
}

bool CJsonStateRestoreTraverser::descend() {
    if (!m_Started) {
        if (this->start() == false) {
            return false;
        }
    }

    if (haveBadState()) {
        return false;
    }

    if (this->currentLevel() != 1 + m_DesiredLevel) {
        return false;
    }

    ++m_DesiredLevel;

    // Don't advance if the next level has no elements.  Instead set the current
    // element to be completely empty so that the sub-level traverser will find
    // nothing and then ascend.
    if (this->nextIsEndOfLevel()) {
        m_Handler.s_Name[1 - m_Handler.s_NextIndex].clear();
        m_Handler.s_Value[1 - m_Handler.s_NextIndex].clear();
        return true;
    }

    return this->advance();
}

bool CJsonStateRestoreTraverser::ascend() {
    // If we're trying to ascend above the root level then something has gone
    // wrong
    if (m_DesiredLevel == 0) {
        LOG_ERROR(<< "Inconsistency - trying to ascend above JSON root");
        return false;
    }

    if (haveBadState()) {
        return false;
    }

    --m_DesiredLevel;

    while (this->nextLevel() > m_DesiredLevel) {
        if (this->advance() == false) {
            return false;
        }
    }

    // This will advance onto the end-of-level marker.  Slightly unintuitively
    // it's then still necessary to call next() to move to the higher level.
    return this->advance();
}

void CJsonStateRestoreTraverser::debug() const {
    LOG_DEBUG(<< "Current: name = " << this->currentName() << " value = "
              << this->currentValue() << " level = " << this->currentLevel()
              << ", Next: name = " << this->nextName()
              << " value = " << this->nextValue() << " level = " << this->nextLevel()
              << " is array of objects = " << m_IsArrayOfObjects);
}

size_t CJsonStateRestoreTraverser::currentLevel() const {
    return m_Handler.s_Level[1 - m_Handler.s_NextIndex];
}

bool CJsonStateRestoreTraverser::currentIsEndOfLevel() const {
    return m_Handler.s_IsEndOfLevel[1 - m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::currentName() const {
    return m_Handler.s_Name[1 - m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::currentValue() const {
    return m_Handler.s_Value[1 - m_Handler.s_NextIndex];
}

size_t CJsonStateRestoreTraverser::nextLevel() const {
    return m_Handler.s_Level[m_Handler.s_NextIndex];
}

bool CJsonStateRestoreTraverser::nextIsEndOfLevel() const {
    return m_Handler.s_IsEndOfLevel[m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::nextName() const {
    return m_Handler.s_Name[m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::nextValue() const {
    return m_Handler.s_Value[m_Handler.s_NextIndex];
}

bool CJsonStateRestoreTraverser::parseNext(bool remember) {
    bool ret{true};
    m_Handler.s_HaveCompleteToken = false;
    do {
        m_Handler.s_RememberValue = remember;

        if (m_BytesRemaining == 0) {
            ::memset(m_Buffer, '\0', BUFFER_SIZE);
            m_ReadStream.get(m_Buffer, BUFFER_SIZE, '\0');
            if (m_ReadStream.bad()) {
                LOG_ERROR(<< "Input stream is bad");
                this->setBadState();
                return false;
            }
            m_BytesRemaining = m_ReadStream.gcount();
            m_BufferPtr = m_Buffer;
            if (*m_BufferPtr == '\0') {
                break;
            }
        }

        json::error_code ec;
        char c = *m_BufferPtr;
        std::size_t written = m_Reader.write_some(true, &c, 1, ec);
        if (ec) {
            this->logError();
            ret = false;
            break;
        }
        m_BytesRemaining -= written;
        m_BufferPtr++;
    } while (m_Handler.s_HaveCompleteToken == false);

    return ret;
}

bool CJsonStateRestoreTraverser::skipArray() {
    int depth = 0;

    // we must have received a key, revert the state change to ignore it
    m_Handler.s_NextIndex = 1 - m_Handler.s_NextIndex;

    do {
        if (m_Handler.s_Type == SBoostJsonHandler::E_TokenArrayStart ||
            m_Handler.s_Type == SBoostJsonHandler::E_TokenObjectStart) {
            ++depth;
        } else if (m_Handler.s_Type == SBoostJsonHandler::E_TokenArrayEnd ||
                   m_Handler.s_Type == SBoostJsonHandler::E_TokenObjectEnd) {
            --depth;
        }

        if (parseNext(depth == 0) == false) {
            this->logError();
            return false;
        }
    } while (depth > 0);
    return true;
}

bool CJsonStateRestoreTraverser::start() {
    m_Started = true;
    m_Reader.reset();

    if (this->parseNext(false) == false) {
        this->logError();
        return false;
    }

    // If the first token is start of array then this could be
    // an array of docs. Next should be start object
    if (m_Handler.s_Type == SBoostJsonHandler::E_TokenArrayStart) {
        if (this->parseNext(false) == false) {
            this->logError();
            return false;
        }

        m_IsArrayOfObjects = true;
    }

    // For Ml state the first token should be the start of a JSON
    // object, but we don't store it
    if (m_Handler.s_Type != SBoostJsonHandler::E_TokenObjectStart) {
        if (m_IsArrayOfObjects &&
            m_Handler.s_Type == SBoostJsonHandler::E_TokenArrayEnd && this->isEof()) {
            LOG_DEBUG(<< "JSON document is an empty array");
            return false;
        }

        // Enhanced error logging with comprehensive debugging information
        std::string tokenTypeName;
        switch (m_Handler.s_Type) {
            case SBoostJsonHandler::E_TokenNull:
                tokenTypeName = "null";
                break;
            case SBoostJsonHandler::E_TokenKey:
                tokenTypeName = "key";
                break;
            case SBoostJsonHandler::E_TokenBool:
                tokenTypeName = "boolean";
                break;
            case SBoostJsonHandler::E_TokenInt64:
                tokenTypeName = "int64";
                break;
            case SBoostJsonHandler::E_TokenUInt64:
                tokenTypeName = "uint64";
                break;
            case SBoostJsonHandler::E_TokenDouble:
                tokenTypeName = "double";
                break;
            case SBoostJsonHandler::E_TokenString:
                tokenTypeName = "string";
                break;
            case SBoostJsonHandler::E_TokenObjectStart:
                tokenTypeName = "object_start";
                break;
            case SBoostJsonHandler::E_TokenObjectEnd:
                tokenTypeName = "object_end";
                break;
            case SBoostJsonHandler::E_TokenArrayStart:
                tokenTypeName = "array_start";
                break;
            case SBoostJsonHandler::E_TokenArrayEnd:
                tokenTypeName = "array_end";
                break;
            case SBoostJsonHandler::E_TokenKeyPart:
                tokenTypeName = "key_part";
                break;
            case SBoostJsonHandler::E_TokenStringPart:
                tokenTypeName = "string_part";
                break;
            default:
                tokenTypeName = "unknown";
                break;
        }

        LOG_ERROR(<< "JSON state must be object at root. Found token type: " 
                  << tokenTypeName << " (value: " << m_Handler.s_Type 
                  << "), isArrayOfObjects: " << m_IsArrayOfObjects 
                  << ", isEof: " << this->isEof()
                  << ", stream state - bad: " << m_ReadStream.bad() 
                  << ", fail: " << m_ReadStream.fail()
                  << ", eof: " << m_ReadStream.eof()
                  << ", bytes remaining: " << m_BytesRemaining
                  << ", buffer position: " << (m_BufferPtr ? (m_BufferPtr - m_Buffer) : -1));
        return false;
    }

    // Advance twice to prime the current and next elements
    return this->advance() && this->advance();
}

bool CJsonStateRestoreTraverser::advance() {
    if (haveBadState()) {
        return false;
    }

    bool keepGoing(true);

    while (keepGoing) {
        if (this->parseNext(true) == false) {
            if (!this->isEof()) {
                this->logError();
            }
            return false;
        }

        if (m_Handler.s_Type == SBoostJsonHandler::E_TokenArrayStart) {
            LOG_ERROR(<< "JSON state should not contain arrays");
            this->skipArray();
        } else if (m_Handler.s_Type != SBoostJsonHandler::E_TokenKey) {
            keepGoing = false;
        }
    }

    return true;
}

void CJsonStateRestoreTraverser::logError() {
    LOG_ERROR(<< "Error parsing JSON: " << m_Reader.last_error()
              << ", stream state - bad: " << m_ReadStream.bad() 
              << ", fail: " << m_ReadStream.fail()
              << ", eof: " << m_ReadStream.eof()
              << ", bytes remaining: " << m_BytesRemaining
              << ", buffer position: " << (m_BufferPtr ? (m_BufferPtr - m_Buffer) : -1)
              << ", current token type: " << m_Handler.s_Type
              << ", isArrayOfObjects: " << m_IsArrayOfObjects
              << ", started: " << m_Started
              << ", desired level: " << m_DesiredLevel);
    this->setBadState();
}

CJsonStateRestoreTraverser::SBoostJsonHandler::SBoostJsonHandler()
    : s_Type(SBoostJsonHandler::E_TokenNull), s_NextIndex(0), s_RememberValue(false) {
    s_Level[0] = 0;
    s_Level[1] = 0;
    s_IsEndOfLevel[0] = false;
    s_IsEndOfLevel[1] = false;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_null(json::error_code& ec) {
    s_Type = E_TokenNull;
    if (ec) {
        LOG_ERROR(<< "on_null: ERROR: " << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_bool(bool b, json::error_code& ec) {
    s_Type = E_TokenBool;
    if (ec) {
        LOG_ERROR(<< "on_bool: ERROR: b: " << b << ". " << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(b));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_int64(std::int64_t i,
                                                             std::string_view s,
                                                             json::error_code& ec) {
    s_Type = E_TokenInt64;
    if (ec) {
        LOG_ERROR(<< "on_int64: ERROR: i: " << i << ", s: '" << s << "'. "
                  << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;
    if (s_RememberValue) {
        // Frustratingly, the string_view passed to this handler points to characters
        // beyond those used to parse the number value. Hence we need to convert the
        // number _back_ to a string.
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(i));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_uint64(std::uint64_t u,
                                                              std::string_view s,
                                                              json::error_code& ec) {
    s_Type = E_TokenUInt64;
    if (ec) {
        LOG_ERROR(<< "on_uint64: ERROR: u: " << u << ", s: '" << s << "'. "
                  << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;
    if (s_RememberValue) {
        // Frustratingly, the string_view passed to this handler points to characters
        // beyond those used to parse the number value. Hence we need to convert the
        // number _back_ to a string.
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(u));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_double(double d,
                                                              std::string_view s,
                                                              json::error_code& ec) {
    s_Type = E_TokenDouble;
    if (ec) {
        LOG_ERROR(<< "on_double: ERROR: d: " << d << ", s: '" << s << "'. "
                  << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;
    if (s_RememberValue) {
        // Frustratingly, the string_view passed to this handler points to characters
        // beyond those used to parse the number value. Hence we need to convert the
        // number _back_ to a string.
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(d));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_string_part(std::string_view s,
                                                                   std::size_t n,
                                                                   json::error_code& ec) {
    s_Type = E_TokenStringPart;
    if (ec) {
        LOG_ERROR(<< "on_string_part: ERROR: s: '" << s << "', n: " << n << ". "
                  << ec.to_string());
        return false;
    }

    if (s_RememberValue) {
        if (m_NewToken) {
            s_Value[s_NextIndex].clear();
            m_NewToken = false;
        }
        s_Value[s_NextIndex].append(s);
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_string(std::string_view s,
                                                              std::size_t n,
                                                              json::error_code& ec) {
    if (ec) {
        LOG_ERROR(<< "on_string: ERROR: s: '" << s << "', n: " << n << ". "
                  << ec.to_string());
        return false;
    }

    if (s_Type != E_TokenStringPart && s.front() == '"') { // Empty string
        s_Value[s_NextIndex].clear();
    }

    s_Type = E_TokenString;
    s_HaveCompleteToken = true;
    if (s_RememberValue) {
        if (s.front() != '"') {
            s_Value[s_NextIndex].append(s);
        }
        m_NewToken = true;
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_document_begin(json::error_code& ec) {
    LOG_TRACE(<< "on_document_begin");

    return (ec) ? false : true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_object_begin(json::error_code& ec) {
    LOG_TRACE(<< "on_object_begin");
    if (ec) {
        return false;
    }

    s_Type = E_TokenObjectStart;
    s_HaveCompleteToken = true;

    if (s_RememberValue) {
        ++s_Level[s_NextIndex];
        s_Value[s_NextIndex].clear();
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_key_part(std::string_view s,
                                                                std::size_t n,
                                                                json::error_code& ec) {
    s_Type = E_TokenKeyPart;
    if (ec) {
        LOG_ERROR(<< "on_key_part: ERROR: s: '" << s << "', n: " << n << ". "
                  << ec.to_string());
        return false;
    }

    if (s_RememberValue) {
        if (m_NewToken) {
            m_NewToken = false;
            s_NextIndex = 1 - s_NextIndex;
            s_Name[s_NextIndex].clear();
        }
        s_Name[s_NextIndex].append(s);
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_key(std::string_view s,
                                                           std::size_t n,
                                                           json::error_code& ec) {
    s_Type = E_TokenKey;
    if (ec) {
        LOG_ERROR(<< "on_key: ERROR: s: '" << s << "', n: " << n << ". " << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;
    if (s_RememberValue) {
        if (s.front() != '"') {
            s_Name[s_NextIndex].append(s);
        }
        s_Level[s_NextIndex] = s_Level[1 - s_NextIndex];
        s_IsEndOfLevel[s_NextIndex] = false;
        m_NewToken = true;
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_object_end(std::size_t n,
                                                                  json::error_code& ec) {
    s_Type = E_TokenObjectEnd;
    if (ec) {
        LOG_ERROR(<< "on_object_end: ERROR: n: " << n << ". " << ec.to_string());
        return false;
    }

    s_HaveCompleteToken = true;

    if (s_RememberValue) {
        s_NextIndex = 1 - s_NextIndex;
        s_Level[s_NextIndex] = s_Level[1 - s_NextIndex] - 1;
        s_IsEndOfLevel[s_NextIndex] = true;
        s_Name[s_NextIndex].clear();
        s_Value[s_NextIndex].clear();
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_array_begin(json::error_code& ec) {
    s_Type = E_TokenArrayStart;
    if (ec) {
        LOG_ERROR(<< "on_array_begin: ERROR: " << ec.to_string());
        return false;
    }

    s_HaveCompleteToken = true;

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_array_end(std::size_t n,
                                                                 json::error_code& ec) {
    s_Type = E_TokenArrayEnd;
    if (ec) {
        LOG_ERROR(<< "on_array_end: ERROR: n: " << n << ". " << ec.to_string());
        return false;
    }
    s_HaveCompleteToken = true;

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_number_part(std::string_view /* s*/,
                                                                   json::error_code& ec) {
    return (ec) ? false : true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_comment_part(std::string_view /* s*/,
                                                                    json::error_code& ec) {
    return (ec) ? false : true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_comment(std::string_view /* s*/,
                                                               json::error_code& ec) {
    return (ec) ? false : true;
}
}
}
