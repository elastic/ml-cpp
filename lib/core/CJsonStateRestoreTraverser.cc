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
    : m_ReadStream(inputStream), m_Reader(json::parse_options()), m_Handler(m_Reader.handler()), m_Started(false),
      m_DesiredLevel(0), m_IsArrayOfObjects(false) {
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
    // This is to match the functionality of the pre-existing XML state
    // traverser.
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
    SBoostJsonHandler::ETokenType currentTokenType = m_Handler.s_Type;
    do {
        if (m_Reader.last_error()) {
            this->logError();
            ret = false;
            break;
        }

        m_Handler.s_RememberValue = remember;

        char c = '\0';
        m_ReadStream.get(c);
        if (c == '\0') {
            break;
        }

        // TODO
//        if (c == '\n') {
//            m_Handler.s_Type = SBoostJsonHandler::E_TokenWhiteSpace;
//            continue ;
//        }

        if (c == '"' && m_Handler.s_Type == SBoostJsonHandler::E_TokenObjectStart) {
            m_Handler.s_Type = SBoostJsonHandler::E_TokenKey;
        }

        if (c == ',') {
            m_Handler.s_Type = SBoostJsonHandler::E_TokenComma;
        }

        if (c == ':') {
            m_Handler.s_Type = SBoostJsonHandler::E_TokenColon;
        }

        json::error_code ec;
        m_Reader.write_some(true, &c, 1, ec);
        if (ec) {
            this->logError();
            ret = false;
            break;
        }
    } while ((m_Handler.s_Type != SBoostJsonHandler::E_TokenObjectEnd ||
              m_Handler.s_Type != SBoostJsonHandler::E_TokenArrayEnd) &&
             (currentTokenType == m_Handler.s_Type ||
              m_Handler.s_Type == SBoostJsonHandler::E_TokenKeyPart ||
              m_Handler.s_Type == SBoostJsonHandler::E_TokenStringPart ||
              m_Handler.s_Type == SBoostJsonHandler::E_TokenComma ||
              m_Handler.s_Type == SBoostJsonHandler::E_TokenColon ||
              m_Handler.s_Type == SBoostJsonHandler::E_TokenWhiteSpace));

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

        LOG_ERROR(<< "JSON state must be object at root" << m_Handler.s_Type);
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
    LOG_ERROR(<< "Error parsing JSON: " << m_Reader.last_error());
    this->setBadState();
}

CJsonStateRestoreTraverser::SBoostJsonHandler::SBoostJsonHandler()
    : s_Type(SBoostJsonHandler::E_TokenNull), s_NextIndex(0), s_RememberValue(false) {
    s_Level[0] = 0;
    s_Level[1] = 0;
    s_IsEndOfLevel[0] = false;
    s_IsEndOfLevel[1] = false;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_null(json::error_code&/* ec*/) {
    s_Type = E_TokenNull;
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_bool(bool b, json::error_code&/* ec*/) {
    s_Type = E_TokenBool;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(b));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_int64(std::int64_t i, std::string_view/* s*/, json::error_code&/* ec*/) {
    s_Type = E_TokenInt64;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(i));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_uint64(std::uint64_t u, std::string_view/* s*/, json::error_code&/* ec*/) {
    s_Type = E_TokenUInt64;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(u));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_double(double d, std::string_view/* s*/, json::error_code&/* ec*/) {
    s_Type = E_TokenDouble;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(d));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_string_part( std::string_view s, std::size_t n, json::error_code&/* ec*/ ) {
    LOG_TRACE(<< "on_string_part: '" << s << "', n: " << n);

    s_Type = E_TokenStringPart;
    if (s_RememberValue) {
        if (m_NewToken) {
            s_Value[s_NextIndex].clear();
            m_NewToken = false;
        }
        s_Value[s_NextIndex].push_back(s.front());
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_string( std::string_view s, std::size_t n, json::error_code&/* ec*/ ) {
    if (s_Type != E_TokenStringPart && s.front() == '"') { // Empty string
        s_Value[s_NextIndex].clear();
    }
    LOG_DEBUG(<< "on_string: '" << s_Value[s_NextIndex] << "', n: " << n << ", s_NextIndex: " << s_NextIndex);

    s_Type = E_TokenString;
    if (s_RememberValue) {
        if (s.front() != '"') {
            s_Value[s_NextIndex].push_back(s.front());
        }
        m_NewToken = true;
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_document_begin( json::error_code& ec ) {
    LOG_DEBUG(<< "on_document_begin");

    return (ec) ? false : true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_object_begin(json::error_code& ec) {
    LOG_DEBUG(<< "on_object_begin");
    s_Type = E_TokenObjectStart;

    if (ec) {
        return false;
    }

    if (s_RememberValue) {
        ++s_Level[s_NextIndex];
        s_Value[s_NextIndex].clear();
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_key_part( std::string_view s, std::size_t n, json::error_code& ec ) {
    s_Type = E_TokenKeyPart;
    LOG_TRACE(<< "on_key_part: '" << s << "', n: " << n);
    if (ec) {
        return false;
    }

    if (s_RememberValue) {
        if (m_NewToken) {
            m_NewToken = false;
            s_NextIndex = 1 - s_NextIndex;
            s_Name[s_NextIndex].clear();
        }
        s_Name[s_NextIndex].push_back(s.front());
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_key( std::string_view s, std::size_t n, json::error_code& ec ) {
    s_Type = E_TokenKey;
    LOG_DEBUG(<< "on_key: '" << s_Name[s_NextIndex] << "', n: " << n << ", s_NextIndex: " << s_NextIndex);
    if (ec) {
        return false;
    }
    if (s_RememberValue) {
        if (s.front() != '"') {
            s_Name[s_NextIndex].push_back(s.front());
        }
        s_Level[s_NextIndex] = s_Level[1 - s_NextIndex];
        s_IsEndOfLevel[s_NextIndex] = false;
        m_NewToken = true;
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_object_end(std::size_t/* n*/, json::error_code& ec) {
    LOG_DEBUG(<< "on_object_end");
    s_Type = E_TokenObjectEnd;
    if (ec) {
        return false;
    }

    if (s_RememberValue) {
        s_NextIndex = 1 - s_NextIndex;
        if (s_Level[1 - s_NextIndex] > 0) {
            s_Level[s_NextIndex] = s_Level[1 - s_NextIndex] - 1;
        }
        s_IsEndOfLevel[s_NextIndex] = true;
        s_Name[s_NextIndex].clear();
        s_Value[s_NextIndex].clear();
    }

    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_array_begin(json::error_code& ec) {
    s_Type = E_TokenArrayStart;
    if (ec) {
        return false;
    }
    return true;
}

bool CJsonStateRestoreTraverser::SBoostJsonHandler::on_array_end(std::size_t/* n*/, json::error_code& ec) {
    s_Type = E_TokenArrayEnd;
    if (ec) {
        return false;
    }
    return true;
}
}
}
