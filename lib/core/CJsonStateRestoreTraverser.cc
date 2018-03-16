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
#include <core/CJsonStateRestoreTraverser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <rapidjson/error/en.h>
#include <rapidjson/rapidjson.h>

namespace ml {
namespace core {

namespace {
const std::string EMPTY_STRING;
}

CJsonStateRestoreTraverser::CJsonStateRestoreTraverser(std::istream& inputStream)
    : m_ReadStream(inputStream), m_Handler(), m_Started(false), m_DesiredLevel(0), m_IsArrayOfObjects(false) {
}

bool CJsonStateRestoreTraverser::isEof(void) const {
    // Rapid JSON istreamwrapper returns \0 when it reaches EOF
    return m_ReadStream.Peek() == '\0';
}

bool CJsonStateRestoreTraverser::next(void) {
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

bool CJsonStateRestoreTraverser::nextObject(void) {
    if (!m_IsArrayOfObjects) {
        return false;
    }

    // Advance to the next start object token
    bool ok = this->advance() && this->advance();
    ok = ok && this->next();

    return ok;
}

bool CJsonStateRestoreTraverser::hasSubLevel(void) const {
    if (!m_Started) {
        if (const_cast<CJsonStateRestoreTraverser*>(this)->start() == false) {
            return false;
        }
    }

    return this->currentLevel() == 1 + m_DesiredLevel;
}

const std::string& CJsonStateRestoreTraverser::name(void) const {
    if (!m_Started) {
        if (const_cast<CJsonStateRestoreTraverser*>(this)->start() == false) {
            return EMPTY_STRING;
        }
    }

    return this->currentName();
}

const std::string& CJsonStateRestoreTraverser::value(void) const {
    if (!m_Started) {
        if (const_cast<CJsonStateRestoreTraverser*>(this)->start() == false) {
            return EMPTY_STRING;
        }
    }

    return this->currentValue();
}

bool CJsonStateRestoreTraverser::descend(void) {
    if (!m_Started) {
        if (this->start() == false) {
            return false;
        }
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

bool CJsonStateRestoreTraverser::ascend(void) {
    // If we're trying to ascend above the root level then something has gone
    // wrong
    if (m_DesiredLevel == 0) {
        LOG_ERROR("Inconsistency - trying to ascend above JSON root");
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

void CJsonStateRestoreTraverser::debug(void) const {
    LOG_DEBUG("Current: name = " << this->currentName() << " value = " << this->currentValue()
                                 << " level = " << this->currentLevel() << ", Next: name = " << this->nextName()
                                 << " value = " << this->nextValue() << " level = " << this->nextLevel()
                                 << " is array of objects = " << m_IsArrayOfObjects);
}

size_t CJsonStateRestoreTraverser::currentLevel(void) const {
    return m_Handler.s_Level[1 - m_Handler.s_NextIndex];
}

bool CJsonStateRestoreTraverser::currentIsEndOfLevel(void) const {
    return m_Handler.s_IsEndOfLevel[1 - m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::currentName(void) const {
    return m_Handler.s_Name[1 - m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::currentValue(void) const {
    return m_Handler.s_Value[1 - m_Handler.s_NextIndex];
}

size_t CJsonStateRestoreTraverser::nextLevel(void) const {
    return m_Handler.s_Level[m_Handler.s_NextIndex];
}

bool CJsonStateRestoreTraverser::nextIsEndOfLevel(void) const {
    return m_Handler.s_IsEndOfLevel[m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::nextName(void) const {
    return m_Handler.s_Name[m_Handler.s_NextIndex];
}

const std::string& CJsonStateRestoreTraverser::nextValue(void) const {
    return m_Handler.s_Value[m_Handler.s_NextIndex];
}

bool CJsonStateRestoreTraverser::parseNext(bool remember) {
    if (m_Reader.HasParseError()) {
        this->logError();
        return false;
    }

    const int parseFlags = rapidjson::kParseDefaultFlags;
    m_Handler.s_RememberValue = remember;

    return m_Reader.IterativeParseNext<parseFlags>(m_ReadStream, m_Handler);
}

bool CJsonStateRestoreTraverser::skipArray() {
    int depth = 0;

    // we must have received a key, revert the state change to ignore it
    m_Handler.s_NextIndex = 1 - m_Handler.s_NextIndex;

    do {
        if (m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayStart ||
            m_Handler.s_Type == SRapidJsonHandler::E_TokenObjectStart) {
            ++depth;
        } else if (m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayEnd ||
                   m_Handler.s_Type == SRapidJsonHandler::E_TokenObjectEnd) {
            --depth;
        }

        if (parseNext(depth == 0) == false) {
            this->logError();
            return false;
        }
    } while (depth > 0);
    return true;
}

bool CJsonStateRestoreTraverser::start(void) {
    m_Started = true;
    m_Reader.IterativeParseInit();

    if (this->parseNext(false) == false) {
        this->logError();
        return false;
    }

    // If the first token is start of array then this could be
    // an array of docs. Next should be start object
    if (m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayStart) {
        if (this->parseNext(false) == false) {
            this->logError();
            return false;
        }

        m_IsArrayOfObjects = true;
    }

    // For Ml state the first token should be the start of a JSON
    // object, but we don't store it
    if (m_Handler.s_Type != SRapidJsonHandler::E_TokenObjectStart) {
        if (m_IsArrayOfObjects && m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayEnd && this->isEof()) {
            LOG_DEBUG("JSON document is an empty array");
            return false;
        }

        LOG_ERROR("JSON state must be object at root" << m_Handler.s_Type);
        return false;
    }

    // Advance twice to prime the current and next elements
    return this->advance() && this->advance();
}

bool CJsonStateRestoreTraverser::advance() {
    bool keepGoing(true);

    while (keepGoing) {
        if (this->parseNext(true) == false) {
            if (!this->isEof()) {
                this->logError();
            }
            return false;
        }

        if (m_Handler.s_Type == SRapidJsonHandler::E_TokenArrayStart) {
            LOG_ERROR("JSON state should not contain arrays");
            this->skipArray();
        } else if (m_Handler.s_Type != SRapidJsonHandler::E_TokenKey) {
            keepGoing = false;
        }
    }

    return true;
}

void CJsonStateRestoreTraverser::logError(void) {
    const char* error(rapidjson::GetParseError_En(m_Reader.GetParseErrorCode()));
    LOG_ERROR("Error parsing JSON at offset " << m_Reader.GetErrorOffset() << ": "
                                              << ((error != 0) ? error : "No message"));
    this->setBadState();
}

CJsonStateRestoreTraverser::SRapidJsonHandler::SRapidJsonHandler()
    : s_Type(SRapidJsonHandler::E_TokenNull), s_NextIndex(0), s_RememberValue(false) {
    s_Level[0] = 0;
    s_Level[1] = 0;
    s_IsEndOfLevel[0] = false;
    s_IsEndOfLevel[1] = false;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Null() {
    s_Type = E_TokenNull;
    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Bool(bool b) {
    s_Type = E_TokenBool;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(b));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Int(int i) {
    s_Type = E_TokenInt;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(i));
    }
    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Uint(unsigned u) {
    s_Type = E_TokenUInt;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(u));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Int64(int64_t i) {
    s_Type = E_TokenInt64;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(i));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Uint64(uint64_t u) {
    s_Type = E_TokenUInt64;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(u));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Double(double d) {
    s_Type = E_TokenDouble;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(CStringUtils::typeToString(d));
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::RawNumber(const char*, rapidjson::SizeType, bool) {
    return false;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::String(const char* str, rapidjson::SizeType length, bool) {
    s_Type = E_TokenString;
    if (s_RememberValue) {
        s_Value[s_NextIndex].assign(str, length);
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::StartObject() {
    s_Type = E_TokenObjectStart;
    if (s_RememberValue) {
        ++s_Level[s_NextIndex];
        s_Value[s_NextIndex].clear();
    }
    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::Key(const char* str, rapidjson::SizeType length, bool) {
    s_Type = E_TokenKey;
    if (s_RememberValue) {
        s_NextIndex = 1 - s_NextIndex;
        s_Level[s_NextIndex] = s_Level[1 - s_NextIndex];
        s_IsEndOfLevel[s_NextIndex] = false;
        s_Name[s_NextIndex].assign(str, length);
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::EndObject(rapidjson::SizeType) {
    s_Type = E_TokenObjectEnd;

    if (s_RememberValue) {
        s_NextIndex = 1 - s_NextIndex;
        s_Level[s_NextIndex] = s_Level[1 - s_NextIndex] - 1;
        s_IsEndOfLevel[s_NextIndex] = true;
        s_Name[s_NextIndex].clear();
        s_Value[s_NextIndex].clear();
    }

    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::StartArray() {
    s_Type = E_TokenArrayStart;
    return true;
}

bool CJsonStateRestoreTraverser::SRapidJsonHandler::EndArray(rapidjson::SizeType) {
    s_Type = E_TokenArrayEnd;
    return true;
}
}
}
