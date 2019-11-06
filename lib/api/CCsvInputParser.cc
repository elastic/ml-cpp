/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CCsvInputParser.h>

#include <core/CLogger.h>
#include <core/CTimeUtils.h>

#include <algorithm>
#include <istream>

#include <string.h>

namespace ml {
namespace api {

// Initialise statics
const char CCsvInputParser::RECORD_END('\n');
const char CCsvInputParser::STRIP_BEFORE_END('\r');
const size_t CCsvInputParser::WORK_BUFFER_SIZE(131072); // 128kB

CCsvInputParser::CCsvInputParser(const std::string& input, char separator)
    : CInputParser(), m_StringInputBuf(input), m_StrmIn(m_StringInputBuf),
      m_WorkBuffer(nullptr), m_WorkBufferPtr(nullptr), m_WorkBufferEnd(nullptr),
      m_NoMoreRecords(false), m_LineParser(separator) {
}

CCsvInputParser::CCsvInputParser(std::istream& strmIn, char separator)
    : CInputParser(), m_StrmIn(strmIn), m_WorkBuffer(nullptr),
      m_WorkBufferPtr(nullptr), m_WorkBufferEnd(nullptr),
      m_NoMoreRecords(false), m_LineParser(separator) {
}

const std::string& CCsvInputParser::fieldNameStr() const {
    return m_FieldNameStr;
}

bool CCsvInputParser::readStreamIntoMaps(const TMapReaderFunc& readerFunc) {

    if (this->readFieldNames() == false) {
        return false;
    }

    TStrVec& fieldNames = this->fieldNames();

    // We reuse the same field map for every record
    TStrStrUMap recordFields;

    // Cache references to the strings in the map corresponding to each field
    // name - this avoids the need to repeatedly compute the same hashes
    TStrRefVec fieldValRefs;
    fieldValRefs.reserve(fieldNames.size());
    for (const auto& name : fieldNames) {
        fieldValRefs.emplace_back(recordFields[name]);
    }

    return this->parseRecordLoop(
        [&readerFunc, &recordFields] { return readerFunc(recordFields); }, fieldValRefs);
}

bool CCsvInputParser::readStreamIntoVecs(const TVecReaderFunc& readerFunc) {

    if (this->readFieldNames() == false) {
        return false;
    }

    TStrVec& fieldNames = this->fieldNames();

    // We reuse the same value vector for every record
    TStrVec fieldValues(fieldNames.size());

    return parseRecordLoop(
        [&readerFunc, &fieldNames, &fieldValues] {
            return readerFunc(fieldNames, fieldValues);
        },
        fieldValues);
}

bool CCsvInputParser::readFieldNames() {
    // Reset the record buffer pointers in case we're reading a new stream
    m_WorkBufferEnd = m_WorkBufferPtr;
    m_NoMoreRecords = false;

    if (this->gotFieldNames() == false) {
        if (this->parseCsvRecordFromStream() == false) {
            LOG_ERROR(<< "Failed to parse CSV record from stream");
            return false;
        }

        if (this->parseFieldNames() == false) {
            LOG_ERROR(<< "Failed to parse field names from stream");
            return false;
        }
    }
    return true;
}

template<typename READER_FUNC, typename STR_VEC>
bool CCsvInputParser::parseRecordLoop(const READER_FUNC& readerFunc, STR_VEC& workSpace) {

    while (!m_NoMoreRecords) {
        if (this->parseCsvRecordFromStream() == false) {
            LOG_ERROR(<< "Failed to parse CSV record from stream");
            return false;
        }

        if (m_NoMoreRecords) {
            break;
        }

        if (this->parseDataRecord(workSpace) == false) {
            LOG_ERROR(<< "Failed to parse data record from stream");
            return false;
        }

        if (readerFunc() == false) {
            LOG_ERROR(<< "Record handler function forced exit");
            return false;
        }
    }

    return true;
}

bool CCsvInputParser::parseCsvRecordFromStream() {
    // For maximum performance, read the stream in large chunks that can be
    // moved around by memcpy().  Using memcpy() is an order of magnitude faster
    // than the naive approach of checking and copying one character at a time.
    // In modern versions of the GNU STL std::getline uses memchr() to search
    // for the delimiter and then memcpy() to transfer data to the target
    // std::string, but sadly this is not the case for the Microsoft and Apache
    // STLs.
    if (m_WorkBuffer == nullptr) {
        m_WorkBuffer.reset(new char[WORK_BUFFER_SIZE]);
        m_WorkBufferPtr = m_WorkBuffer.get();
        m_WorkBufferEnd = m_WorkBufferPtr;
    }

    bool startOfRecord(true);
    size_t quoteCount(0);
    for (;;) {
        size_t avail(m_WorkBufferEnd - m_WorkBufferPtr);
        if (avail == 0) {
            if (m_StrmIn.eof()) {
                // We have no buffered data and there's no more to read, so stop
                m_NoMoreRecords = true;
                break;
            }

            m_WorkBufferPtr = m_WorkBuffer.get();
            m_StrmIn.read(m_WorkBuffer.get(), static_cast<std::streamsize>(WORK_BUFFER_SIZE));
            if (m_StrmIn.bad()) {
                LOG_ERROR(<< "Input stream is bad");
                m_CurrentRowStr.clear();
                m_WorkBufferEnd = m_WorkBufferPtr;
                return false;
            }

            avail = static_cast<size_t>(m_StrmIn.gcount());
            m_WorkBufferEnd = m_WorkBufferPtr + avail;
        }

        const char* delimPtr(reinterpret_cast<const char*>(
            ::memchr(m_WorkBufferPtr, RECORD_END, avail)));
        const char* endPtr(m_WorkBufferEnd);
        if (delimPtr != nullptr) {
            endPtr = delimPtr;
            if (endPtr > m_WorkBufferPtr && *(endPtr - 1) == STRIP_BEFORE_END) {
                --endPtr;
            }
        }

        if (startOfRecord) {
            m_CurrentRowStr.assign(m_WorkBufferPtr, endPtr - m_WorkBufferPtr);
            startOfRecord = false;
        } else {
            if (endPtr == m_WorkBufferPtr) {
                size_t strLen(m_CurrentRowStr.length());
                if (strLen > 0 && m_CurrentRowStr[strLen - 1] == STRIP_BEFORE_END) {
                    m_CurrentRowStr.erase(strLen - 1);
                }
            } else {
                m_CurrentRowStr.append(m_WorkBufferPtr, endPtr - m_WorkBufferPtr);
            }
        }

        quoteCount += std::count(m_WorkBufferPtr, endPtr, core::CCsvLineParser::QUOTE);
        if (delimPtr != nullptr) {
            m_WorkBufferPtr = delimPtr + 1;

            // In Excel style CSV, quote characters are escaped by doubling them
            // up.  Therefore, if what we've read of a record up to now contains
            // an odd number of quote characters then we need to read more.
            if ((quoteCount % 2) == 0) {
                break;
            }
            m_CurrentRowStr += RECORD_END;
        } else {
            m_WorkBufferPtr = m_WorkBufferEnd;
        }
    }

    m_LineParser.reset(m_CurrentRowStr);

    return true;
}

bool CCsvInputParser::parseFieldNames() {
    LOG_TRACE(<< "Parse field names");

    m_FieldNameStr.clear();
    TStrVec& fieldNames = this->fieldNames();
    fieldNames.clear();

    m_LineParser.reset(m_CurrentRowStr);
    while (!m_LineParser.atEnd()) {
        std::string fieldName;
        if (m_LineParser.parseNext(fieldName) == false) {
            LOG_ERROR(<< "Failed to get next CSV token");
            return false;
        }

        fieldNames.emplace_back(std::move(fieldName));
    }

    if (fieldNames.empty()) {
        // Don't scare the user with error messages if we've just received an
        // empty input
        if (m_NoMoreRecords) {
            LOG_DEBUG(<< "Received input with settings only");
        } else {
            LOG_ERROR(<< "No field names found in:" << core_t::LINE_ENDING << m_CurrentRowStr);
        }
        return false;
    }

    m_FieldNameStr = m_CurrentRowStr;
    this->gotFieldNames(true);

    LOG_TRACE(<< "Field names " << m_FieldNameStr);

    return true;
}

template<typename STR_VEC>
bool CCsvInputParser::parseDataRecord(STR_VEC& values) {
    for (auto& value : values) {
        if (m_LineParser.parseNext(value) == false) {
            LOG_ERROR(<< "Failed to get next CSV token");
            return false;
        }
    }

    if (!m_LineParser.atEnd()) {
        std::string extraField;
        size_t numExtraFields(0);
        while (m_LineParser.parseNext(extraField) == true) {
            ++numExtraFields;
        }
        LOG_ERROR(<< "Data record contains " << numExtraFields << " more fields than header:"
                  << core_t::LINE_ENDING << m_CurrentRowStr << core_t::LINE_ENDING
                  << "and:" << core_t::LINE_ENDING << m_FieldNameStr);
        return false;
    }

    this->gotData(true);

    return true;
}
}
}
