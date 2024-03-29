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
#include <api/CLengthEncodedInputParser.h>

#include <core/CLogger.h>
#include <core/CSetMode.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <type_traits>

// For ntohl
#ifdef Windows
#include <WinSock2.h>
#else
#include <netinet/in.h>
#endif

namespace ml {
namespace api {

// Initialise statics

// We keep the buffer at 2KB because it has to be smaller
// than the OS buffer for the named pipes (smallest is windows at 4KB).
// This allows flushing the buffer from the java side by sending spaces
// without the risk of blocking the thread that writes to the process.
const std::size_t CLengthEncodedInputParser::WORK_BUFFER_SIZE{2048}; // 2kB

CLengthEncodedInputParser::CLengthEncodedInputParser(std::istream& strmIn)
    : CLengthEncodedInputParser{TStrVec{}, strmIn} {
}

CLengthEncodedInputParser::CLengthEncodedInputParser(TStrVec mutableFieldNames,
                                                     std::istream& strmIn)
    : CInputParser{std::move(mutableFieldNames)}, m_StrmIn{strmIn} {
    // This test is not ideal because std::cin's stream buffer could have been
    // changed
    if (strmIn.rdbuf() == std::cin.rdbuf()) {
        LOG_DEBUG(<< "Length encoded input parser input is connected to stdin");

        int result{core::CSetMode::setBinaryMode(::fileno(stdin))};
        if (result == -1) {
            LOG_WARN(<< "Cannot set the stdin to binary mode");
        }
    } else {
        LOG_DEBUG(<< "Length encoded input parser input is not connected to stdin");
    }
}

bool CLengthEncodedInputParser::readStreamIntoMaps(const TMapReaderFunc& readerFunc,
                                                   const TRegisterMutableFieldFunc& registerFunc) {

    if (this->readFieldNames() == false) {
        return false;
    }

    TStrVec& fieldNames{this->fieldNames()};

    // We reuse the same field map for every record
    TStrStrUMap recordFields;

    // Cache references to the strings in the map corresponding to each field
    // name - this avoids the need to repeatedly compute the same hashes
    TStrRefVec fieldValRefs;
    fieldValRefs.reserve(fieldNames.size());
    for (const auto& fieldName : fieldNames) {
        fieldValRefs.emplace_back(recordFields[fieldName]);
    }

    this->registerMutableFields(registerFunc, recordFields);

    return this->parseRecordLoop(
        [&readerFunc, &recordFields] { return readerFunc(recordFields); }, fieldValRefs);
}

bool CLengthEncodedInputParser::readStreamIntoVecs(const TVecReaderFunc& readerFunc,
                                                   const TRegisterMutableFieldFunc& registerFunc) {

    if (this->readFieldNames() == false) {
        return false;
    }

    TStrVec& fieldNames{this->fieldNames()};
    std::size_t parsedFieldCount{fieldNames.size()};

    // We reuse the same value vector for every record
    TStrVec fieldValues{fieldNames.size()};

    this->registerMutableFields(registerFunc, fieldNames, fieldValues);

    TStrRefVec fieldValRefs{fieldValues.begin(), fieldValues.begin() + parsedFieldCount};

    return parseRecordLoop(
        [&readerFunc, &fieldNames, &fieldValues] {
            return readerFunc(fieldNames, fieldValues);
        },
        fieldValRefs);
}

bool CLengthEncodedInputParser::readFieldNames() {
    // Reset the record buffer pointers in case we're reading a new stream
    m_WorkBufferEnd = m_WorkBufferPtr;
    m_NoMoreRecords = false;
    TStrVec& fieldNames{this->fieldNames()};

    if (this->parseRecordFromStream<true>(fieldNames) == false) {
        LOG_ERROR(<< "Failed to parse length encoded header from stream");
        return false;
    }

    if (fieldNames.empty()) {
        // If we parsed no field names at all, return true, as
        // completely empty input is technically valid
        LOG_INFO(<< "Field names are empty");
        m_NoMoreRecords = true;
        return true;
    }

    return true;
}

template<typename READER_FUNC>
bool CLengthEncodedInputParser::parseRecordLoop(const READER_FUNC& readerFunc,
                                                TStrRefVec& workSpace) {
    while (m_NoMoreRecords == false) {
        if (this->parseRecordFromStream<false>(workSpace) == false) {
            LOG_ERROR(<< "Failed to parse length encoded data record from stream");
            return false;
        }

        if (m_NoMoreRecords) {
            break;
        }

        if (readerFunc() == false) {
            LOG_ERROR(<< "Record handler function forced exit");
            return false;
        }
    }

    return true;
}

template<bool RESIZE_ALLOWED, typename STR_VEC>
bool CLengthEncodedInputParser::parseRecordFromStream(STR_VEC& values) {
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

    std::uint32_t numFields{0};
    if (this->parseUInt32FromStream(numFields) == false) {
        if (m_StrmIn.eof()) {
            // End-of-file is not an error at this point in the parsing
            m_NoMoreRecords = true;
            return true;
        }

        LOG_ERROR(<< "Unable to read field count from input stream");
        return false;
    }

    if (values.size() != numFields) {
        if (RESIZE_ALLOWED) {
            if (numFields == 0) {
                LOG_WARN(<< "Number of fields is 0 in input");
            }

            // The two argument form of resize is used to avoid compiler errors
            // caused by std::reference_wrapper not having a default
            // constructor.  However, it's clearly wrong to create references to
            // a temporary string like this, so it is crucial that the
            // RESIZE_ALLOWED template argument is set to false when STR_VEC is
            // some type of vector of std::reference_wrappers.
            using TVecValue = std::remove_cv_t<typename STR_VEC::value_type>;
            static_assert(!RESIZE_ALLOWED || !std::is_same_v<TVecValue, TStrRef>,
                          "RESIZE_ALLOWED must be false for reference vectors");
            std::string temp;
            values.resize(numFields, typename STR_VEC::value_type(temp));
        } else {
            LOG_ERROR(<< "Incorrect number of fields in input stream record: expected "
                      << values.size() << " but got " << numFields);
            return false;
        }
    }

    for (std::size_t index = 0; index < numFields; ++index) {
        std::uint32_t length{0};
        if (this->parseUInt32FromStream(length) == false) {
            LOG_ERROR(<< "Unable to read field length from input stream");
            return false;
        }

        // If the stream gets corrupted then we may end up parsing string data
        // into the length variable.  If this happens it's highly likely that
        // the high byte of the length variable will be non-zero, as zero bytes
        // are unlikely to occur in strings.  Also, a length where the high byte
        // of the length variable is non-zero implies a field of 16MB or more,
        // which is unlikely, so assume corruption in this case.  See bug 1040
        // in Bugzilla for more details.
        static const std::uint32_t HIGH_BYTE_MASK{0xFF000000};
        if ((length & HIGH_BYTE_MASK) != 0u) {
            LOG_ERROR(<< "Parsed field length " << length
                      << " is suspiciously large - assuming corrupt input stream");
            return false;
        }

        if (this->parseStringFromStream(length, values[index]) == false) {
            LOG_ERROR(<< "Unable to read field data from input stream");
            return false;
        }
    }

    return true;
}

bool CLengthEncodedInputParser::parseUInt32FromStream(std::uint32_t& num) {
    std::ptrdiff_t avail{m_WorkBufferEnd - m_WorkBufferPtr};
    if (avail < static_cast<std::ptrdiff_t>(sizeof(std::uint32_t))) {
        avail = this->refillBuffer();
        if (avail < static_cast<std::ptrdiff_t>(sizeof(std::uint32_t))) {
            return false;
        }
    }

    std::uint32_t netNum{0};
    std::memcpy(&netNum, m_WorkBufferPtr, sizeof(std::uint32_t));
    m_WorkBufferPtr += sizeof(std::uint32_t);

    // Integers are encoded in network byte order, so convert to host byte order
    // before interpreting
    num = ntohl(netNum);

    return true;
}

bool CLengthEncodedInputParser::parseStringFromStream(std::size_t length, std::string& str) {
    if (length == 0) {
        str.clear();
        return true;
    }

    bool append{false};
    std::ptrdiff_t avail{m_WorkBufferEnd - m_WorkBufferPtr};
    do {
        if (avail == 0) {
            avail = this->refillBuffer();
            if (avail == 0) {
                return false;
            }
        }

        std::size_t copyLen{std::min(length, static_cast<std::size_t>(avail))};
        if (append) {
            str.append(m_WorkBufferPtr, copyLen);
        } else {
            str.assign(m_WorkBufferPtr, copyLen);
            append = true;
        }
        m_WorkBufferPtr += copyLen;
        avail -= copyLen;
        length -= copyLen;
    } while (length > 0);

    return true;
}

std::ptrdiff_t CLengthEncodedInputParser::refillBuffer() {
    // NB: This assumes the buffer is allocated, which is OK for a private
    // method.  Callers are responsible for ensuring that the buffer isn't NULL
    // when calling this method.

    std::ptrdiff_t avail{m_WorkBufferEnd - m_WorkBufferPtr};
    if (m_StrmIn.eof()) {
        // We can't read any more data - whatever's available now won't change
        return avail;
    }

    if (avail > 0) {
        std::memcpy(m_WorkBuffer.get(), m_WorkBufferPtr, avail);
    }

    m_WorkBufferPtr = m_WorkBuffer.get();
    m_StrmIn.read(m_WorkBuffer.get() + avail,
                  static_cast<std::streamsize>(WORK_BUFFER_SIZE - avail));
    if (m_StrmIn.bad()) {
        LOG_ERROR(<< "Input stream is bad");
    } else {
        avail += static_cast<std::ptrdiff_t>(m_StrmIn.gcount());
    }
    m_WorkBufferEnd = m_WorkBufferPtr + avail;

    return avail;
}
}
}
