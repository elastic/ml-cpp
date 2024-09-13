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

#include "CResultWriter.h"

#include <core/CBoostJsonConcurrentLineWriter.h>
#include <core/CStringBufWriter.h>

#include "CCommandParser.h"
#include "CThreadSettings.h"

namespace ml {
namespace torch {

const std::string CResultWriter::RESULT{"result"};
const std::string CResultWriter::INFERENCE{"inference"};
const std::string CResultWriter::ERROR{"error"};
const std::string CResultWriter::TIME_MS{"time_ms"};
const std::string CResultWriter::CACHE_HIT{"cache_hit"};
const std::string CResultWriter::THREAD_SETTINGS{"thread_settings"};
const std::string CResultWriter::ACK{"ack"};
const std::string CResultWriter::ACKNOWLEDGED{"acknowledged"};
const std::string CResultWriter::NUM_ALLOCATIONS{"num_allocations"};
const std::string CResultWriter::NUM_THREADS_PER_ALLOCATION{"num_threads_per_allocation"};
const std::string CResultWriter::PROCESS_STATS{"process_stats"};
const std::string CResultWriter::MEMORY_RESIDENT_SET_SIZE{"memory_rss"};
const std::string CResultWriter::MEMORY_MAX_RESIDENT_SET_SIZE{"memory_max_rss"};

CResultWriter::CResultWriter(std::ostream& strmOut)
    : m_WrappedOutputStream{strmOut} {
}

void CResultWriter::writeInnerError(const std::string& message, TStringBufWriter& jsonWriter) {
    jsonWriter.onKey(ERROR);
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(ERROR);
    jsonWriter.onString(message);
    jsonWriter.onObjectEnd();
}

void CResultWriter::writeError(const std::string_view& requestId, const std::string& message) {
    core::CBoostJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(CCommandParser::REQUEST_ID);
    jsonWriter.onString(requestId);
    writeInnerError(message, jsonWriter);
    jsonWriter.onObjectEnd();
}

void CResultWriter::wrapAndWriteInnerResponse(const std::string& innerResponse,
                                              const std::string& requestId,
                                              bool isCacheHit,
                                              std::uint64_t timeMs) {
    core::CBoostJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(CCommandParser::REQUEST_ID);
    jsonWriter.onString(requestId);
    jsonWriter.onKey(CACHE_HIT);
    jsonWriter.onBool(isCacheHit);
    jsonWriter.onKey(TIME_MS);
    jsonWriter.onUint64(timeMs);
    jsonWriter.rawKeyAndValue(innerResponse);
    jsonWriter.onObjectEnd();
}

void CResultWriter::writeThreadSettings(const std::string_view& requestId,
                                        const CThreadSettings& threadSettings) {
    core::CBoostJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(CCommandParser::REQUEST_ID);
    jsonWriter.onString(requestId);
    jsonWriter.onKey(THREAD_SETTINGS);
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(NUM_THREADS_PER_ALLOCATION);
    jsonWriter.onUint(threadSettings.numThreadsPerAllocation());
    jsonWriter.onKey(NUM_ALLOCATIONS);
    jsonWriter.onUint(threadSettings.numAllocations());
    jsonWriter.onObjectEnd();
    jsonWriter.onObjectEnd();
}

void CResultWriter::writeSimpleAck(const std::string_view& requestId) {
    core::CBoostJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.onString(requestId);
    jsonWriter.onKey(ACK);
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(ACKNOWLEDGED);
    jsonWriter.onBool(true);
    jsonWriter.onObjectEnd();
    jsonWriter.onObjectEnd();
}

void CResultWriter::writeProcessStats(const std::string_view& requestId,
                                      const std::size_t residentSetSize,
                                      const std::size_t maxResidentSetSize) {
    core::CBoostJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(CCommandParser::REQUEST_ID);
    jsonWriter.onString(requestId);
    jsonWriter.onKey(PROCESS_STATS);
    jsonWriter.onObjectBegin();
    jsonWriter.onKey(MEMORY_RESIDENT_SET_SIZE);
    jsonWriter.onUint64(residentSetSize);
    jsonWriter.onKey(MEMORY_MAX_RESIDENT_SET_SIZE);
    jsonWriter.onUint64(maxResidentSetSize);
    jsonWriter.onObjectEnd();
    jsonWriter.onObjectEnd();
}

std::string CResultWriter::createInnerResult(const ::torch::Tensor& results) {
    std::string stringBuffer;
    {
        TStringBufWriter jsonWriter{stringBuffer};

        // Even though we don't really want the outer braces on the
        // inner result we have to write them or else the JSON
        // writer will not put commas in the correct places.
        jsonWriter.onObjectBegin();
        try {
            auto sizes = results.sizes();

            switch (sizes.size()) {
            case 3:
                this->writePrediction<3>(results, jsonWriter);
                break;
            case 2:
                this->writePrediction<2>(results, jsonWriter);
                break;
            case 1:
                this->writePrediction<1>(results, jsonWriter);
                break;
            default: {
                std::ostringstream ss;
                ss << "Cannot convert results tensor of size [" << sizes << ']';
                writeInnerError(ss.str(), jsonWriter);
                break;
            }
            }
        } catch (const c10::Error& e) {
            writeInnerError(e.what(), jsonWriter);
        } catch (const std::runtime_error& e) {
            writeInnerError(e.what(), jsonWriter);
        }
        jsonWriter.onObjectEnd();
    }
    // Return the object without the opening and closing braces and
    // the trailing newline. The resulting partial document will
    // later be wrapped, so does not need these.
    return stringBuffer.substr(1, stringBuffer.size() - 3);
}
}
}
