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

#include <core/CRapidJsonConcurrentLineWriter.h>

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

CResultWriter::CResultWriter(std::ostream& strmOut)
    : m_WrappedOutputStream{strmOut} {
}

void CResultWriter::writeInnerError(const std::string& message,
                                    TRapidJsonLineWriter& jsonWriter) {
    jsonWriter.Key(ERROR);
    jsonWriter.StartObject();
    jsonWriter.Key(ERROR);
    jsonWriter.String(message);
    jsonWriter.EndObject();
}

void CResultWriter::writeError(const std::string& requestId, const std::string& message) {
    core::CRapidJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.StartObject();
    jsonWriter.Key(CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    writeInnerError(message, jsonWriter);
    jsonWriter.EndObject();
}

void CResultWriter::wrapAndWriteInnerResponse(const std::string& innerResponse,
                                              const std::string& requestId,
                                              bool isCacheHit,
                                              std::uint64_t timeMs) {
    core::CRapidJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.StartObject();
    jsonWriter.Key(CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(CACHE_HIT);
    jsonWriter.Bool(isCacheHit);
    jsonWriter.Key(TIME_MS);
    jsonWriter.Uint64(timeMs);
    jsonWriter.RawValue(innerResponse.c_str(), innerResponse.length(), rapidjson::kObjectType);
    jsonWriter.EndObject();
}

void CResultWriter::writeThreadSettings(const std::string& requestId,
                                        const CThreadSettings& threadSettings) {
    core::CRapidJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.StartObject();
    jsonWriter.Key(CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(THREAD_SETTINGS);
    jsonWriter.StartObject();
    jsonWriter.Key(NUM_THREADS_PER_ALLOCATION);
    jsonWriter.Uint(threadSettings.numThreadsPerAllocation());
    jsonWriter.Key(NUM_ALLOCATIONS);
    jsonWriter.Uint(threadSettings.numAllocations());
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

void CResultWriter::writeSimpleAck(const std::string& requestId) {
    core::CRapidJsonConcurrentLineWriter jsonWriter{m_WrappedOutputStream};
    jsonWriter.StartObject();
    jsonWriter.Key(ml::torch::CCommandParser::REQUEST_ID);
    jsonWriter.String(requestId);
    jsonWriter.Key(ACK);
    jsonWriter.StartObject();
    jsonWriter.Key(ACKNOWLEDGED);
    jsonWriter.Bool(true);
    jsonWriter.EndObject();
    jsonWriter.EndObject();
}

std::string CResultWriter::createInnerResult(const ::torch::Tensor& results) {
    rapidjson::StringBuffer stringBuffer;
    {
        TRapidJsonLineWriter jsonWriter{stringBuffer};
        // Even though we don't really want the outer braces on the
        // inner result we have to write them or else the JSON
        // writer will not put commas in the correct places.
        jsonWriter.StartObject();
        try {
            auto sizes = results.sizes();

            switch (sizes.size()) {
            case 3:
                this->writePrediction<3>(results, jsonWriter);
                break;
            case 2:
                this->writePrediction<2>(results, jsonWriter);
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
        jsonWriter.EndObject();
    }
    // Return the object without the opening and closing braces and
    // the trailing newline. The resulting partial document will
    // later be wrapped, so does not need these.
    return std::string{stringBuffer.GetString() + 1, stringBuffer.GetLength() - 3};
}
}
}
