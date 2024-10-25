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

#ifndef INCLUDED_ml_torch_CResultWriter_h
#define INCLUDED_ml_torch_CResultWriter_h

#include <core/CBoostJsonConcurrentLineWriter.h>
#include <core/CBoostJsonLineWriter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CStringBufWriter.h>

#include <c10/util/BFloat16.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/script.h>

#include <boost/json.hpp>

#include <cstdint>
#include <iosfwd>
#include <sstream>
#include <string>

namespace ml {
namespace torch {
class CThreadSettings;

//! \brief
//! Formats and writes results for PyTorch inference.
//!
//! DESCRIPTION:\n
//! There are four types of result:
//!
//! 1. Inference results
//! 2. Thread settings
//! 3. Acknowledgements
//! 4. Errors
//!
//! IMPLEMENTATION DECISIONS:\n
//! We can cache inference results and errors, but when we reply with a
//! cached value we still need to change the request ID, time taken, and
//! cache hit indicator. Therefore this class contains functionality for
//! building the invariant portion of results to be cached and later
//! spliced into a complete response.
//!
using TStringBufWriter = ml::core::CStringBufWriter;
class CResultWriter : public TStringBufWriter {
public:
    using TBoostJsonLineWriter = core::CBoostJsonLineWriter<std::string>;

public:
    explicit CResultWriter(std::ostream& strmOut);

    //! No copying
    CResultWriter(const CResultWriter&) = delete;
    CResultWriter& operator=(const CResultWriter&) = delete;

    //! Write an error directly to the output stream.
    void writeError(const std::string_view& requestId, const std::string& message);

    //! Write thread settings to the output stream.
    void writeThreadSettings(const std::string_view& requestId,
                             const CThreadSettings& threadSettings);

    //! Write a simple acknowledgement to the output stream.
    void writeSimpleAck(const std::string_view& requestId);

    //! Write memory usage information to the output stream.
    void writeProcessStats(const std::string_view& requestId,
                           const std::size_t residentSetSize,
                           const std::size_t maxResidentSetSize);

    //! Wrap the invariant portion of a cached result with request ID,
    //! cache hit indicator and time taken. Then write the full document
    //! to the output stream.
    void wrapAndWriteInnerResponse(const std::string& innerResponse,
                                   const std::string& requestId,
                                   bool isCacheHit,
                                   std::uint64_t timeMs);

    //! Write the prediction portion of an inference result.
    template<std::size_t N>
    void writePrediction(const ::torch::Tensor& prediction, TStringBufWriter& jsonWriter) {

        // Creating the accessor will throw if the tensor does not have exactly
        // N dimensions. Do this before writing any output so the error message
        // isn't mingled with a partial result.

        if (prediction.dtype() == ::torch::kFloat32) {
            auto accessor = prediction.accessor<float, N>();
            this->writeInferenceResults(accessor, jsonWriter);

        } else if (prediction.dtype() == ::torch::kFloat64) {
            auto accessor = prediction.accessor<double, N>();
            this->writeInferenceResults(accessor, jsonWriter);

        } else if (prediction.dtype() == ::c10::kBFloat16) {
            // cast the tensor to float32 and then write it
            auto float32Tensor = prediction.toType(::torch::kFloat32);
            auto accessor = float32Tensor.accessor<float, N>();
            this->writeInferenceResults(accessor, jsonWriter);

        } else {
            std::ostringstream ss;
            ss << "Cannot process result tensor of type [" << prediction.dtype() << ']';
            writeInnerError(ss.str(), jsonWriter);
        }
    }

    //! Create the invariant portion of an inference result, suitable for
    //! caching and later splicing into a full result.
    std::string createInnerResult(const ::torch::Tensor& results);

private:
    //! Field names.
    static const std::string RESULT;
    static const std::string INFERENCE;
    static const std::string ERROR;
    static const std::string TIME_MS;
    static const std::string CACHE_HIT;
    static const std::string THREAD_SETTINGS;
    static const std::string ACK;
    static const std::string ACKNOWLEDGED;
    static const std::string NUM_ALLOCATIONS;
    static const std::string NUM_THREADS_PER_ALLOCATION;
    static const std::string PROCESS_STATS;
    static const std::string MEMORY_RESIDENT_SET_SIZE;
    static const std::string MEMORY_MAX_RESIDENT_SET_SIZE;

private:
    //! Create the invariant portion of an error result, suitable for
    //! caching and later splicing into a full result.
    //core::CBoostJsonConcurrentLineWriter
    //    static void writeInnerError(const std::string& message, TStringBufWriter& jsonWriter);
    static void writeInnerError(const std::string& message, TStringBufWriter& jsonWriter);

    //! Write a one dimensional tensor.
    template<typename T>
    void writeTensor(const ::torch::TensorAccessor<T, 1UL>& accessor,
                     TStringBufWriter& jsonWriter) {
        jsonWriter.onArrayBegin();
        for (int i = 0; i < accessor.size(0); ++i) {
            jsonWriter.onDouble(static_cast<double>(accessor[i]));
        }
        jsonWriter.onArrayEnd();
    }

    //! Write an N dimensional tensor for N > 1.
    template<typename T, std::size_t N_DIMS>
    void writeTensor(const ::torch::TensorAccessor<T, N_DIMS>& accessor,
                     TStringBufWriter& jsonWriter) {
        jsonWriter.onArrayBegin();
        for (int i = 0; i < accessor.size(0); ++i) {
            this->writeTensor(accessor[i], jsonWriter);
        }
        jsonWriter.onArrayEnd();
    }

    //! Write a 3D inference result
    template<typename T>
    void writeInferenceResults(const ::torch::TensorAccessor<T, 3UL>& accessor,
                               TStringBufWriter& jsonWriter) {

        jsonWriter.onKey(RESULT);
        jsonWriter.onObjectBegin();
        jsonWriter.onKey(INFERENCE);
        this->writeTensor(accessor, jsonWriter);
        jsonWriter.onObjectEnd();
    }

    //! Write a 2D inference result
    template<typename T>
    void writeInferenceResults(const ::torch::TensorAccessor<T, 2UL>& accessor,
                               TStringBufWriter& jsonWriter) {

        jsonWriter.onKey(RESULT);
        jsonWriter.onObjectBegin();
        jsonWriter.onKey(INFERENCE);
        // The Java side requires a 3D array, so wrap the 2D result in an
        // extra outer array.
        jsonWriter.onArrayBegin();
        this->writeTensor(accessor, jsonWriter);
        jsonWriter.onArrayEnd();
        jsonWriter.onObjectEnd();
    }

    //! Write a 1D inference result
    template<typename T>
    void writeInferenceResults(const ::torch::TensorAccessor<T, 1UL>& accessor,
                               TStringBufWriter& jsonWriter) {

        jsonWriter.onKey(RESULT);
        jsonWriter.onObjectBegin();
        jsonWriter.onKey(INFERENCE);
        // The Java side requires a 3D array, so wrap the 1D result in an
        // extra outer array twice.
        jsonWriter.onArrayBegin();
        jsonWriter.onArrayBegin();
        this->writeTensor(accessor, jsonWriter);
        jsonWriter.onArrayEnd();
        jsonWriter.onArrayEnd();
        jsonWriter.onObjectEnd();
    }

private:
    core::CJsonOutputStreamWrapper m_WrappedOutputStream;
};
}
}

#endif // INCLUDED_ml_torch_CResultWriter_h
