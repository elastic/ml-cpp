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

#ifndef INCLUDED_ml_torch_CBufferedIStreamAdapter_h
#define INCLUDED_ml_torch_CBufferedIStreamAdapter_h

#include <caffe2/serialize/read_adapter_interface.h>

#include <iosfwd>
#include <memory>

namespace ml {
namespace torch {

//! \brief
//! A buffered stream implementation of ReadAdapterInterface
//! for reading TorchScript models (.pt files).
//!
//! DESCRIPTION:\n
//! TorchScript model readers require seek and tell type
//! functionality which is not provided in all input streams.
//! The entire model is read into a buffer to support this.
//!
//! The max supported model size is 4GB limited by the 32 bit number
//! the size is serialized as.
//!
//! See https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/inline_container.h
//! for details of the serialized TorchScript model format.
//!
//! IMPLEMENTATION DECISIONS:\n
//! First reads the size of the model file from the stream
//! then allocates a buffer large enough to hold the model
//! definition and reads the model into that buffer.
//!
class CBufferedIStreamAdapter : public caffe2::serialize::ReadAdapterInterface {
public:
    explicit CBufferedIStreamAdapter(std::istream& inputStream);

    //! True if the model is successfully read.
    //! Must be called before read or size
    bool init();

    std::size_t size() const override;
    std::size_t read(std::uint64_t pos, void* buf, std::size_t n, const char* what = "") const override;

    CBufferedIStreamAdapter(const CBufferedIStreamAdapter&) = delete;
    CBufferedIStreamAdapter& operator=(const CBufferedIStreamAdapter&) = delete;

private:
    //! Reads a 4 bytes unsigned int from the stream into \p num.
    //! \p num will not be larger than 2^32
    bool parseSizeFromStream(std::size_t& num);

    std::size_t m_Size{0};
    std::unique_ptr<char[]> m_Buffer;
    std::istream& m_InputStream;
};
}
}

#endif // INCLUDED_ml_torch_CBufferedIStreamAdapter_h
