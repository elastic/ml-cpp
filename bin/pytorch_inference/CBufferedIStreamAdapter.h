/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_torch_CBufferedIStreamAdapter_h
#define INCLUDED_ml_torch_CBufferedIStreamAdapter_h

#include <core/CNamedPipeFactory.h>

#include <caffe2/serialize/read_adapter_interface.h>


namespace ml {
namespace torch {

class CBufferedIStreamAdapter : public caffe2::serialize::ReadAdapterInterface {    
public:
    CBufferedIStreamAdapter(core::CNamedPipeFactory::TIStreamP inputStream);

    std::size_t size() const override;
    std::size_t read(uint64_t pos, void* buf, std::size_t n, const char* what = "") const override;

    char* buffer() const;
        
    CBufferedIStreamAdapter(const CBufferedIStreamAdapter&) = delete;
    CBufferedIStreamAdapter& operator=(const CBufferedIStreamAdapter&) = delete;

private:
	bool parseSizeFromStream(std::size_t& num, core::CNamedPipeFactory::TIStreamP inputStream);

    std::size_t m_Size;
    std::unique_ptr<char[]> m_Buffer;
};

}
}

#endif // INCLUDED_ml_torch_CBufferedIStreamAdapter_h
