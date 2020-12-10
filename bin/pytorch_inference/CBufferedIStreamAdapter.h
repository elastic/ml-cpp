/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_torch_CBufferedIStreamAdapter_h
#define INCLUDED_ml_torch_CBufferedIStreamAdapter_h

#include <caffe2/serialize/read_adapter_interface.h>

namespace ml {
namespace torch {

class CBufferedIStreamAdapter : public caffe2::serialize::ReadAdapterInterface {    
public:
    CBufferedIStreamAdapter(size_t size, std::istream& inputStream);

    size_t size() const override;
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "") const override;
        
    CBufferedIStreamAdapter(const CBufferedIStreamAdapter&) = delete;
    CBufferedIStreamAdapter& operator=(const CBufferedIStreamAdapter&) = delete;

private:
    size_t m_Size;
    std::unique_ptr<char[]> m_Buffer;
};

}
}

#endif // INCLUDED_ml_torch_CBufferedIStreamAdapter_h
