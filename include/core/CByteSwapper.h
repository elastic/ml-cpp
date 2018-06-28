/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CByteSwapper_h
#define INCLUDED_ml_core_CByteSwapper_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <algorithm>

#include <stdint.h>

namespace ml {
namespace core {

//! \brief
//! Swap the order of bytes in a variable.
//!
//! DESCRIPTION:\n
//! Class to convert between big endian and little endian
//! representations of values.  Usually this would be done
//! with functions like ntohs() or htonl(), but there may
//! be times when we want to convert endianness regardless
//! of the network and host orders, e.g. when reading a
//! file from another system.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has the potential to trash memory, e.g. a complex
//! class with multiple data members and/or a vtable.  It's up
//! to the caller to ensure it's only called for types where
//! it's sensible, e.g. builtin numeric types.
//!
class CORE_EXPORT CByteSwapper : private CNonInstantiatable {
public:
    template<typename TYPE>
    static TYPE swapBytes(TYPE var) {
        void* varAddress(&var);
        uint8_t* begin(static_cast<uint8_t*>(varAddress));
        uint8_t* end(begin + sizeof(var));

        std::reverse(begin, end);

        return var;
    }
};
}
}

#endif // INCLUDED_ml_core_CByteSwapper_h
