/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CStreamUtils_h
#define INCLUDED_ml_core_CStreamUtils_h

#include <iosfwd>

namespace ml
{
namespace core
{

//! \brief Stream utility functions.
class CStreamUtils
{
    public:
        //! boost::ini_parser doesn't like UTF-8 ini files that begin
        //! with byte order markers.  This function advances the seek
        //! pointer of the stream over a UTF-8 BOM, but only if one
        //! exists.
        static void skipUtf8Bom(std::ifstream &strm);
};

}
}

#endif // INCLUDED_ml_core_CStreamUtils_h
