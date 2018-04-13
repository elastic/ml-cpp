/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CNullOutput_h
#define INCLUDED_ml_api_CNullOutput_h

#include <api/COutputHandler.h>
#include <api/ImportExport.h>


namespace ml
{
namespace api
{

//! \brief
//! Output handler that ignores all output.
//!
//! DESCRIPTION:\n
//! An output handler where all operations are no-ops.
//! This can be used to terminate a chain of data processors
//! where the final output is consumed in some non-standard
//! way.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class is necessary because the categoriser outputs
//! its results via a reference to the JSON output writer
//! that is separate to the output handler that all data
//! processors have.
//!
class API_EXPORT CNullOutput : public COutputHandler
{
    public:
        //! Does nothing with the field names provided.
        virtual bool fieldNames(const TStrVec &fieldNames,
                                const TStrVec &extraFieldNames);

        //! Get field names - always empty.
        virtual const TStrVec &fieldNames() const;

        // Bring the other overload of fieldNames() into scope
        using COutputHandler::fieldNames;

        //! Does nothing with the row provided.
        virtual bool writeRow(const TStrStrUMap &dataRowFields,
                              const TStrStrUMap &overrideDataRowFields);

        // Bring the other overload of writeRow() into scope
        using COutputHandler::writeRow;
};


}
}

#endif // INCLUDED_ml_api_CNullOutput_h

