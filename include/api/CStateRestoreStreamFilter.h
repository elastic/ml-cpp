/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CRestoreStreamFilter_h
#define INCLUDED_ml_api_CRestoreStreamFilter_h

#include <api/ImportExport.h>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/line.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <string>

namespace ml {
namespace api {

//! \brief
//! A streaming filter that maps persistence to restore format, specific to
//! ES requirements.
//!
//! DESCRIPTION:\n
//! The restore format differs from the persistence format as it is optimized
//! for bulk indexing into ES.
//! Persist format is:
//! { bulk metadata }
//! { document source }
//! '\0'
//!
//! Restore format is:
//! { Elasticsearch get response }
//! '\0'
//!
//! IMPLEMENTATION DECISIONS:\n
//! Using boost::iostreams for a real streaming implementation to avoid large allocations.
//!
//! If the filter is applied on data in the restore format it does not change it.
//!
//! NOTE:\n
//! When using it with boost::iostreams::filtering_ostream note that the filters gets
//! copied once pushed to the ostream instance.
//!
class API_EXPORT CStateRestoreStreamFilter
    : public boost::iostreams::basic_line_filter<char> {
public:
    using boost::iostreams::basic_line_filter<char>::string_type;

    CStateRestoreStreamFilter();

    size_t getDocCount() const;

private:
    //! number of documents found in the stream
    size_t m_DocCount;

    //! whether the previous line has been rewritten
    bool m_RewrotePreviousLine;

    string_type do_filter(const string_type& line) override;
};
}
}

#endif /* INCLUDED_ml_api_CRestoreStreamFilter_h */
