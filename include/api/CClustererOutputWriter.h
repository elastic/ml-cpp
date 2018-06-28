/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CClustererOutputWriter_h
#define INCLUDED_ml_api_CClustererOutputWriter_h

#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <boost/shared_ptr.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>

namespace ml {
namespace api {

//! \brief Writes out clustering results.
class API_EXPORT CClustererOutputWriter : public COutputHandler {
public:
    using TDoubleVec = std::vector<double>;
    using TStrDoubleUMap = boost::unordered_map<std::string, double>;

public:
    CClustererOutputWriter(std::ostream& stream);

    //! Start building a new result.
    void startResult(void);
    //! Add a named boolean member.
    void addMember(const std::string& name, bool value);
    //! Add a named double member.
    void addMember(const std::string& name, double value);
    //! Add a named std::size_t member.
    void addMember(const std::string& name, std::size_t value);
    //! Add a named std::string member.
    void addMember(const std::string& name, const std::string& value);
    //! Add a named double array member.
    void addMember(const std::string& name, const TDoubleVec& values);
    //! Add a named string array member.
    void addMember(const std::string& name, const TStrVec& values);
    //! Add a named mapping from \p values.
    void addMember(const std::string& name, const TStrDoubleUMap& values);
    //! Write the result which has been constructed.
    void writeResult(void);

private:
    struct SState;
    using TStatePtr = boost::shared_ptr<SState>;

private:
    //! No-op.
    virtual bool fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames);
    //! No-op.
    virtual bool writeRow(const TStrStrUMap& dataRowFields,
                          const TStrStrUMap& overrideDataRowFields);

private:
    //! The class internals.
    TStatePtr m_State;
};
}
}

#endif // INCLUDED_ml_api_CClustererOutputWriter_h
