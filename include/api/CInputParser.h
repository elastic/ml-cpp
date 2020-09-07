/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInputParser_h
#define INCLUDED_ml_api_CInputParser_h

#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <functional>
#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief
//! Input parser interface
//!
//! DESCRIPTION:\n
//! Abstract base class for input parser classes.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Abstract interface declares the readStreamIntoMaps and readStreamIntoVecs
//! methods that must be implemented in sub-classes.
//!
class API_EXPORT CInputParser {
public:
    using TStrVec = std::vector<std::string>;

    using TStrStrUMap = boost::unordered_map<std::string, std::string>;

    //! For fast access to the field values without repeatedly computing the
    //! hash, we maintain references to the values in the hash map
    using TStrRef = std::reference_wrapper<std::string>;
    using TStrRefVec = std::vector<TStrRef>;

    //! Callback function prototype for informing the consumer which fields
    //! in the input data they are permitted to mutate.
    using TRegisterMutableFieldFunc = std::function<void(const std::string&, std::string&)>;

    //! Callback function prototype that gets called for each record read
    //! from the input stream when reading into a map.  Return false to exit
    //! reader loop.  The argument is a map of field name to field value.
    using TMapReaderFunc = std::function<bool(const TStrStrUMap&)>;

    //! Callback function prototype that gets called for each record read
    //! from the input stream when reading into vectors.  Return false to exit
    //! reader loop.  The arguments are vectors of field names and field values.
    using TVecReaderFunc = std::function<bool(const TStrVec&, const TStrVec&)>;

public:
    CInputParser(TStrVec mutableFieldNames);
    virtual ~CInputParser() = default;

    //! No copying
    CInputParser(const CInputParser&) = delete;
    CInputParser& operator=(const CInputParser&) = delete;

    //! Get field names
    const TStrVec& fieldNames() const;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    bool readStreamIntoMaps(const TMapReaderFunc& readerFunc) {
        return this->readStreamIntoMaps(readerFunc, TRegisterMutableFieldFunc{});
    }

    //! As above, but also supplying function for registering mutable fields.
    virtual bool readStreamIntoMaps(const TMapReaderFunc& readerFunc,
                                    const TRegisterMutableFieldFunc& registerFunc) = 0;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    bool readStreamIntoVecs(const TVecReaderFunc& readerFunc) {
        return this->readStreamIntoVecs(readerFunc, TRegisterMutableFieldFunc{});
    }

    //! As above, but also supplying function for registering mutable fields.
    virtual bool readStreamIntoVecs(const TVecReaderFunc& readerFunc,
                                    const TRegisterMutableFieldFunc& registerFunc) = 0;

protected:
    //! Add any mutable fields to the map that will be passed to the reader
    //! function, calling the registration function for each one.
    void registerMutableFields(const TRegisterMutableFieldFunc& registerFunc,
                               TStrStrUMap& dataRowFields) const;

    //! Add any mutable fields to the vectors that will be passed to the reader
    //! function, calling the registration function for each one.
    void registerMutableFields(const TRegisterMutableFieldFunc& registerFunc,
                               TStrVec& fieldNames,
                               TStrVec& fieldValues) const;

    //! Writable access to the field names for derived classes only
    TStrVec& fieldNames();

private:
    //! Field names parsed from the input
    TStrVec m_FieldNames;

    //! Names of mutable fields, which may or may not be in the input
    TStrVec m_MutableFieldNames;
};
}
}

#endif // INCLUDED_ml_api_CInputParser_h
