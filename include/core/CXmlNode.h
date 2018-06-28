/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CXmlNode_h
#define INCLUDED_ml_core_CXmlNode_h

#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CXmlNodeWithChildrenPool;
class CXmlParser;

//! \brief
//! Simple representation of a XML node.
//!
//! DESCRIPTION:\n
//! Simple representation of a XML node.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only values of a XML node required for XPath returns
//! implemented.
//!
//! The XML parser is a friend so that it can efficiently
//! populate attributes with minimal copying of data
//!
class CORE_EXPORT CXmlNode {
public:
    using TStrStrMap = std::map<std::string, std::string>;
    using TStrStrPr = std::pair<std::string, std::string>;
    using TStrStrPrVec = std::vector<TStrStrPr>;
    using TStrStrPrVecItr = TStrStrPrVec::iterator;
    using TStrStrPrVecCItr = TStrStrPrVec::const_iterator;

private:
    class CFirstElementEquals {
    public:
        CFirstElementEquals(const std::string& str) : m_Str(str) {}

        template<typename PAIR>
        bool operator()(const PAIR& pr) {
            const std::string& prFirst = pr.first;
            return prFirst == m_Str;
        }

    private:
        const std::string& m_Str;
    };

public:
    CXmlNode();

    CXmlNode(const std::string& name);

    CXmlNode(const std::string& name, const std::string& value);

    CXmlNode(const std::string& name, const std::string& value, const TStrStrMap& attributes);

    virtual ~CXmlNode();

    //! Accessors
    const std::string& name() const;
    const std::string& value() const;
    const TStrStrPrVec& attributes() const;

    //! Set name
    void name(const std::string& name);

    //! Set value
    void value(const std::string& value);

    //! Debug dump of all
    virtual std::string dump() const;

    //! Retrieve an attribute (if exists), and convert it to the supplied
    //! type
    template<typename TYPE>
    bool attribute(const std::string& name, TYPE& value) const {
        TStrStrPrVecCItr iter = std::find_if(
            m_Attributes.begin(), m_Attributes.end(), CFirstElementEquals(name));
        if (iter == m_Attributes.end()) {
            return false;
        }

        if (CStringUtils::stringToType(iter->second, value) == false) {
            LOG_ERROR(<< "Unable to convert " << iter->second);
            return false;
        }

        return true;
    }

    //! Set an attribute.  The caller specifies whether to overwrite an
    //! existing attribute of the same name or not.  The value must be
    //! convertible to a string using CStringUtils.
    template<typename TYPE>
    bool attribute(const std::string& name, const TYPE& value, bool overwrite) {
        TStrStrPrVecItr iter = std::find_if(m_Attributes.begin(), m_Attributes.end(),
                                            CFirstElementEquals(name));
        if (iter == m_Attributes.end()) {
            m_Attributes.push_back(TStrStrPr(name, CStringUtils::typeToString(value)));
            return true;
        }

        if (!overwrite) {
            return false;
        }

        CStringUtils::typeToString(value).swap(iter->second);

        return true;
    }

private:
    std::string m_Name;
    std::string m_Value;
    TStrStrPrVec m_Attributes;

    friend class CRapidXmlParser;
    friend class CXmlNodeWithChildrenPool;
    friend class CXmlParser;
};
}
}

#endif // INCLUDED_ml_core_CXmlNode_h
