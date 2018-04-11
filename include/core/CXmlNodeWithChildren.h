/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_ml_core_CXmlNodeWithChildren_h
#define INCLUDED_ml_core_CXmlNodeWithChildren_h

#include <core/CXmlNode.h>
#include <core/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <vector>

namespace ml {
namespace core {
class CXmlNodeWithChildrenPool;

//! \brief
//! Representation of a XML node that can model a hierarchy.
//!
//! DESCRIPTION:\n
//! Representation of a XML node that can model a hierarchy.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Currently no support for sorting children into any order.
//!
class CORE_EXPORT CXmlNodeWithChildren : public CXmlNode {
public:
    using TXmlNodeWithChildrenP = boost::shared_ptr<CXmlNodeWithChildren>;

    using TChildNodePVec = std::vector<TXmlNodeWithChildrenP>;
    using TChildNodePVecItr = TChildNodePVec::iterator;
    using TChildNodePVecCItr = TChildNodePVec::const_iterator;

public:
    CXmlNodeWithChildren();

    CXmlNodeWithChildren(const std::string& name);

    CXmlNodeWithChildren(const std::string& name, const std::string& value);

    CXmlNodeWithChildren(const std::string& name,
                         const std::string& value,
                         const CXmlNode::TStrStrMap& attributes);

    CXmlNodeWithChildren(const CXmlNodeWithChildren& arg);

    virtual ~CXmlNodeWithChildren();

    CXmlNodeWithChildren& operator=(const CXmlNodeWithChildren& rhs);

    //! Add a child with no children of its own
    void addChild(const CXmlNode& child);

    //! Add a child
    void addChild(const CXmlNodeWithChildren& child);

    //! Add a child wrapped in a shared pointer
    void addChildP(const TXmlNodeWithChildrenP& childP);

    //! Get children
    const TChildNodePVec& children() const;

    //! Debug dump of hierarchy
    virtual std::string dump() const;
    virtual std::string dump(size_t indent) const;

private:
    //! Vector of children of this node - stored by pointer
    //! rather than by value to avoid slicing if derived classes
    //! are ever added
    TChildNodePVec m_Children;

    friend class CXmlNodeWithChildrenPool;
};
}
}

#endif // INCLUDED_ml_core_CXmlNodeWithChildren_h
