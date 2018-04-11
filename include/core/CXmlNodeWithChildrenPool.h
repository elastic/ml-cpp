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
#ifndef INCLUDED_ml_core_CXmlNodeWithChildrenPool_h
#define INCLUDED_ml_core_CXmlNodeWithChildrenPool_h

#include <core/CStringUtils.h>
#include <core/CXmlNodeWithChildren.h>
#include <core/ImportExport.h>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Pool to provide XML nodes efficiently.
//!
//! DESCRIPTION:\n
//! Pool of XML nodes.  Will provide recycled nodes if available, or
//! create new nodes as required.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Recycling a node with children will also recycle all the children.
//!
//! Nodes obtained from the pool may have a pre-existing name and
//! value - the class that obtains the node MUST set the name and
//! value to correct values.  The reason for not clearing the old
//! name and value is that it can cause unnecessary memory allocations
//! with GNU's reference counted copy-on-write string implementation.
//!
//! This class is not thread safe.  Each object must only be used in
//! one thread at a time.  Adding locking around this class would
//! negate its benefit, because much of the cost of dynamic memory
//! allocation is in locking.  Where multiple threads are dealing
//! with XML, it is better to have one node pool per thread.
//!
class CORE_EXPORT CXmlNodeWithChildrenPool {
public:
    //! Construct a pool that will accept as many nodes as a vector will hold
    CXmlNodeWithChildrenPool();

    //! Construct a pool that will never contain more than the specified
    //! number of recycled nodes - any nodes that are recycled once the
    //! limit is reached will be deleted rather than cached
    CXmlNodeWithChildrenPool(size_t maxRecycled);

    //! Allocate a new XML node - callers MUST set the name and value of the
    //! returned node, as recycled nodes will still have their old name and
    //! value
    CXmlNodeWithChildren::TXmlNodeWithChildrenP newNode();

    //! Allocate a new XML node with the provided name and value
    CXmlNodeWithChildren::TXmlNodeWithChildrenP
    newNode(std::string name, std::string value);

    //! Allocate a new XML node with the provided name and value
    template<typename TYPE>
    CXmlNodeWithChildren::TXmlNodeWithChildrenP
    newNode(std::string name, const TYPE& value) {
        return this->newNode(name, CStringUtils::typeToString(value));
    }

    //! Allocate a new XML node with the provided name and value, specifying
    //! whether the double should be output with full precision (e.g. for
    //! persistence) or not (e.g. for human readability)
    CXmlNodeWithChildren::TXmlNodeWithChildrenP
    newNode(const std::string& name, double value, CIEEE754::EPrecision precision);

    //! Recycle an XML node, plus any children it may have
    void recycle(CXmlNodeWithChildren::TXmlNodeWithChildrenP& nodePtr);

private:
    //! Vector of recycled nodes that can be quickly provided
    //! without performing any memory allocations.
    CXmlNodeWithChildren::TChildNodePVec m_Recycled;

    //! The maximum number of nodes that will ever be cached by this pool
    size_t m_MaxRecycled;
};
}
}

#endif // INCLUDED_ml_core_CXmlNodeWithChildrenPool_h
