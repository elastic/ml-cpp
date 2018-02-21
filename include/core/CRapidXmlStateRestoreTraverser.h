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
#ifndef INCLUDED_ml_core_CRapidXmlStateRestoreTraverser_h
#define INCLUDED_ml_core_CRapidXmlStateRestoreTraverser_h

#include <core/CRapidXmlParser.h>
#include <core/CStateRestoreTraverser.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{


//! \brief
//! For restoring state in XML format.
//!
//! DESCRIPTION:\n
//! Concrete implementation of the CStateRestoreTraverser interface
//! that restores state in XML format.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Directly uses RapidXml to avoid the inefficiency of an
//! intermediate node hierarchy.
//!
//! Stores a const reference to an XML parser, which must not be
//! modified externally during the lifetime of any object of this
//! class that references it.
//!
//! Does NOT support CDATA in state XML - any CDATA content is
//! currently ignored.
//!
class CORE_EXPORT CRapidXmlStateRestoreTraverser : public CStateRestoreTraverser
{
    public:
        CRapidXmlStateRestoreTraverser(const CRapidXmlParser &parser);

        //! Navigate to the next element at the current level, or return false
        //! if there isn't one
        virtual bool next(void);

        //! Does the current element have a sub-level?
        virtual bool hasSubLevel(void) const;

        //! Get the name of the current element - the returned reference is only
        //! valid for as long as the traverser is pointing at the same element
        virtual const std::string &name(void) const;

        //! Get the value of the current element - the returned reference is
        //! only valid for as long as the traverser is pointing at the same
        //! element
        virtual const std::string &value(void) const;

        //! Has the end of the underlying document been reached?
        virtual bool isEof(void) const;

    protected:
        //! Navigate to the start of the sub-level of the current element, or
        //! return false if there isn't one
        virtual bool descend(void);

        //! Navigate to the element of the level above from which descend() was
        //! called, or return false if there isn't a level above
        virtual bool ascend(void);

    private:
        //! Get a pointer to the next node element sibling of the current node,
        //! or return NULL if there isn't one
        CRapidXmlParser::TCharRapidXmlNode *nextNodeElement(void) const;

        //! Get a pointer to the first child node element of the current node,
        //! or return NULL if there isn't one
        CRapidXmlParser::TCharRapidXmlNode *firstChildNodeElement(void) const;

    private:
        //! The parser that has been used to parse the document to be traversed
        const CRapidXmlParser              &m_Parser;

        //! Pointer to current node within the document
        CRapidXmlParser::TCharRapidXmlNode *m_CurrentNode;

        //! RapidXml stores strings as const char *s, which we don't want to
        //! use widely throughout our codebase.  These strings store copies of
        //! the name and value of the current node so that the name() and
        //! value() methods can return them quickly.
        mutable std::string                m_CachedName;
        mutable std::string                m_CachedValue;

        //! Are m_CachedName and m_CachedValue valid?
        mutable bool                       m_IsNameCacheValid;
        mutable bool                       m_IsValueCacheValid;
};


}
}

#endif // INCLUDED_ml_core_CRapidXmlStateRestoreTraverser_h

