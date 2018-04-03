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
#ifndef INCLUDED_ml_core_CStateRestoreTraverser_h
#define INCLUDED_ml_core_CStateRestoreTraverser_h

#include <core/CLogger.h>
#include <core/CNonCopyable.h>

#include <core/ImportExport.h>

#include <string>


namespace ml {
namespace core {


//! \brief
//! Abstract interface for restoring state.
//!
//! DESCRIPTION:\n
//! Classes that need to restore state may accept this interface
//! as a means to retrieve values being restored without needing
//! to know the exact format that they were stored in.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable.
//!
//! This class has been named "traverser" rather than "iterator" as
//! it works on a two dimensional structure: as well as iterating
//! forwards it's possible to descend to sub-levels of the current
//! element and iterate through the values at that level.  Since
//! each level would have its own "end" iterator, the interface is
//! that the next() method returns false when the end of a particular
//! sub-level is reached.
//!
//! All values are returned as strings.
//!
class CORE_EXPORT CStateRestoreTraverser : private CNonCopyable {
    public:
        CStateRestoreTraverser(void);

        //! Virtual destructor for abstract class
        virtual ~CStateRestoreTraverser(void);

        //! Navigate to the next element at the current level, or return false
        //! if there isn't one
        virtual bool next(void) = 0;

        //! Does the current element have a sub-level?
        virtual bool hasSubLevel(void) const = 0;

        //! Traverse the sub-level of the current element.  The supplied
        //! function will be called with the traverser pointing at the first
        //! element of the sub-level.  When the function returns, the traverser
        //! will ascend back to the element at the higher level.  The supplied
        //! should return a bool and this will be passed on as the return value
        //! of this method.
        template <typename FUNC>
        bool traverseSubLevel(FUNC f) {
            if (!this->hasSubLevel()) {
                return false;
            }

            CAutoLevel level(*this);
            try {
                return f(*this);
            } catch (const std::exception &e)   {
                LOG_ERROR("Restoration failed: " << e.what());
                level.setBadState();
                return false;
            }
        }

        //! Get the name of the current element - the returned reference is only
        //! valid for as long as the traverser is pointing at the same element
        virtual const std::string &name(void) const = 0;

        //! Get the value of the current element - the returned reference is
        //! only valid for as long as the traverser is pointing at the same
        //! element
        virtual const std::string &value(void) const = 0;

        //! Has the end of the inputstream been reached?
        virtual bool isEof(void) const = 0;

        //! Is the state document unintelligible?
        bool haveBadState(void) const;

    protected:
        //! Set the bad state flag, which indicates that the state document was
        //! unintelligible.
        void setBadState(void);

        //! Navigate to the start of the sub-level of the current element, or
        //! return false if there isn't one
        virtual bool descend(void) = 0;

        //! Navigate to the element of the level above from which descend() was
        //! called, or return false if there isn't a level above
        virtual bool ascend(void) = 0;

    private:
        //! Class to implement RAII for traversing the next level down
        class CORE_EXPORT CAutoLevel : private CNonCopyable {
            public:
                CAutoLevel(CStateRestoreTraverser &traverser);
                ~CAutoLevel(void);

                //! Set the bad state flag, called from an exception handler
                //! further up, so that we don't try and read from the stream
                //! in the destructor
                void setBadState(void);

            private:
                CStateRestoreTraverser &m_Traverser;

                //! Remember whether descent on construction succeeded
                bool m_Descended;

                //! If a stream parsing error occurs, don't try and descend
                //! in the destructor
                bool m_BadState;
        };

    private:
        //! Flag that should be set when the state document is unintelligible.
        bool m_BadState;
};


}
}

#endif // INCLUDED_ml_core_CStateRestoreTraverser_h

