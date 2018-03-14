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

#ifndef INCLUDED_ml_model_CHierarchicalResultsPopulator_h
#define INCLUDED_ml_model_CHierarchicalResultsPopulator_h

#include <model/CHierarchicalResults.h>
#include <model/ImportExport.h>

namespace ml {
namespace model {
class CLimits;

//! \brief FIXME
//!
//! DESCRIPTION:\n
//! FIXME
class MODEL_EXPORT CHierarchicalResultsPopulator : public CHierarchicalResultsVisitor {
    public:
        //! Constructor
        CHierarchicalResultsPopulator(const CLimits &limits);

        //! Visit \p node.
        virtual void visit(const CHierarchicalResults &results, const TNode &node, bool pivot);

    private:
        const CLimits &m_Limits;

};

}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsPopulator_h
