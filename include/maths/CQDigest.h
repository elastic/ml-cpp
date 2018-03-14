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

#ifndef INCLUDED_ml_maths_CQDigest_h
#define INCLUDED_ml_maths_CQDigest_h

#include <core/CNonCopyable.h>

#include <maths/ImportExport.h>

#include <functional>
#include <list>
#include <string>
#include <vector>

#include <stdint.h>


namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief This class implements the q-digest quantile approximation.
//!
//! DESCRIPTION:\n
//! The q-digest algorithm computes approximate quantiles for a data
//! stream in fixed space. In particular, if the universe of values
//! is \f$[N] = {1,...,N}\f$ then it can provide worst case quantile
//! errors of \f$\displaystyle O(\frac{log(N)}{k})\f$ using space of
//! \f$O(k)\f$ integers. Note that error is actually governed by the
//! observed data range which may be much smaller than the size of
//! the universe. Note also that the definition of the error is in
//! terms of the error in the number of values less than or greater
//! than value returned *not* the value itself, i.e.
//! <pre class="fragment">
//!   \f$\displaystyle \epsilon = \frac{|r - qn|}{n}\f$
//! </pre>
//!
//! This is worth being aware of since if there are large gaps in the
//! values the error in the value of the quantile can be large even if
//! the number of values less than that value is very close to \f$qn\f$.
//!
//! For fixed range double precision values we can convert our universe
//! to integers by transforming the data as follows:
//! <pre class="fragment">
//!   \f$\displaystyle x \rightarrow \left \lfloor xn \right \rfloor\f$
//! </pre>
//!
//! for some large value of \f$n\f$ and then mapping the quantiles back
//! using:
//! <pre class="fragment">
//!   \f$\displaystyle q \rightarrow \frac{x}{n} + 0.5\f$
//! </pre>
//!
//! See "Medians and Beyond: New Aggregation Techniques for Sensor Networks"
//! Nisheeth Shrivastava, Chiranjeeb Buragohain, Divyakant Agrawal, Subhash Suri
//!
//! IMPLEMENTATION DECISIONS:\n
//! This uses a less space compact representation to simplify the
//! implementation. In particular, the minimum node size is two integers
//! (strictly we only need to store the post-order in the full tree).
//! The overhead of k integers is fine for our purposes.
//!
//! As extensions to the paper we reduce the scope of compression where
//! possible, ensure that the tree invariants are met by running multiple
//! compression passes when updating incrementally and store the nodes
//! as a rooted tree for efficiency and maintains the count in the subtree
//! rooted at each node to accelerate quantile queries.
//!
//! This uses the fact the maximum length of the q-digest is \f$3k\f$
//! to ensure constant complexity of all operations at various points
//! and to reserve sufficient memory up front for our node allocator.
class MATHS_EXPORT CQDigest : private core::CNonCopyable {
    public:
        typedef std::pair<uint32_t, uint64_t> TUInt32UInt64Pr;
        typedef std::vector<TUInt32UInt64Pr>  TUInt32UInt64PrVec;

    public:
        //! \name XML Tag Names
        //!
        //! These tag the member variables for persistence.
        //@{
        static const std::string K_TAG;
        static const std::string N_TAG;
        static const std::string NODE_TAG;
    //@}

    public:
        CQDigest(uint64_t k, double decayRate = 0.0);

        //! \name Serialization
        //@{
        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Create from an XML node tree.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);
        //@}

        //! Add \p n values \p value to the q-digest.
        void add(uint32_t value, uint64_t n = 1ull);

        //! Merge this and \p digest.
        void merge(const CQDigest &digest);

        //! Lose information from the digest. This amounts to aging
        //! the counts held by each node and reducing the total count
        //! in the digest.
        void propagateForwardsByTime(double time);

        //! Scale the quantiles by the specified factor.  To be used
        //! after upgrades if different versions of the product produce
        //! different raw anomaly scores.
        bool scale(double factor);

        //! Reset the quantiles to the state before any values were added.
        void clear(void);

        //! Compute the quantile \p q.
        //!
        //! \param[in] q The quantile should be in the range [0, 1]
        //! and represents the fraction of values less than the
        //! quantile value required.
        //! \param[out] result Filled in with the quantile if the
        //! digest isn't empty.
        //! \return True if the quantile could be computed and
        //! false otherwise.
        bool quantile(double q, uint32_t &result) const;

        //! Find the largest value x such that upper bound of the
        //! c.d.f. is less than \p f, i.e. \f$\sup_y{\{y:F(y)<f\}}\f$.
        //!
        //! \note For this calculation we assume that the c.d.f. is
        //! only defined on the points where it changes, i.e. the
        //! q-digest node end points. So this returns the rightmost
        //! end point where the upper c.d.f. is less than \p f.
        bool quantileSublevelSetSupremum(double f, uint32_t &result) const;

        //! Get the quantile corresponding to the c.d.f. value \p.
        //!
        //! \param[in] n The number of samples.
        //! \param[in] p The c.d.f. value.
        //! \param[in] q The quantile in the range [0,1].
        static double cdfQuantile(double n, double p, double q);

        //! Compute the fraction of values less than \p x.
        //!
        //! \param[in] x The value for which to compute the c.d.f.
        //! \param[in] confidence The symmetric confidence interval
        //! for the c.d.f as a percentage.
        //! \param[out] lowerBound Filled in with the lower bound for
        //! the c.d.f. at \p x.
        //! \param[out] upperBound Filled in with the upper bound for
        //! the c.d.f. at \p x.
        bool cdf(uint32_t x, double confidence,
                 double &lowerBound, double &upperBound) const;

        //! Compute the value of the p.d.f. at \p x.
        //!
        //! \param[in] x The value for which to compute the p.d.f.
        //! \param[in] confidence The symmetric confidence interval
        //! for the p.d.f as a percentage.
        //! \param[out] lowerBound Filled in with the lower bound for
        //! the p.d.f. at \p x.
        //! \param[out] upperBound Filled in with the upper bound for
        //! the p.d.f. at \p x.
        void pdf(uint32_t x, double confidence,
                 double &lowerBound, double &upperBound) const;

        //! Get the maximum knot point less than \p x.
        void sublevelSetSupremum(uint32_t x, uint32_t &result) const;

        //! Get the minimum knot point greater than \p x.
        void superlevelSetInfimum(uint32_t x, uint32_t &result) const;

        //! Get a summary of the q-digest. This is the counts less
        //! than or equal to each distinct integer in the quantile
        //! summary.
        //!
        //! \param[out] result Filled in with the summary.
        void summary(TUInt32UInt64PrVec &result) const;

        //! Get the total number of values added to the q-digest.
        uint64_t n(void) const;

        //! Get the size factor "k" for the q-digest.
        uint64_t k(void) const;

        //! Get a checksum of this object.
        uint64_t checksum(uint64_t seed) const;

        //! \name Test Methods
        //@{
        //! Check the digest invariants.
        bool checkInvariants(void) const;

        //! Print the q-digest.
        std::string print(void) const;
    //@}

    private:
        typedef std::vector<std::size_t>  TSizeVec;
        typedef TSizeVec::const_iterator  TSizeVecCItr;
        typedef std::greater<std::size_t> TSizeGreater;

        class CNode;
        class CNodeAllocator;

        typedef std::vector<CNode*>                 TNodePtrVec;
        typedef TNodePtrVec::iterator               TNodePtrVecItr;
        typedef TNodePtrVec::const_iterator         TNodePtrVecCItr;
        typedef TNodePtrVec::const_reverse_iterator TNodePtrVecCRItr;

        //! Orders node pointers by level order.
        struct MATHS_EXPORT SLevelLess {
            bool operator()(const CNode *lhs, const CNode *rhs) const;
        };

        //! Order node pointers by post order in completed tree.
        struct MATHS_EXPORT SPostLess {
            bool operator()(const CNode *lhs, const CNode *rhs) const;
        };

        //! Represents a node of the q-digest with convenience
        //! operations for compression.
        class MATHS_EXPORT  CNode {
            public:
                //! \name XML Tag Names
                //!
                //! These tag the member variables for persistence.
                //@{
                static const std::string MIN_TAG;
                static const std::string MAX_TAG;
                static const std::string COUNT_TAG;
            //@}

            public:
                CNode(void);
                CNode(uint32_t min,
                      uint32_t max,
                      uint64_t count,
                      uint64_t subtreeCount);

                //! Get the size of the q-digest rooted at this node.
                std::size_t size(void) const;

                //! Get the approximate quantile \p n.
                uint32_t quantile(uint64_t leftCount,
                                  uint64_t n) const;

                //! Get the largest value of x for which the upper count
                //! i.e. count of values definitely to the right of x, is
                //! less than \p n.
                bool quantileSublevelSetSupremum(uint64_t n,
                                                 uint64_t leftCount,
                                                 uint32_t &result) const;

                //! Get the lower bound for the c.d.f. at \p x.
                void cdfLowerBound(uint32_t x, uint64_t &result) const;

                //! Get the upper bound for the c.d.f. at \p x.
                void cdfUpperBound(uint32_t x, uint64_t &result) const;

                //! Get the maximum knot point less than \p x.
                void sublevelSetSupremum(const int64_t x, uint32_t &result) const;

                //! Get the minimum knot point greater than \p x.
                void superlevelSetInfimum(uint32_t x, uint32_t &result) const;

                //! Fill in \p nodes with q-digest nodes in post-order.
                void postOrder(TNodePtrVec &nodes) const;

                //! Expand the node to fit \p value.
                CNode *expand(CNodeAllocator &allocator, const uint32_t &value);

                //! Insert the specified node at its lowest ancestor
                //! in the q-digest.
                CNode &insert(CNodeAllocator &allocator, const CNode &node);

                //! Compress the digest at the triple comprising this node,
                //! its sibling and parent in the complete tree if they are
                //! in the q-digest.
                CNode *compress(CNodeAllocator &allocator,
                                uint64_t compressionFactor);

                //! Age the counts by the specified factor.
                uint64_t age(double factor);

                //! Get the span of universe values covered by the node.
                uint32_t span(void) const;
                //! Get the minimum value covered by the node.
                uint32_t min(void) const;
                //! Get the maximum value covered by the node.
                uint32_t max(void) const;
                //! Get the count of entries in the node range.
                const uint64_t &count(void) const;
                //! Get the count in the subtree rooted at this node.
                const uint64_t &subtreeCount(void) const;

                //! Persist this node and descendents
                void persistRecursive(const std::string &nodeTag,
                                      core::CStatePersistInserter &inserter) const;

                //! Create from an XML node tree.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Check the node invariants in the q-digest rooted at this node.
                bool checkInvariants(uint64_t compressionFactor) const;

                //! Print for debug.
                std::string print(void) const;

            private:
                //! Persist state by passing information to the supplied
                //! inserter - this should only be called by persistRecursive()
                //! to ensure the whole tree gets persisted
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Test for equality.
                bool operator==(const CNode &node) const;

                //! Get the index of the immediate ancestor in the q-digest.
                CNode *ancestor(void) const;
                //! Get the number of descendants.
                std::size_t numberDescendants(void) const;
                //! Get an iterator over the descendants.
                TNodePtrVecCItr beginDescendants(void) const;
                //! Get the end of the descendants.
                TNodePtrVecCItr endDescendants(void) const;
                //! Get the sibling of \p node if it exists in the q-digest.
                CNode *sibling(const CNode &node) const;

                //! Is this a sibling of \p node?
                bool isSibling(const CNode &node) const;
                //! Is this a parent of \p node?
                bool isParent(const CNode &node) const;
                //! Is this an ancestor of \p node in the complete tree.
                bool isAncestor(const CNode &node) const;
                //! Is this node the root.
                bool isRoot(void) const;
                //! Is this node a leaf.
                bool isLeaf(void) const;
                //! Is this the left child of a node in the complete tree.
                bool isLeftChild(void) const;

                //! Detach this node from the q-digest.
                void detach(CNodeAllocator &allocator);
                //! Remove \p node from the descendants.
                void removeDescendant(CNode &node);
                //! Take the descendants of \p node.
                bool takeDescendants(CNode &node);

            private:
                //! The immediate ancestor of this node in the q-digest.
                CNode       *m_Ancestor;

                //! The immediate descendants of this node in the q-digest.
                TNodePtrVec m_Descendants;

                //! The minimum value covered by the node.
                uint32_t    m_Min;

                //! The maximum value covered by the node.
                uint32_t    m_Max;

                //! The count of the node.
                uint64_t    m_Count;

                //! The count in the subtree root at this node.
                uint64_t    m_SubtreeCount;
        };

        //! Manages the creation and recycling of nodes.
        class MATHS_EXPORT CNodeAllocator {
            public:
                CNodeAllocator(std::size_t size);

                //! Create a new node.
                CNode &create(const CNode &node);

                //! Recycle \p node.
                void release(CNode &node);

            private:
                typedef std::vector<TNodePtrVec>           TNodePtrVecVec;
                typedef std::vector<CNode>                 TNodeVec;
                typedef std::vector<CNode>::const_iterator TNodeVecCItr;
                typedef std::list<TNodeVec>                TNodeVecList;
                typedef TNodeVecList::iterator             TNodeVecListItr;
                typedef TNodeVecList::const_iterator       TNodeVecListCItr;

            private:
                //! Find the block to which \p node belongs.
                std::size_t findBlock(const CNode &node) const;

            private:
                TNodeVecList   m_Nodes;
                TNodePtrVecVec m_FreeNodes;
        };

    private:
        //! Compress the q-digest bottom up in level order.
        void compress(void);

        //! Starting at the lowest nodes in \p compress in level order
        //! compress all q-digest paths bottom up in level order to the
        //! root.
        bool compress(TNodePtrVec &compress);

    private:
        //! Controls the maximum number of values stored. In particular,
        //! the number of nodes is less than \f$3k\f$.
        uint64_t       m_K;
        //! The number of values added to the q-digest.
        uint64_t       m_N;
        //! The root node.
        CNode          *m_Root;
        //! The node allocator.
        CNodeAllocator m_NodeAllocator;
        //! The rate at which information is lost by the digest.
        double         m_DecayRate;
};

}
}

#endif // INCLUDED_ml_maths_CQDigest_h
