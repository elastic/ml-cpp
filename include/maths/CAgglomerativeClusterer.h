/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CAgglomerativeClusterer_h
#define INCLUDED_ml_maths_CAgglomerativeClusterer_h

#include <maths/ImportExport.h>

#include <string>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Implements optimum runtime agglomerative clustering for
//! arbitrary distance matrices.
//!
//! DESCRIPTION:\n
//! Agglomerative clustering builds a binary tree bottom up on a set
//! of points. At each stage it merges the closest pair of clusters
//! subject to a specified distance function. This implements a few
//! choices, comprising single, complete, average, weighted link and
//! ward, for which an optimum \f$O(N^2)\f$ algorithm is known. The
//! single link objective defines the distance between clusters
//! \f$A = \{x_i\}\f$ and \f$B = \{x_j\}\f$ as
//! <pre class="fragment">
//!   \f$\displaystyle d_s(A, B) = min_{i,j}{d(x_i, x_j)\f$
//! </pre>
//!
//! and the complete link objective defines it as
//! <pre class="fragment">
//!   \f$\displaystyle d_c(A,B) = min_{i,j}{d(x_i, x_j)\f$
//! </pre>
//!
//! For other styles see https://en.wikipedia.org/wiki/Hierarchical_clustering#Agglomerative_clustering_example.
class MATHS_EXPORT CAgglomerativeClusterer
{
    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;
        using TSizeVec = std::vector<std::size_t>;
        using TSizeVecVec = std::vector<TSizeVec>;
        using TDoubleSizeVecPr = std::pair<double, TSizeVec>;
        using TDoubleSizeVecPrVec = std::vector<TDoubleSizeVecPr>;

        //! \brief A representation of a node in the tree of clusters.
        class MATHS_EXPORT CNode
        {
            public:
                //! Set the rightmost point below this node.
                CNode(std::size_t index, double height);

                //! Add a child node and update connectivity.
                bool addChild(CNode &child);

                //! Get the unique index of this node.
                std::size_t index() const;

                //! Get the height of this node.
                double height() const;

                //! Get the root of the branch containing this node.
                //!
                //! \note This is the root of the tree unless it is
                //! under construction.
                CNode &root();

                //! Get the points in this node's cluster.
                void points(TSizeVec &result) const;

                //! Get the joins and their heights.
                void clusters(TDoubleSizeVecPrVec &result) const;

                //! Get the clustering at the specified height.
                void clusteringAt(double height, TSizeVecVec &result) const;

                //! Get a debug representation of the branch rooted at
                //! this node.
                std::string print(const std::string &indent = std::string("  ")) const;

            private:
                //! The parent cluster.
                CNode *m_Parent;
                //! The left child cluster.
                CNode *m_LeftChild;
                //! The right child cluster.
                CNode *m_RightChild;
                //! The unique index of this cluster.
                std::size_t m_Index;
                //! The height of this cluster, i.e. the value of the
                //! objective function at which the cluster forms.
                double m_Height;
        };

        using TNodeVec = std::vector<CNode>;

    public:
        //! Possible clustering objective functions supported.
        enum EObjective
        {
            E_Single,
            E_Complete,
            E_Average,
            E_Weighted,
            E_Ward
        };

    public:
        //! Setup the distance matrix from which to compute the
        //! agglomerative clustering.
        bool initialize(TDoubleVecVec &distanceMatrix);

        //! Run agglomerative clustering targeting \p objective
        //! and build the cluster tree.
        void run(EObjective objective, TNodeVec &tree);

    private:
        //! The distance matrix on the points to cluster.
        TDoubleVecVec m_DistanceMatrix;
        //! Filled in with the last object in each cluster to which
        //! i'th point connects.
        TSizeVec m_Pi;
        //! Filled in with the lowest level at which the i'th point
        //! is no longer the last object in its cluster.
        TDoubleVec m_Lambda;
        //! Holds a copy of a column of the distance matrix during
        //! update point representation.
        TDoubleVec m_M;
};

}
}

#endif // INCLUDED_ml_maths_CAgglomerativeClusterer_h
