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
#ifndef INCLUDED_ml_core_CStringSimilarityTester_h
#define INCLUDED_ml_core_CStringSimilarityTester_h

#include <core/CCompressUtils.h>
#include <core/CLogger.h>
#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include <stdlib.h>

class CStringSimilarityTesterTest;


namespace ml {
namespace core {


//! \brief
//! Class to measure how similar strings are.
//!
//! DESCRIPTION:\n
//! Measures how similar strings are.  There are currently two
//! available methods.
//!
//! 1) The compression method
//!
//! The two strings are compressed individually, and then
//! concatenated together (both ways around).  The idea is
//! that the concatenated string will compress much better if
//! the two input strings are similar.  And where there is no
//! commonality between the input strings, the length of the
//! compressed concatenated string should be roughly equal to
//! the sum of the lengths of the compressed input strings.
//!
//! 2) Levenshtein distance
//!
//! This is the number of substitutions, insertions or
//! deletions required to convert one string to another.
//! For example, "cat" -> "mouse" is 5, but "dog" -> "mouse"
//! is only 4, because the "o" is already in the right
//! place.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Produces symmetric results, i.e. if the similarity of
//! string 1 to string 2 is 0.8, then the similarity of string 2
//! to string 1 will also be 0.8.  This symmetry unfortunately
//! comes at the cost of an extra compression.
//!
//! The compression data structure is reused for efficiency.
//! When using the compression method, a given object must not
//! be used from multiple threads.  For multi-threaded use, with
//! the compression method, create a separate object per thread.
//! The Levenshtein distance method CAN be used from multiple
//! threads.
//!
class CORE_EXPORT CStringSimilarityTester : private CNonCopyable {
    public:
        //! Used by the simple Levenshtein distance algorithm
        typedef boost::scoped_array<size_t> TScopedSizeArray;

        //! Used by the more advanced Berghel-Roach algorithm
        typedef boost::scoped_array<int>    TScopedIntArray;
        typedef boost::scoped_array<int *>  TScopedIntPArray;

    public:
        CStringSimilarityTester(void);

        //! Calculate how similar two strings are
        bool similarity(const std::string &first,
                        const std::string &second,
                        double &result) const;

        //! Calculate how similar two strings are in the case where
        //! we already know their individual compressed lengths
        bool similarity(const std::string &first,
                        size_t firstCompLength,
                        const std::string &second,
                        size_t secondCompLength,
                        double &result) const;

        //! Remove those characters from a string that cause a provided
        //! predicate to return true (can be used with ctype.h functions
        //! like isalpha() etc, or with a functor)
        template <typename PREDICATE>
        std::string strippedString(const std::string &original,
                                   PREDICATE excludePred) const {
            std::string stripped;
            stripped.reserve(original.size());

            std::remove_copy_if(original.begin(),
                                original.end(),
                                std::back_inserter(stripped),
                                excludePred);

            return stripped;
        }

        //! Calculate how similar two strings are, excluding
        //! certain characters
        template <typename PREDICATE>
        bool similarityEx(const std::string &first,
                          const std::string &second,
                          PREDICATE excludePred,
                          double &result) const {
            return this->similarity(this->strippedString(first, excludePred),
                                    this->strippedString(second, excludePred),
                                    result);
        }

        //! Find the length of the compressed version of a string - note
        //! that the actual compressed version is discarded
        bool compressedLengthOf(const std::string &str,
                                size_t &length) const;

        //! Calculate the Levenshtein distance between two strings,
        //! excluding certain characters
        template <typename STRINGLIKE, typename PREDICATE>
        size_t levenshteinDistanceEx(const STRINGLIKE &first,
                                     const STRINGLIKE &second,
                                     PREDICATE excludePred) const {
            return this->levenshteinDistance(this->strippedString(first, excludePred),
                                             this->strippedString(second, excludePred));
        }

        //! Calculate the Levenshtein distance between two strings or
        //! string-like containers (including vector and deque).
        //! Can be applied to any container that implements size() and
        //! operator[]() where the elements held in the container implement
        //! operator==().
        template <typename STRINGLIKE>
        size_t levenshteinDistance(const STRINGLIKE &first,
                                   const STRINGLIKE &second) const {
            // Levenshtein distance is the number of operations required to
            // convert one string into another, where an operation means
            // inserting 1 character, deleting 1 character or changing 1
            // character.
            //
            // There are some examples with pretty pictures of the matrix on
            // Wikipedia.
            //
            // This has been generalised to the case of vectors, where this
            // method calculates the number of operations to convert one vector
            // to another.

            size_t firstLen(first.size());
            size_t secondLen(second.size());

            // The Berghel-Roach algorithm works in time O(n + d ^ 2), and the
            // simple algorithm works in time O(m * n), where the shorter
            // sequence length is m, the longer sequence length is m and the
            // edit distance is d.  Therefore the Berghel-Roach algorithm is
            // much faster when the two sequences are similar, but in the case
            // of them being very different, it's slower.  Additionally, each
            // operation it performs is much slower than the simple algorithm,
            // because it has to do more complex calculations to get the matrix
            // cell values.  Therefore, we need a heuristic of when to use the
            // Berghel-Roach algorithm and when to use the simple algorithm.
            // The chosen heuristic is that if the longer sequence is double the
            // length of the shorter sequence, we'll use the simple algorithm.

            if (firstLen >= secondLen) {
                // Rule out boundary case
                if (secondLen == 0) {
                    return firstLen;
                }

                if (firstLen >= secondLen * 2) {
                    return this->levenshteinDistanceSimple(second, first);
                }

                return this->berghelRoachEditDistance(second, first);
            }

            if (secondLen >= firstLen * 2) {
                // Rule out boundary case
                if (firstLen == 0) {
                    return secondLen;
                }

                return this->levenshteinDistanceSimple(first, second);
            }

            return this->berghelRoachEditDistance(first, second);
        }

        //! Calculate the weighted edit distance between two sequences.  Each
        //! element of each sequence has an associated weight, such that some
        //! elements can be considered more expensive to add/remove/replace than
        //! others.  Can be applied to any container that implements size() and
        //! operator[]() where the elements are std::pairs<T, size_t>.  The
        //! first element of each pair must implement operator==().
        //!
        //! Unfortunately, in the case of arbitrary weightings, the
        //! Berghel-Roach algorithm cannot be applied.  Ukkonen gives a
        //! counter-example on page 114 of Information and Control, Vol 64,
        //! Nos. 1-3, January/February/March 1985.  The problem is that the
        //! matrix diagonals are not necessarily monotonically increasing.
        //! See http://www.cs.helsinki.fi/u/ukkonen/InfCont85.PDF
        //!
        //! TODO - It may be possible to apply some of the lesser optimisations
        //! from section 2 of Ukkonen's paper to this algorithm.
        template <typename PAIRCONTAINER>
        size_t weightedEditDistance(const PAIRCONTAINER &first,
                                    const PAIRCONTAINER &second) const {
            // This is similar to the levenshteinDistanceSimple() method below,
            // but adding the concept of different costs for each element.  If
            // you are trying to understand this method, you should first make
            // sure you fully understand the levenshteinDistance() method above
            // (and the Wikipedia article referenced in it will help with that).

            size_t firstLen(first.size());
            size_t secondLen(second.size());

            // Rule out boundary cases
            if (firstLen == 0) {
                size_t cost(0);
                for (size_t index = 0; index < secondLen; ++index) {
                    cost += second[index].second;
                }
                return cost;
            }

            if (secondLen == 0) {
                size_t cost(0);
                for (size_t index = 0; index < firstLen; ++index) {
                    cost += first[index].second;
                }
                return cost;
            }

            // We need to store two columns of the matrix, but allocate both in
            // one go for efficiency.  Then the current and previous column
            // pointers alternate between pointing and the first and second half
            // of the memory block.
            typedef boost::scoped_array<size_t> TScopedSizeArray;
            TScopedSizeArray data(new size_t[(secondLen + 1) * 2]);
            size_t           *         currentCol(data.get());
            size_t           *         prevCol(currentCol + (secondLen + 1));

            // Populate the left column
            currentCol[0] = 0;
            for (size_t downMinusOne = 0; downMinusOne < secondLen; ++downMinusOne) {
                currentCol[downMinusOne + 1] = currentCol[downMinusOne] + second[downMinusOne].second;
            }

            // Calculate the other entries in the matrix
            for (size_t acrossMinusOne = 0; acrossMinusOne < firstLen; ++acrossMinusOne) {
                std::swap(currentCol, prevCol);
                size_t firstCost(first[acrossMinusOne].second);
                currentCol[0] = prevCol[0] + firstCost;

                for (size_t downMinusOne = 0; downMinusOne < secondLen; ++downMinusOne) {
                    size_t secondCost(second[downMinusOne].second);

                    // There are 3 options, and due to the possible differences
                    // in the weightings, we must always evaluate all 3:

                    // 1) Deletion => cell to the left's value plus cost of
                    //    deleting the element from the first sequence
                    size_t option1(prevCol[downMinusOne + 1] + firstCost);

                    // 2) Insertion => cell above's value plus cost of
                    //    inserting the element from the second sequence
                    size_t option2(currentCol[downMinusOne] + secondCost);

                    // 3) Substitution => cell above left's value plus the
                    //    higher of the two element weights
                    // OR
                    //    No extra cost in the case where the corresponding
                    //    elements are equal
                    size_t option3(prevCol[downMinusOne] +
                                   ((first[acrossMinusOne].first == second[downMinusOne].first) ?
                                    0 :
                                    std::max(firstCost, secondCost)));

                    // Take the cheapest option of the 3
                    currentCol[downMinusOne + 1] = std::min(std::min(option1,
                                                                     option2),
                                                            option3);
                }
            }

            // Result is the value in the bottom right hand corner of the matrix
            return currentCol[secondLen];
        }

    private:
        //! Calculate the Levenshtein distance using the naive method of
        //! calculating the entire distance matrix.  This private method
        //! assumes that first.size() > 0 and second.size() > 0.  However,
        //! it's best if second.size() >= first.size() in addition.
        template <typename STRINGLIKE>
        size_t levenshteinDistanceSimple(const STRINGLIKE &first,
                                         const STRINGLIKE &second) const {
            // This method implements the simple algorithm for calculating
            // Levenshtein distance.
            //
            // There are some examples with pretty pictures of the matrix on
            // Wikipedia here http://en.wikipedia.org/wiki/Levenshtein_distance

            // It's best if secondLen >= firstLen.  Although this uses more
            // space for the array below, the total number of calculations will
            // be the same, but the bigger array will make compiler
            // optimisations such as loop unrolling and vectorisation more
            // beneficial.  Most internet pages will recommend the opposite,
            // i.e. allocate the two arrays based on the size of the smaller
            // sequence, but we're more interested in speed than space.
            size_t firstLen(first.size());
            size_t secondLen(second.size());

            // We need to store two columns of the matrix, but allocate both in
            // one go for efficiency.  Then the current and previous column
            // pointers alternate between pointing and the first and second half
            // of the memory block.
            TScopedSizeArray data(new size_t[(secondLen + 1) * 2]);
            size_t           *         currentCol(data.get());
            size_t           *         prevCol(currentCol + (secondLen + 1));

            // Populate the left column
            for (size_t down = 0; down <= secondLen; ++down) {
                currentCol[down] = down;
            }

            // Calculate the other entries in the matrix
            for (size_t acrossMinusOne = 0; acrossMinusOne < firstLen; ++acrossMinusOne) {
                std::swap(currentCol, prevCol);
                currentCol[0] = acrossMinusOne + 1;

                for (size_t downMinusOne = 0; downMinusOne < secondLen; ++downMinusOne) {
                    // Do the strings differ at the point we've reached?
                    if (first[acrossMinusOne] == second[downMinusOne]) {
                        // No, they're the same => no extra cost

                        currentCol[downMinusOne + 1] = prevCol[downMinusOne];
                    } else {
                        // Yes, they differ, so there are 3 options:

                        // 1) Deletion => cell to the left's value plus 1
                        size_t option1(prevCol[downMinusOne + 1]);

                        // 2) Insertion => cell above's value plus 1
                        size_t option2(currentCol[downMinusOne]);

                        // 3) Substitution => cell above left's value plus 1
                        size_t option3(prevCol[downMinusOne]);

                        // Take the cheapest option of the 3
                        currentCol[downMinusOne + 1] = std::min(std::min(option1,
                                                                         option2),
                                                                option3) + 1;
                    }
                }
            }

            // Result is the value in the bottom right hand corner of the matrix
            return currentCol[secondLen];
        }

        //! Calculate the Levenshtein distance using the Berghel-Roach
        //! algorithm, described at http://berghel.net/publications/asm/asm.pdf
        //! This private method assumes that first.size() > 0 and
        //! second.size() >= first.size().
        template <typename STRINGLIKE>
        size_t berghelRoachEditDistance(const STRINGLIKE &first,
                                        const STRINGLIKE &second) const {
            // We need to do the calculation using signed variables
            int shortLen(static_cast<int>(first.size()));
            int maxDist(static_cast<int>(second.size()));

            // Allocate the matrix memory, and setup pointers so that we can
            // access it using negative arguments.  This enables the
            // implementation in this method to vaguely resemble the original
            // paper.
            TScopedIntArray  dataArray;
            TScopedIntPArray matrixArary;
            int              **           matrix;
            matrix = this->setupBerghelRoachMatrix(maxDist,
                                                   dataArray,
                                                   matrixArary);
            if (matrix == 0) {
                return 0;
            }

            // The remaining code corresponds to the pseudo-code in the
            // sub-section titled "The Driver Algorithm" on
            // http://berghel.net/publications/asm/asm.pdf

            // k is the difference in lengths between the two sequences, i.e.
            // the minimum distance
            int k(maxDist - shortLen);
            // p will end up storing the result
            int p(k);
            do {
                int inc(p);
                for (int tempP = 0; tempP < p; ++tempP, --inc) {
                    if (::abs(k - inc) <= tempP) {
                        this->calcDist(first,
                                       second,
                                       k - inc,
                                       tempP,
                                       matrix);
                    }
                    if (::abs(k + inc) <= tempP) {
                        this->calcDist(first,
                                       second,
                                       k + inc,
                                       tempP,
                                       matrix);
                    }
                }
                this->calcDist(first, second, k, p, matrix);

                if (matrix[k][p] == shortLen) {
                    break;
                }
            } while (++p < maxDist);

            return static_cast<size_t>(p);
        }

        //! Helper function for the Berghel-Roach edit distance algorithm.  This
        //! is called f(k, p) in http://berghel.net/publications/asm/asm.pdf
        template <typename STRINGLIKE>
        void calcDist(const STRINGLIKE &first,
                      const STRINGLIKE &second,
                      int row,
                      int column,
                      int **matrix) const {
            // 1) Substitution
            int option1(matrix[row][column - 1] + 1);

            // NB: Unlike the Berghel-Roach paper, we DO NOT consider
            // transposition at this point

            // 2) Insertion
            int option2(matrix[row - 1][column - 1]);

            // 3) Deletion
            int option3(matrix[row + 1][column - 1] + 1);

            int t(std::max(std::max(option1, option2), option3));
            int limit(std::min(static_cast<int>(first.size()),
                               static_cast<int>(second.size()) - row));
            while (t < limit && first[t] == second[t + row]) {
                ++t;
            }
            matrix[row][column] = t;
        }

        //! Setup the matrices needed for the Berghel-Roach method of
        //! calculating edit distance
        static int **setupBerghelRoachMatrix(int longLen,
                                             TScopedIntArray &data,
                                             TScopedIntPArray &matrix);

    private:
        //! Required for initialisation of the Berghel-Roach matrix (don't call
        //! this MINUS_INFINITY because that can clash with 3rd party macros)
        static const int       MINUS_INFINITE_INT;

        //! Used by the compression-based similarity measures
        mutable CCompressUtils m_Compressor;

        // For unit testing
        friend class ::CStringSimilarityTesterTest;
};


}
}

#endif // INCLUDED_ml_core_CStringSimilarityTester_h

