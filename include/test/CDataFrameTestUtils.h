/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_test_CDataFrameTestUtils_h
#define INCLUDED_ml_test_CDataFrameTestUtils_h

#include <core/CDataFrame.h>

#include <maths/CLinearAlgebraShims.h>

#include <test/ImportExport.h>

#include <vector>

namespace ml {
namespace test {

//! \brief Utility functionality for writing tests for data frame analytics.
class TEST_EXPORT CDataFrameTestUtils {
public:
    //! Create a data frame in main memory whose rows are initialized with \p points.
    template<typename POINT>
    static auto toMainMemoryDataFrame(const std::vector<POINT>& points) {
        return toDataFrame(
            [](std::size_t dimension) {
                return core::makeMainStorageDataFrame(dimension);
            },
            points);
    }

    //! Create a data frame on disk whose rows are initialized with \p points.
    template<typename POINT>
    static auto toOnDiskDataFrame(const std::string& directory,
                                  const std::vector<POINT>& points) {
        return toDataFrame(
            [&](std::size_t dimension) {
                return core::makeDiskStorageDataFrame(directory, dimension,
                                                      points.size());
            },
            points);
    }

private:
    template<typename FACTORY, typename POINT>
    static auto toDataFrame(FACTORY factory, const std::vector<POINT>& points) {

        std::size_t dimension{maths::las::dimension(points[0])};

        auto result = factory(dimension).first;

        for (std::size_t i = 0; i < points.size(); ++i) {
            result->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t& id) {
                for (std::size_t d = 0; d < dimension; ++d, ++column) {
                    *column = points[i](d);
                }
                id = static_cast<std::int32_t>(i);
            });
        }
        result->finishWritingRows();

        return result;
    }
};
}
}

#endif // INCLUDED_ml_test_CDataFrameTestUtils_h
