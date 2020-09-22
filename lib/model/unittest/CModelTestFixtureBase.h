#ifndef INCLUDED_CModelTestFixtureBase_h
#define INCLUDED_CModelTestFixtureBase_h

#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CModel.h>
#include <maths/CMultivariatePrior.h>

#include <model/CResourceMonitor.h>

using TDouble1Vec = ml::core::CSmallVector<double, 1>;
using TDouble2Vec = ml::core::CSmallVector<double, 2>;
using TDouble4Vec = ml::core::CSmallVector<double, 4>;
using TDouble4Vec1Vec = ml::core::CSmallVector<TDouble4Vec, 1>;
using TDoubleDoublePr = std::pair<double, double>;
using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
using TDoubleSizePr = std::pair<double, std::size_t>;
using TDoubleStrPr = std::pair<double, std::string>;
using TDoubleStrPrVec = std::vector<TDoubleStrPr>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;

using TMathsModelPtr = std::shared_ptr<ml::maths::CModel>;
using TMeanAccumulator = ml::maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMultivariatePriorPtr = std::shared_ptr<ml::maths::CMultivariatePrior>;

using TOptionalDouble = boost::optional<double>;
using TOptionalDoubleVec = std::vector<TOptionalDouble>;
using TOptionalStr = boost::optional<std::string>;
using TOptionalUInt64 = boost::optional<uint64_t>;

using TPriorPtr = std::shared_ptr<ml::maths::CPrior>;

using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr1Vec = ml::core::CSmallVector<TSizeDoublePr, 1>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrVec = std::vector<TSizeSizePr>;
using TSizeSizePrVecVec = std::vector<TSizeSizePrVec>;
using TSizeSizePrUInt64Map = std::map<TSizeSizePr, uint64_t>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TSizeVecVecVec = std::vector<TSizeVecVec>;

using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;

using TTimeDoublePr = std::pair<ml::core_t::TTime, double>;
using TOptionalTimeDoublePr = boost::optional<TTimeDoublePr>;
using TTimeStrVecPr = std::pair<ml::core_t::TTime, TStrVec>;
using TTimeStrVecPrVec = std::vector<TTimeStrVecPr>;
using TTimeVec = std::vector<ml::core_t::TTime>;

using TUInt64Vec = std::vector<uint64_t>;
using TUIntVec = std::vector<unsigned int>;

class CModelTestFixtureBase {
protected:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif //INCLUDED_CModelTestFixtureBase_h
