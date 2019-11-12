/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CBaseTokenListDataCategorizer.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <model/CTokenListReverseSearchCreatorIntf.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <set>

namespace ml {
namespace model {

// Initialise statics
const std::string CBaseTokenListDataCategorizer::PRETOKENISED_TOKEN_FIELD("...");

// We use short field names to reduce the state size
namespace {
const std::string TOKEN_TAG("a");
const std::string TOKEN_CATEGORY_COUNT_TAG("b");
const std::string CATEGORY_TAG("c");

const std::string TIME_ATTRIBUTE("time");

const std::string EMPTY_STRING;
}

CBaseTokenListDataCategorizer::CBaseTokenListDataCategorizer(
    const TTokenListReverseSearchCreatorIntfCPtr& reverseSearchCreator,
    double threshold,
    const std::string& fieldName)
    : CDataCategorizer(fieldName), m_ReverseSearchCreator(reverseSearchCreator),
      m_LowerThreshold(std::min(0.99, std::max(0.01, threshold))),
      // Upper threshold is half way between the lower threshold and 1
      m_UpperThreshold((1.0 + m_LowerThreshold) / 2.0), m_HasChanged(false) {
}

void CBaseTokenListDataCategorizer::dumpStats() const {
    // ML category number is vector index plus one
    int categoryId(1);
    for (const auto& category : m_Categories) {
        LOG_DEBUG(<< "ML category=" << categoryId << '-'
                  << category.numMatches() << ' ' << category.baseString());
        ++categoryId;
    }
}

int CBaseTokenListDataCategorizer::computeCategory(bool isDryRun,
                                                   const TStrStrUMap& fields,
                                                   const std::string& str,
                                                   size_t rawStringLen) {
    // First tokenise string
    size_t workWeight(0);
    auto preTokenisedIter = fields.find(PRETOKENISED_TOKEN_FIELD);
    if (preTokenisedIter != fields.end()) {
        if (this->addPretokenisedTokens(preTokenisedIter->second, m_WorkTokenIds,
                                        m_WorkTokenUniqueIds, workWeight) == false) {
            return -1;
        }
    } else {
        this->tokeniseString(fields, str, m_WorkTokenIds, m_WorkTokenUniqueIds, workWeight);
    }

    // Determine the minimum and maximum token weight that could possibly
    // match the weight we've got
    size_t minWeight(CBaseTokenListDataCategorizer::minMatchingWeight(workWeight, m_LowerThreshold));
    size_t maxWeight(CBaseTokenListDataCategorizer::maxMatchingWeight(workWeight, m_LowerThreshold));

    // We search previous categories in descending order of the number of matches
    // we've seen for them
    TSizeSizePrListItr bestSoFarIter(m_CategoriesByCount.end());
    double bestSoFarSimilarity(m_LowerThreshold);
    for (TSizeSizePrListItr iter = m_CategoriesByCount.begin();
         iter != m_CategoriesByCount.end(); ++iter) {
        const CTokenListCategory& compCategory = m_Categories[iter->second];
        const TSizeSizePrVec& baseTokenIds = compCategory.baseTokenIds();
        size_t baseWeight(compCategory.baseWeight());

        // Check whether the current record matches the search for the existing
        // category - if it does then we'll put it in the existing category without any
        // further checks.  The first condition here ensures that we never say
        // a string with tokens matches the reverse search of a string with no
        // tokens (which the other criteria alone might say matched).
        bool matchesSearch((baseWeight == 0) == (workWeight == 0) &&
                           compCategory.maxMatchingStringLen() >= rawStringLen &&
                           compCategory.isMissingCommonTokenWeightZero(m_WorkTokenUniqueIds) &&
                           compCategory.containsCommonTokensInOrder(m_WorkTokenIds));
        if (!matchesSearch) {
            // Quickly rule out wildly different token weights prior to doing
            // the expensive similarity calculations
            if (baseWeight < minWeight || baseWeight > maxWeight) {
                continue;
            }

            // Rule out categories where adding the current string would unacceptably
            // reduce the number of unique common tokens
            size_t origUniqueTokenWeight(compCategory.origUniqueTokenWeight());
            size_t commonUniqueTokenWeight(compCategory.commonUniqueTokenWeight());
            size_t missingCommonTokenWeight(
                compCategory.missingCommonTokenWeight(m_WorkTokenUniqueIds));
            double proportionOfOrig(double(commonUniqueTokenWeight - missingCommonTokenWeight) /
                                    double(origUniqueTokenWeight));
            if (proportionOfOrig < m_LowerThreshold) {
                continue;
            }
        }

        double similarity(this->similarity(m_WorkTokenIds, workWeight, baseTokenIds, baseWeight));

        LOG_TRACE(<< similarity << '-' << compCategory.baseString() << '|' << str);

        if (matchesSearch || similarity > m_UpperThreshold) {
            if (similarity <= m_LowerThreshold) {
                // Not an ideal situation, but log at trace level to avoid
                // excessive log file spam
                LOG_TRACE(<< "Reverse search match below threshold : " << similarity
                          << '-' << compCategory.baseString() << '|' << str);
            }

            // This is a strong match, so accept it immediately and stop
            // looking for better matches - use vector index plus one as category
            int categoryId(1 + int(iter->second));
            this->addCategoryMatch(isDryRun, str, rawStringLen, m_WorkTokenIds,
                                   m_WorkTokenUniqueIds, similarity, iter);
            return categoryId;
        }

        if (similarity > bestSoFarSimilarity) {
            // This is a weak match, but remember it because it's the best we've
            // seen
            bestSoFarIter = iter;
            bestSoFarSimilarity = similarity;

            // Recalculate the minimum and maximum token counts that might
            // produce a better match
            minWeight = CBaseTokenListDataCategorizer::minMatchingWeight(workWeight, similarity);
            maxWeight = CBaseTokenListDataCategorizer::maxMatchingWeight(workWeight, similarity);
        }
    }

    if (bestSoFarIter != m_CategoriesByCount.end()) {
        // Return the best match - use vector index plus one as ML category
        int categoryId(1 + int(bestSoFarIter->second));
        this->addCategoryMatch(isDryRun, str, rawStringLen, m_WorkTokenIds,
                               m_WorkTokenUniqueIds, bestSoFarSimilarity, bestSoFarIter);
        return categoryId;
    }

    // If we get here we haven't matched, so create a new category
    CTokenListCategory obj(isDryRun, str, rawStringLen, m_WorkTokenIds,
                           workWeight, m_WorkTokenUniqueIds);
    m_CategoriesByCount.push_back(TSizeSizePr(1, m_Categories.size()));
    m_Categories.push_back(obj);
    m_HasChanged = true;

    // Increment the counts of categories that use a given token
    for (const auto& workTokenId : m_WorkTokenIds) {
        // We get away with casting away constness ONLY because the category count
        // is not used in any of the multi-index keys
        const_cast<CTokenInfoItem&>(m_TokenIdLookup[workTokenId.first]).incCategoryCount();
    }

    // ML category is vector index plus one
    return int(m_Categories.size());
}

bool CBaseTokenListDataCategorizer::createReverseSearch(int categoryId,
                                                        std::string& part1,
                                                        std::string& part2,
                                                        size_t& maxMatchingLength,
                                                        bool& wasCached) {
    if (m_ReverseSearchCreator == nullptr) {
        LOG_ERROR(<< "Cannot create reverse search - no reverse search creator");

        part1.clear();
        part2.clear();

        return false;
    }

    // Find the correct category object - ML category is vector index plus one
    if (categoryId < 1 || static_cast<size_t>(categoryId) > m_Categories.size()) {
        // -1 is a special case for a NULL/empty field
        if (categoryId != -1) {
            LOG_ERROR(<< "Programmatic error - invalid ML category: " << categoryId);

            part1.clear();
            part2.clear();

            return false;
        }

        return m_ReverseSearchCreator->createNullSearch(part1, part2);
    }

    CTokenListCategory& category = m_Categories[categoryId - 1];
    maxMatchingLength = category.maxMatchingStringLen();

    // If we can retrieve cached reverse search terms we'll save a lot of time
    if (category.cachedReverseSearch(part1, part2) == true) {
        wasCached = true;
        return true;
    }

    const TSizeSizePrVec& baseTokenIds = category.baseTokenIds();
    const TSizeSizePrVec& commonUniqueTokenIds = category.commonUniqueTokenIds();
    if (commonUniqueTokenIds.empty()) {
        // There's quite a high chance this call will return false
        if (m_ReverseSearchCreator->createNoUniqueTokenSearch(
                categoryId, category.baseString(),
                category.maxMatchingStringLen(), part1, part2) == false) {
            // More detail should have been logged by the failed call
            LOG_ERROR(<< "Could not create reverse search");

            part1.clear();
            part2.clear();

            return false;
        }

        category.cacheReverseSearch(part1, part2);

        return true;
    }

    size_t availableCost(m_ReverseSearchCreator->availableCost());

    // Determine the rarest tokens that we can afford within the available
    // length
    using TSizeSizeSizePrMMap = std::multimap<size_t, TSizeSizePr>;
    TSizeSizeSizePrMMap rareIdsWithCost;
    size_t lowestCost(std::numeric_limits<size_t>::max());
    for (const auto& commonUniqueTokenId : commonUniqueTokenIds) {
        size_t tokenId(commonUniqueTokenId.first);
        size_t occurrences(std::count_if(baseTokenIds.begin(), baseTokenIds.end(),
                                         CSizePairFirstElementEquals(tokenId)));
        const CTokenInfoItem& info = m_TokenIdLookup[tokenId];
        size_t cost(m_ReverseSearchCreator->costOfToken(info.str(), occurrences));
        rareIdsWithCost.insert(TSizeSizeSizePrMMap::value_type(
            info.categoryCount(), TSizeSizePr(tokenId, cost)));
        lowestCost = std::min(cost, lowestCost);
    }

    using TSizeSet = std::set<size_t>;
    TSizeSet costedCommonUniqueTokenIds;
    size_t cheapestCost(std::numeric_limits<size_t>::max());
    auto cheapestIter = rareIdsWithCost.end();
    for (auto iter = rareIdsWithCost.begin();
         iter != rareIdsWithCost.end() && availableCost > lowestCost; ++iter) {
        if (iter->second.second < cheapestCost) {
            cheapestCost = iter->second.second;
            cheapestIter = iter;
        }

        if (availableCost < iter->second.second) {
            // We can't afford this token
            continue;
        }

        // By this point we don't care about the weights or costs
        costedCommonUniqueTokenIds.insert(iter->second.first);
        availableCost -= iter->second.second;
    }

    if (costedCommonUniqueTokenIds.empty()) {
        if (cheapestIter == rareIdsWithCost.end()) {
            LOG_ERROR(<< "Inconsistency - rareIdsWithCost is empty but "
                         "commonUniqueTokenIds wasn't for "
                      << categoryId);
        } else {
            LOG_ERROR(<< "No token was short enough to include in reverse search "
                         "for "
                      << categoryId << " - cheapest token was "
                      << cheapestIter->second.first << " with cost " << cheapestCost);
        }

        part1.clear();
        part2.clear();

        return false;
    }

    // If we get here we're going to create a search in the standard way - there
    // shouldn't be any more errors after this point

    m_ReverseSearchCreator->initStandardSearch(categoryId, category.baseString(),
                                               category.maxMatchingStringLen(),
                                               part1, part2);

    for (auto costedCommonUniqueTokenId : costedCommonUniqueTokenIds) {
        m_ReverseSearchCreator->addCommonUniqueToken(
            m_TokenIdLookup[costedCommonUniqueTokenId].str(), part1, part2);
    }

    bool first(true);
    size_t end(category.outOfOrderCommonTokenIndex());
    for (size_t index = 0; index < end; ++index) {
        size_t tokenId(baseTokenIds[index].first);
        if (costedCommonUniqueTokenIds.find(tokenId) !=
            costedCommonUniqueTokenIds.end()) {
            m_ReverseSearchCreator->addInOrderCommonToken(
                m_TokenIdLookup[tokenId].str(), first, part1, part2);
            first = false;
        }
    }

    m_ReverseSearchCreator->closeStandardSearch(part1, part2);

    category.cacheReverseSearch(part1, part2);

    return true;
}

namespace {

class CPairFirstElementGreater {
public:
    //! This operator is designed for pairs that are small enough for
    //! passing by value to be most efficient
    template<typename PAIR>
    bool operator()(const PAIR pr1, const PAIR pr2) {
        return pr1.first > pr2.first;
    }
};
}

bool CBaseTokenListDataCategorizer::hasChanged() const {
    return m_HasChanged;
}

bool CBaseTokenListDataCategorizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_Categories.clear();
    m_CategoriesByCount.clear();
    m_TokenIdLookup.clear();
    m_WorkTokenIds.clear();
    m_WorkTokenUniqueIds.clear();
    m_HasChanged = false;

    do {
        const std::string& name = traverser.name();
        if (name == TOKEN_TAG) {
            size_t nextIndex(m_TokenIdLookup.size());
            m_TokenIdLookup.push_back(CTokenInfoItem(traverser.value(), nextIndex));
        } else if (name == TOKEN_CATEGORY_COUNT_TAG) {
            if (m_TokenIdLookup.empty()) {
                LOG_ERROR(<< "Token category count precedes token string in "
                          << traverser.value());
                return false;
            }

            size_t categoryCount(0);
            if (core::CStringUtils::stringToType(traverser.value(), categoryCount) == false) {
                LOG_ERROR(<< "Invalid token category count in " << traverser.value());
                return false;
            }

            // We get away with casting away constness ONLY because the category
            // count is not used in any of the multi-index keys
            const_cast<CTokenInfoItem&>(m_TokenIdLookup.back()).categoryCount(categoryCount);
        } else if (name == CATEGORY_TAG) {
            CTokenListCategory category(traverser);
            TSizeSizePr countAndIndex(category.numMatches(), m_Categories.size());
            m_Categories.push_back(category);
            m_CategoriesByCount.push_back(countAndIndex);
        }
    } while (traverser.next());

    // Categories are persisted in order of creation, but this list needs to be
    // sorted by count instead
    m_CategoriesByCount.sort(CPairFirstElementGreater());

    return true;
}

void CBaseTokenListDataCategorizer::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    CBaseTokenListDataCategorizer::acceptPersistInserter(m_TokenIdLookup,
                                                         m_Categories, inserter);
}

void CBaseTokenListDataCategorizer::acceptPersistInserter(const TTokenMIndex& tokenIdLookup,
                                                          const TTokenListCategoryVec& categories,
                                                          core::CStatePersistInserter& inserter) {
    for (const CTokenInfoItem& item : tokenIdLookup) {
        inserter.insertValue(TOKEN_TAG, item.str());
        inserter.insertValue(TOKEN_CATEGORY_COUNT_TAG, item.categoryCount());
    }

    for (const CTokenListCategory& category : categories) {
        inserter.insertLevel(CATEGORY_TAG,
                             std::bind(&CTokenListCategory::acceptPersistInserter,
                                       &category, std::placeholders::_1));
    }
}

CDataCategorizer::TPersistFunc CBaseTokenListDataCategorizer::makePersistFunc() const {
    return std::bind(
        static_cast<void (*)(const TTokenMIndex&, const TTokenListCategoryVec&, core::CStatePersistInserter&)>(
            &CBaseTokenListDataCategorizer::acceptPersistInserter),
        std::cref(m_TokenIdLookup), std::cref(m_Categories), std::placeholders::_1);
}

void CBaseTokenListDataCategorizer::addCategoryMatch(bool isDryRun,
                                                     const std::string& str,
                                                     size_t rawStringLen,
                                                     const TSizeSizePrVec& tokenIds,
                                                     const TSizeSizeMap& tokenUniqueIds,
                                                     double similarity,
                                                     TSizeSizePrListItr& iter) {
    if (m_Categories[iter->second].addString(isDryRun, str, rawStringLen, tokenIds,
                                             tokenUniqueIds, similarity) == true) {
        m_HasChanged = true;
    }

    size_t& count = iter->first;
    ++count;

    // Search backwards for the point where the incremented count belongs
    TSizeSizePrListItr swapIter(m_CategoriesByCount.end());
    TSizeSizePrListItr checkIter(iter);
    while (checkIter != m_CategoriesByCount.begin()) {
        --checkIter;
        if (count <= checkIter->first) {
            break;
        }
        swapIter = checkIter;
    }

    // Move the iterator we've matched nearer the front of the list if it
    // deserves this
    if (swapIter != m_CategoriesByCount.end()) {
        std::iter_swap(swapIter, iter);
    }
}

size_t CBaseTokenListDataCategorizer::minMatchingWeight(size_t weight, double threshold) {
    if (weight == 0) {
        return 0;
    }

    // When we build with aggressive optimisation, the result of the floating
    // point multiplication can be slightly out, so add a small amount of
    // tolerance
    static const double EPSILON(0.00000000001);

    // This assumes threshold is not negative - other code in this file must
    // enforce this.  Using floor + 1 due to threshold check being exclusive.
    // If threshold check is changed to inclusive, change formula to ceil
    // (without the + 1).
    return static_cast<size_t>(std::floor(double(weight) * threshold + EPSILON)) + 1;
}

size_t CBaseTokenListDataCategorizer::maxMatchingWeight(size_t weight, double threshold) {
    if (weight == 0) {
        return 0;
    }

    // When we build with aggressive optimisation, the result of the floating
    // point division can be slightly out, so subtract a small amount of
    // tolerance
    static const double EPSILON(0.00000000001);

    // This assumes threshold is not negative - other code in this file must
    // enforce this.  Using ceil - 1 due to threshold check being exclusive.
    // If threshold check is changed to inclusive, change formula to floor
    // (without the - 1).
    return static_cast<size_t>(std::ceil(double(weight) / threshold - EPSILON)) - 1;
}

size_t CBaseTokenListDataCategorizer::idForToken(const std::string& token) {
    auto iter = boost::multi_index::get<SToken>(m_TokenIdLookup).find(token);
    if (iter != boost::multi_index::get<SToken>(m_TokenIdLookup).end()) {
        return iter->index();
    }

    size_t nextIndex(m_TokenIdLookup.size());
    m_TokenIdLookup.push_back(CTokenInfoItem(token, nextIndex));
    return nextIndex;
}

bool CBaseTokenListDataCategorizer::addPretokenisedTokens(const std::string& tokensCsv,
                                                          TSizeSizePrVec& tokenIds,
                                                          TSizeSizeMap& tokenUniqueIds,
                                                          size_t& totalWeight) {
    tokenIds.clear();
    tokenUniqueIds.clear();
    totalWeight = 0;

    m_CsvLineParser.reset(tokensCsv);
    std::string token;
    while (!m_CsvLineParser.atEnd()) {
        if (m_CsvLineParser.parseNext(token) == false) {
            return false;
        }

        this->tokenToIdAndWeight(token, tokenIds, tokenUniqueIds, totalWeight);
    }

    return true;
}

CBaseTokenListDataCategorizer::CTokenInfoItem::CTokenInfoItem(const std::string& str, size_t index)
    : m_Str(str), m_Index(index), m_CategoryCount(0) {
}

const std::string& CBaseTokenListDataCategorizer::CTokenInfoItem::str() const {
    return m_Str;
}

size_t CBaseTokenListDataCategorizer::CTokenInfoItem::index() const {
    return m_Index;
}

size_t CBaseTokenListDataCategorizer::CTokenInfoItem::categoryCount() const {
    return m_CategoryCount;
}

void CBaseTokenListDataCategorizer::CTokenInfoItem::categoryCount(size_t categoryCount) {
    m_CategoryCount = categoryCount;
}

void CBaseTokenListDataCategorizer::CTokenInfoItem::incCategoryCount() {
    ++m_CategoryCount;
}

CBaseTokenListDataCategorizer::CSizePairFirstElementEquals::CSizePairFirstElementEquals(size_t value)
    : m_Value(value) {
}

CBaseTokenListDataCategorizer::SIdTranslater::SIdTranslater(const CBaseTokenListDataCategorizer& categorizer,
                                                            const TSizeSizePrVec& tokenIds,
                                                            char separator)
    : s_Categorizer(categorizer), s_TokenIds(tokenIds), s_Separator(separator) {
}

std::ostream& operator<<(std::ostream& strm,
                         const CBaseTokenListDataCategorizer::SIdTranslater& translator) {
    for (auto iter = translator.s_TokenIds.begin();
         iter != translator.s_TokenIds.end(); ++iter) {
        if (iter != translator.s_TokenIds.begin()) {
            strm << translator.s_Separator;
        }

        if (iter->first < translator.s_Categorizer.m_TokenIdLookup.size()) {
            strm << translator.s_Categorizer.m_TokenIdLookup[iter->first].str();
        } else {
            strm << "Out of bounds!";
        }
    }

    return strm;
}
}
}
