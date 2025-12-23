#pragma once

#include <string>
#include <unordered_map>

namespace phy_graph {
namespace utils {

/**
 * @brief 将标签标准化：去首尾空格、转小写，并将空格/连字符替换为下划线。
 */
std::string normalizeLabel(const std::string& label);

/**
 * @brief 应用同义词映射，若存在映射则返回映射后的标签。
 */
std::string applySynonyms(
    const std::string& normalized_label,
    const std::unordered_map<std::string, std::string>& synonym_map);

} // namespace utils
} // namespace phy_graph

