#include "phy_graph/graph_utils.h"

#include <algorithm>
#include <cctype>

namespace phy_graph {
namespace utils {

std::string normalizeLabel(const std::string& label) {
    std::string result = label;

    // 去首尾空格
    auto not_space = [](int ch) { return !std::isspace(ch); };
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), not_space));
    result.erase(std::find_if(result.rbegin(), result.rend(), not_space).base(), result.end());

    // 转小写并将空格/连字符替换为下划线
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        if (c == ' ' || c == '-') return '_';
        return static_cast<char>(std::tolower(c));
    });

    // 合并连续下划线
    std::string compact;
    compact.reserve(result.size());
    bool prev_underscore = false;
    for (char c : result) {
        if (c == '_') {
            if (!prev_underscore) {
                compact.push_back(c);
                prev_underscore = true;
            }
        } else {
            compact.push_back(c);
            prev_underscore = false;
        }
    }

    // 去掉首尾下划线
    while (!compact.empty() && compact.front() == '_') compact.erase(compact.begin());
    while (!compact.empty() && compact.back() == '_') compact.pop_back();

    return compact;
}

std::string applySynonyms(
    const std::string& normalized_label,
    const std::unordered_map<std::string, std::string>& synonym_map) {
    auto it = synonym_map.find(normalized_label);
    if (it != synonym_map.end()) {
        return it->second;
    }
    return normalized_label;
}

}  // namespace utils
}  // namespace phy_graph

