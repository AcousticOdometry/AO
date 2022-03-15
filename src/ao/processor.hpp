#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <vector>

namespace ao {
namespace processor {

// Base class for all processors
class BaseProcessor {};

// audio frame goes in -> feature frame goes out
class GammatoneFilterbank {
    public:
    std::vector<int> operator()(std::vector<int>& a);
};

} // namespace processor
} // namespace ao

#endif // PROCESSOR_HPP