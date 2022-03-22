#include "extractor.hpp"

// This gammatone filter is based on the implementation by Ning Ma from
// University of Sheffield who, in turn, based his implementation on an
// original algorithm from Martin Cooke's Ph.D thesis (Cooke, 1993) using
// the base-band impulse invariant transformation. This implementation is
// highly efficient in that a mathematical rearrangement is used to
// significantly reduce the cost of computing complex exponential. For
// more detail on this implementation see
//   http://www.dcs.shef.ac.uk/~ning/resources/gammatone/
//
// Note: Martin Cooke's PhD has been reprinted as M. Cooke (1993): "Modelling
// Auditory Processing and Organisation", Cambridge University Press, Series
// "Distinguished Dissertations in Computer Science", August.