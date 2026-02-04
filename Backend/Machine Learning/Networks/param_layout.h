#pragma once
/*
 * Parameter layout canonicalization.
 *
 * This codebase historically stores biases as a "final edge weight" on each node
 * (index == fanIn), and for gated recurrent units stores gate blocks contiguously:
 *   - Node edges:  gateCount * (fanIn + 1)   (includes a per-gate bias edge)
 *   - Context edges: gateCount * hiddenSize  (recurrent matrix rows per gate)
 *
 * Those conventions were duplicated across many translation units, which is fragile.
 * This header is the single source of truth for parameter indexing.
 *
 * IMPORTANT: This is a *layout* helper only. It does not change model semantics.
 */

namespace glades {
namespace param_layout {

// Dense (feedforward) node edges:
//   [0..fanIn-1] weights, [fanIn] bias
inline unsigned dense_weight_edge(unsigned inIndex) { return inIndex; }
inline unsigned dense_bias_edge(unsigned fanIn) { return fanIn; }

// Simple RNN hidden node:
// - Node edges: [0..fanIn-1] Wx, [fanIn] bias
// - Context node edges: [0..hiddenSize-1] Wh row
inline unsigned rnn_wx_edge(unsigned inIndex) { return inIndex; }
inline unsigned rnn_bias_edge(unsigned fanIn) { return fanIn; }
inline unsigned rnn_wh_edge(unsigned hiddenIndex) { return hiddenIndex; }

// Gated recurrent layouts (GRU/LSTM):
// - Node edges are partitioned by gate, with stride = (fanIn + 1)
//   gate g:
//     Wg[p] at edge = g*stride + p, for p in [0..fanIn-1]
//     bg    at edge = g*stride + fanIn
// - Context node edges are partitioned by gate, length hiddenSize per gate:
//   Ug[j] at edge = g*hiddenSize + j
struct Gated
{
	unsigned fanIn;
	unsigned hiddenSize;
	unsigned gateCount;

	inline unsigned stride() const { return fanIn + 1u; }

	inline unsigned w_edge(unsigned gate, unsigned inIndex) const
	{
		return gate * stride() + inIndex;
	}
	inline unsigned b_edge(unsigned gate) const
	{
		return gate * stride() + fanIn;
	}
	inline unsigned u_edge(unsigned gate, unsigned hiddenIndex) const
	{
		return gate * hiddenSize + hiddenIndex;
	}
};

} // namespace param_layout
} // namespace glades

