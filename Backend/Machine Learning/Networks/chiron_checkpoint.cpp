// chiron_checkpoint.cpp — CHRN/CHRF format module: struct ctors/dtors + stubs.
//
// Task 2: skeleton only.  Real implementations land in Tasks 3–4.
// Every function stub returns false; chiron_load_model also sets err/errCode.
//
// C++98.

#include "chiron_checkpoint.h"

namespace glades {
namespace chiron {

// ---------------------------------------------------------------------------
// ChironScfaState
// ---------------------------------------------------------------------------

ChironScfaState::ChironScfaState()
    : k(0), w(0), present(false), dLoaded(false)
{
}

ChironScfaState::~ChironScfaState()
{
    for (size_t i = 0; i < D.size(); ++i)
        delete D[i];
    D.clear();
}

// ---------------------------------------------------------------------------
// ChironModelWeights
// ---------------------------------------------------------------------------

ChironModelWeights::ChironModelWeights()
    : hasADrift(false)
{
}

ChironModelWeights::~ChironModelWeights()
{
    for (size_t i = 0; i < Wq.size();    ++i) delete Wq[i];
    for (size_t i = 0; i < Wk.size();    ++i) delete Wk[i];
    for (size_t i = 0; i < Wv.size();    ++i) delete Wv[i];
    for (size_t i = 0; i < Wo.size();    ++i) delete Wo[i];
    for (size_t i = 0; i < gamma.size(); ++i) delete gamma[i];
    for (size_t i = 0; i < beta.size();  ++i) delete beta[i];
    for (size_t i = 0; i < gamma_p.size(); ++i) delete gamma_p[i];
    for (size_t i = 0; i < beta_p.size(); ++i) delete beta_p[i];
}

// ---------------------------------------------------------------------------
// Block codec stubs
// ---------------------------------------------------------------------------

bool chiron_read_block(std::FILE* /*fp*/, std::vector<float>& /*fp32buf*/,
                       size_t /*n*/, bool /*bf16OnDisk*/)
{
    return false;
}

bool chiron_write_block(std::FILE* /*fp*/, const std::vector<float>& /*fp32buf*/,
                        size_t /*n*/, bool /*bf16OnDisk*/)
{
    return false;
}

// ---------------------------------------------------------------------------
// Header writer stub
// ---------------------------------------------------------------------------

bool chiron_write_header(std::FILE* /*fp*/, const ChironCkptHeader& /*h*/)
{
    return false;
}

// ---------------------------------------------------------------------------
// Model-section writer stubs
// ---------------------------------------------------------------------------

bool chiron_write_scfa_section(std::FILE* /*fp*/, int /*k*/, int /*w*/,
                               const std::vector<const float*>& /*D_host*/,
                               size_t /*D_sz*/)
{
    return false;
}

bool chiron_write_qknorm_section(std::FILE* /*fp*/, const float* /*gammaHost*/,
                                 size_t /*n*/)
{
    return false;
}

bool chiron_write_f32_tail_section(std::FILE* /*fp*/, const float* /*vals*/,
                                   size_t /*n*/)
{
    return false;
}

// ---------------------------------------------------------------------------
// Serving reader stub
// ---------------------------------------------------------------------------

bool chiron_load_model(const std::string& /*path*/, ChironModelDims& /*dims*/,
                       ChironModelWeights& /*w*/, std::string& err, int& errCode)
{
    err     = "unimplemented";
    errCode = 4;
    return false;
}

} // namespace chiron
} // namespace glades
