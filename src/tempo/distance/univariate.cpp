#include "univariate.private.hpp"

namespace tempo::distance::univariate {

  // Implementation through template explicit instantiation

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Double implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using F = double;

  // --- --- --- Elastic distances --- --- ---

  template F adtw<F>(F const *, size_t, F const *, size_t, F, F, F);

  template F dtw<F>(F const *, size_t, F const *, size_t, F, size_t, F);

  template F wdtw<F>(F const *, size_t, F const *, size_t, F, F const *, F);
  template void wdtw_weights(F, F *, size_t, F);
  template std::vector<F> wdtw_weights(F, size_t, F);

  template F erp<F>(F const *, size_t, F const *, size_t, F, F, size_t, F);

  template F lcss<F>(F const *, size_t, F const *, size_t, F, size_t, F);

  template F msm<F>(F const *, size_t, F const *, size_t, F, F);

  template F twe<F>(F const *, size_t, F const *, size_t, F, F, F);

  // --- --- --- DTW Lower bounds --- --- ---

  template F lb_Keogh<F>(F const *, size_t, F const *, F const *, F, F);

  template F lb_Keogh2j<F>(F const *, size_t, F const *, F const *, F const *, size_t, F const *, F const *, F, F);

  template void get_keogh_envelopes<F>(F const *, size_t, F *, F *, size_t);

  template void get_keogh_up_envelope<F>(F const *, size_t, F *, size_t);

  template void get_keogh_lo_envelope<F>(F const *, size_t, F *, size_t);

  template F lb_Enhanced<F>(const F *, size_t, const F *, size_t, const F *, const F *, F, size_t, size_t, F);

  template F lb_Enhanced2j<F>(
    const F *, size_t, const F *, const F *, const F *, size_t, const F *, const F *, F, size_t, size_t, F
  );

  template F lb_Webb<F>(
    F const *a, size_t a_len, F const *a_up, F const *a_lo, F const *a_lo_up, F const *a_up_lo,
    F const *b, size_t b_len, F const *b_up, F const *b_lo, F const *b_lo_up, F const *b_up_lo,
    F cfe, size_t w, F cutoff
  );

  template void get_keogh_envelopes_Webb<F>(
    F const *series, size_t length, F *upper, F *lower, F *lower_upper, F *upper_lower, size_t w
  );

  // --- --- --- Lock Step distances --- --- ---

  template F directa<F>(F const *, size_t, F const *, size_t, F, F);

} // End of namespace tempo::distance:univariate
