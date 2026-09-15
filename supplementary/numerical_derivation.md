# Numerical enclosure construction

This note describes the implemented conditional calculation. Branching recursions, monotone fixed-point bounds and contraction arguments have prior art, including Hawkes and Oakes (1974), DOI10.2307/3212693, and Møller and Rasmussen (2005), DOI10.1239/aap/1127483739. This implementation applies them to the threshold event and propagates the resulting numerical bounds into empirical score comparisons. It does not establish that ETAS is the physical earthquake-generating process.

## Model and family probability

Let rho(m)=beta exp[-beta(m-m0)], beta=b ln10, m>=m0. Productivity is kappa(m)=K0 exp[alpha ln10(m-m0)]. The lag density is normalized Omori with survival S(t)=(1+t/c)^(1-p), c>0,p>1. Future marks and lags are independent across offspring. Require b>alpha and n=K0*b/(b-alpha)<1. Conditional immigrant intensity is b0(t)=mu+sum over fixed historical events kappa(mi)g(t-ti), with ti<0.

For threshold h>=m0, f(t) is the probability of no threshold event in a family through remaining horizon t, including its root:

    f(t) = integral_[m0,h) rho(m) exp{kappa(m) integral_0^t g(s)[f(t-s)-1] ds} dm.

An above-threshold root makes this event impossible. The magnitude integral therefore retains its original, unconditional mass1-q, where q=10^[-b(h-m0)]. It is not renormalized. Conditional zero-target probability is

    P0(H) = exp{integral_0^H b0(s)[f(H-s)-1] ds}.

The desired exceedance probability is1-P0(H). Above-threshold historical magnitudes retain their actual productivity in b0; only future target encounters are absorbing.

## Rounded processes and computable bounds

Use K time bins of width Delta=H/K and J magnitude bins with edges a0=m0,...,aJ=h. Let pi_j=exp[-beta(a_j-m0)]-exp[-beta(a_(j+1)-m0)], and let kappa_j^- and kappa_j^+ be the lower and upper endpoint productivities. Let w_l=S(l Delta)-S((l+1)Delta), and B_j=integral_[j Delta,(j+1)Delta) b0(s)ds. The code evaluates B_j analytically as mu Delta plus historical productivity times differences of S.

For early rounding and upper productivity, the family recursion at time index k is the fixed point

    F_early(k) = sum_j pi_j exp{kappa_j^+ [w_0(F_early(k)-1)
                         + sum_(l=1..k) w_l(F_early(k-l)-1)]}.

For late rounding and lower productivity,

    F_late(k) = sum_j pi_j exp{kappa_j^- sum_(l=1..k) w_(l-1)(F_late(k-l)-1)}.

Empty sums are zero. Late rounding has no zero-lag reproduction; F_late(0)=1-q. Early rounding includes zero-lag cascades. Its derivative on[0,1] is bounded by w0*sum_j pi_j*kappa_j^+. The implementation requires the stronger sufficient condition sum_j pi_j*kappa_j^+<1, certified in outward arithmetic. It iterates an interval initially covering[0,1] and retains an enclosure of the unique fixed point. Failure to reach the internal radius tolerance is recorded rather than converted into an unsupported point estimate.

To justify the ordering, couple candidate offspring using common marks and Poisson thinning. Increasing subthreshold productivity can add descendants; decreasing it can only remove them. Absorb the family at the first target encounter. Rounding each nonnegative lag and immigrant time down can only move a retained target encounter earlier; rounding up can only delay it. Thus the early/high-productivity family supplies a lower no-target probability, and the late/low-productivity family supplies an upper one. Subcriticality and the certified upper non-target reproduction mean ensure finite relevant genealogies and the unique early fixed point. Horizon boundary conventions agree almost surely under the continuous-time model.

The implemented bounds are

    exp{sum_(j=0..K-1) B_j [F_early(K-j)-1]} <= P0(H)
        <= exp{sum_(j=0..K-1) B_j [F_late(K-j-1)-1]}.

If h=m0, all future events are targets and P0(H)=exp[-sum_j B_j]. No descendants need be generated. This special case is distinct from the unbounded magnitude law being truncated.

All integrals, fixed-point intervals and final endpoints use160-bit Arb arithmetic. Endpoints are saved as dyadics or exact rational numbers. Adaptive refinement intersects valid enclosures and compares their exact rational width with twice the requested midpoint tolerance. A resource limit can leave a valid wider interval; it does not imply accuracy at the requested tolerance.

## Matched-mean Poisson comparator

An event's own future mark is independent of its existence and birth time, although it affects descendants. Therefore the expected target count is q times the expected total count. This identity does not make the target process Poisson.

Expected rounded family sizes, including their root, satisfy

    E_early(k) = [1+n sum_(l=1..k) w_l E_early(k-l)]/(1-n*w0),
    E_late(k)  = 1+n sum_(l=1..k) w_(l-1) E_late(k-l).

Integrating against B_j gives total-count bounds sum_j B_j E_late(K-j-1) and sum_j B_j E_early(K-j). Intersect with[B,B/(1-n)], B=sum_j B_j, then multiply by q. Transform a target-mean interval[L,U] into[1-exp(-L),1-exp(-U)]. This encloses the matched-mean Poisson approximation, not the true branching occurrence probability. Mark integration is analytic for this mean, so it has no mark-grid approximation.

## Structural probability ordering

For an immigrant arriving at s, let Y_s be its nonnegative target-event count within the remaining horizon. Poisson superposition gives P(N=0)=exp[-integral b0(s)P(Y_s>0)ds] and E[N]=integral b0(s)E[Y_s]ds. Because P(Y_s>0)<=E[Y_s], the branching exceedance probability is no larger than1-exp(-E[N]) under this fixed-history model. This established cluster consequence is not an empirical discovery and does not imply better calibration or Brier scores under model misspecification. A branching lower bound exceeding the matched-mean upper bound would violate this necessary consistency condition.

## Score interpretation

For outcome y=1 and probability[l,u], Brier bounds are[(1-u)^2,(1-l)^2]; for y=0 they are[l^2,u^2]. Subtract the comparator's upper loss from the branching lower loss for the contrast lower bound, and conversely for the upper bound. Average with exact rational arithmetic. The bootstrap resamples these paired bound series at common circular blocks. Its floating-point percentile envelope is a sampling summary conditional on the saved forecasts, not an outward arithmetic or exact coverage theorem. Parameter, catalog, magnitude-law and operational-availability uncertainties remain outside these guarantees.
