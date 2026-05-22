# Stylised Facts Used in the Simulator

The project uses simulated data only. The default cube is scaled to common
interest-rate volatility stylised facts rather than to a live broker screen.

- ATM normal swaption vol is anchored around 60-90bp in the front and around
  80bp in the back, then converted into a Black-vol scale for SABR pricing.
- The first three ATM surface modes are level, expiry slope, and humped
  curvature. This is consistent with the low-dimensional factor structure
  discussed in Rebonato-style term-structure volatility modelling.
- SABR beta is fixed at 0.5. Rho is bounded between -0.95 and 0.0. Log-nu is
  mean reverting and correlated with the ATM level shock.
- Skew steepens when the level factor rises. The model encodes that with a
  negative rho-level correlation and a positive nu-level correlation.

Primary references: Andersen and Piterbarg (2010), Hagan et al. (2002),
Rebonato (2002), Bergomi (2016), Bartlett (2006), Gatheral (2006), and
Avellaneda and Stoikov (2008).
