# v8_candidate_solution_a
Candidate Solution A Optimized: Condition-Adaptive Mixture + Hierarchical Calibration.
Mixture:
- steady machines: 15% Config 3 + 85% Champion v6
- balanced machines: 28% Config 3 + 72% Champion v6
- jackpot machines: 35% Config 3 + 65% Champion v6
Followed by L2-regularized Hierarchical Calibration (lambda_base=0.01, lambda_hier=0.08).

Validation Scores:
- Full Points: 38.4227 / 60
- Tuning Points: 38.4566 / 60
- Audit Points: 38.3854 / 60
- Loss Ratio: 0.5616 (43.84% error reduction vs frozen starter reference)
- Paired 95% Bootstrap CI vs Champion v6: [+0.1040, +0.6916]
