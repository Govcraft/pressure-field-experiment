# JAAMAS Information Sheet

**Paper Title:** Emergent Coordination in Multi-Agent Systems via Pressure Fields and Temporal Decay

**Author:** Roland R. Rodriguez, Jr.

**Submission to:** Special Issue "When Foundation Models Meet Multi-Agent Systems"

---

## 1. What is the main claim of the paper? Why is this an important contribution to the autonomous agents and multi-agent systems literature?

**Main Claim:** Implicit coordination through shared pressure gradients dramatically outperforms explicit hierarchical coordination and conversation-based approaches for multi-agent constraint satisfaction tasks, achieving 30x higher solve rates than hierarchical control and 4x higher than conversation-based coordination.

**Why This Matters for MAS:**

The paper challenges a foundational assumption in multi-agent systems: that explicit coordination mechanisms (message passing, role assignment, intention sharing) are necessary for effective collaboration. We demonstrate empirically that for artifact refinement tasks with measurable quality signals, the coordination overhead of explicit approaches actively harms performance.

This contribution is important for three reasons:

**(a) Theoretical Integration:** The paper provides the first formal connection between stigmergic coordination (from swarm intelligence) and foundation model capabilities, showing how FMs' broad pretraining solves the action enumeration problem that historically limited stigmergic approaches to discrete, enumerable solution spaces. This extends MAS coordination theory to open-ended artifact refinement tasks.

**(b) Architectural Simplification:** Pressure-field coordination eliminates roles (unlike organizational paradigms surveyed by Horling & Lesser), messages (unlike GPGP), and intention reasoning (unlike SharedPlans and Joint Intentions) while providing formal convergence guarantees through potential game theory. This demonstrates that for an important class of problems, the complexity of traditional MAS architectures is unnecessary.

**(c) FM-MAS Reciprocity:** The paper articulates a bidirectional synthesis: FMs solve the MAS action enumeration problem (enabling stigmergic coordination for unbounded improvement spaces), while MAS coordination mechanisms solve the FM output combination problem (replacing ad-hoc voting with principled pressure-based selection). This reciprocity framework offers design principles for future FM-MAS integration.

---

## 2. What is the evidence you provide to support your claim? Be precise.

**Empirical Evidence (1350 trials):**

- **Scale:** 1350 total trials across 5 coordination strategies, 3 difficulty levels, and 3 agent counts (1, 2, 4 agents)
- **Primary Result:** Pressure-field achieves 48.5% aggregate solve rate (131/270) versus conversation 11.1% (30/270), hierarchical 1.5% (4/270), sequential 0.4% (1/270), and random 0.4% (1/270)
- **Statistical Significance:** All pairwise comparisons p < 0.001; Chi-square across all strategies chi-squared > 200
- **Effect Sizes:** Cohen's h = 1.16 (vs conversation), h > 1.97 (vs hierarchical/sequential/random)—all exceed "large effect" threshold
- **Difficulty Scaling:** Pressure-field is the only strategy achieving non-zero solve rates on medium (43.3%) and hard (15.6%) problems; all baselines achieve 0%
- **Convergence Speed:** Pressure-field solves 1.65x faster than conversation, 2.2x faster than hierarchical on easy problems
- **Token Efficiency:** Despite higher per-trial cost, pressure-field achieves 12% better token efficiency per successful solve (1.27M vs 1.45M tokens/solve)

**Ablation Studies (150 trials):**

- Temporal decay shows +10 percentage point improvement (96.7% vs 86.7%), consistent with theoretical predictions about escaping local minima, though not statistically significant at n=30
- Inhibition shows no detectable effect in this domain
- Few-shot examples contribute +6.7%

**Theoretical Evidence:**

- **Theorem 1 (Convergence):** Proves convergence to stable basins under pressure alignment with bounded coupling, with explicit bounds on convergence time
- **Theorem 3 (Basin Separation):** Explains why temporal decay is necessary to escape suboptimal basins
- **Theorem 4 (Linear Scaling):** Establishes O(1) coordination overhead independent of agent count
- **Appendix B:** Empirically verifies pressure alignment (epsilon = 0 separability) through analysis of 9,873 tick-to-tick transitions showing zero pressure degradation events

**Baseline Fairness:**

- All strategies use identical LLM models (qwen2.5:0.5b/1.5b/3b)
- Identical prompts and parsing logic across strategies
- Same problem instances via deterministic seeding
- Same tick budget (50 ticks per trial)

---

## 3. What papers by other authors make the most closely related contributions, and how is your paper related to them?

**Multi-Agent Coordination Frameworks:**

- Horling, B., & Lesser, V. (2004). A survey of multi-agent organizational paradigms. *The Knowledge Engineering Review*, 19(4), 281-316.
  - Surveys organizational paradigms (hierarchies, holarchies, teams, coalitions); our work demonstrates these structures are unnecessary for constraint satisfaction tasks with measurable quality signals.

- Decker, K., & Lesser, V. (1995). Designing a family of coordination algorithms. *Proceedings of the First International Conference on Multi-Agent Systems (ICMAS-95)*, 73-80.
  - Introduces GPGP with explicit coordination through task structures and commitments; we achieve coordination without any inter-agent messages.

- Grosz, B. J., & Kraus, S. (1996). Collaborative plans for complex group action. *Artificial Intelligence*, 86(2), 269-357.
  - SharedPlans requires agents to reason about mutual beliefs and intentions; pressure-field eliminates intention reasoning entirely.

**Foundation Model Multi-Agent Systems:**

- Wu, Q., Bansal, G., Zhang, J., et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv:2308.08155*.
  - Conversation-based FM coordination through multi-turn dialogue; we compare directly and demonstrate 4x lower solve rates due to coordination overhead.

- Hong, S., Zhuge, M., Chen, J., et al. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv:2308.00352*.
  - Role-based FM coordination with SOPs; our approach eliminates roles and explicit protocols entirely.

- Li, G., Hammoud, H., Itani, H., et al. (2023). CAMEL: Communicative agents for "mind" exploration of large language model society. *Advances in Neural Information Processing Systems*, 36.
  - Studies emergent behaviors in FM agent conversations; we show that for constraint satisfaction, implicit coordination outperforms conversational approaches.

**Stigmergic Coordination:**

- Dorigo, M., & Stutzle, T. (2004). *Ant Colony Optimization*. MIT Press.
  - Classic stigmergic optimization requiring discrete, enumerable action spaces; FMs overcome this limitation through broad pretraining.

- Theraulaz, G., & Bonabeau, E. (1999). A brief history of stigmergy. *Artificial Life*, 5(2), 97-116.
  - Theoretical foundations of stigmergy; we extend these principles to FM-based artifact refinement.

**Relationship Summary:** Unlike GPGP which requires explicit coordination messages, SharedPlans which requires intention reasoning, and AutoGen which requires multi-turn conversation, pressure-field coordination achieves effective collaboration through shared artifact state alone. We provide the first empirical comparison demonstrating that this implicit approach outperforms explicit coordination for FM-based constraint satisfaction systems.

---

## 4. Have you published parts of your paper before? If so, give details and a precise statement of added value.

An earlier version of this paper was posted to arXiv (January 2026) as a preprint. The JAAMAS submission is the first submission to a peer-reviewed venue. No prior conference publication exists.

The arXiv preprint and this submission contain the same core experimental results and theoretical analysis. This submission incorporates additional discussion of FM-MAS reciprocity (Section 7.7) specifically framed for the special issue theme.

---

## 5. Fit with Special Issue Theme

This paper directly addresses the special issue theme "When Foundation Models Meet Multi-Agent Systems" by demonstrating bidirectional enablement:

1. **FM capabilities enable MAS coordination:** Broad pretraining, instruction-following, and zero-shot reasoning allow FMs to propose quality-improving patches from local pressure signals alone, solving the action enumeration problem that historically limited stigmergic coordination to discrete solution spaces.

2. **MAS mechanisms enable FM output combination:** Pressure gradients and temporal decay provide principled frameworks for combining multiple FM outputs, replacing ad-hoc voting or ranking with objective quality-based selection.

Section 7.7 (FM-MAS Reciprocity) explicitly articulates this bidirectional synthesis and derives three design principles for future FM-MAS integration:
- Leverage FM coverage, constrain via MAS gradients
- Prefer stigmergic over explicit coordination when FMs serve as actors
- Design pressure functions as the critical FM-MAS interface
