#import "template.typ": *

#show: neurips.with(
  title: [Emergent Coordination in Multi-Agent Systems via Gradient Fields and Temporal Decay],
  authors: (
    (
      name: "Roland Rodriguez",
      affiliation: "Independent Researcher",
    ),
  ),
  abstract: [
    Current multi-agent LLM frameworks rely on explicit orchestration patterns borrowed from human organizational structures: planners delegate to executors, managers coordinate workers, and hierarchical control flow governs agent interactions. These approaches suffer from coordination overhead that scales poorly with agent count and task complexity. We propose a fundamentally different paradigm inspired by natural coordination mechanisms: agents operate locally on a shared artifact, guided only by pressure gradients derived from measurable quality signals, with temporal decay preventing premature convergence. We formalize this as optimization over a pressure landscape and prove convergence guarantees under mild conditions. Empirically, we demonstrate that this approach achieves competitive quality with significantly reduced coordination overhead on [benchmark tasks], scaling linearly with agent count where hierarchical approaches plateau. Our results suggest that constraint-driven emergence, rather than explicit planning, may be a more scalable foundation for multi-agent AI systems.
  ],
  keywords: (
    "multi-agent systems",
    "emergent coordination",
    "decentralized optimization",
    "LLM agents",
  ),
)

= Introduction

Multi-agent systems built on large language models have emerged as a promising approach to complex task automation @wu2023autogen @hong2023metagpt @li2023camel. The dominant paradigm treats agents as organizational units: planners decompose tasks, managers delegate subtasks, and workers execute instructions under hierarchical supervision. This mirrors human project management but imports its coordination costs.

We argue this design choice is fundamentally limiting. Human organizations evolved coordination mechanisms under constraints—limited communication bandwidth, cognitive load, trust verification—that do not apply to software agents. Importing these mechanisms into agent systems introduces artificial bottlenecks: central planners become serialization points, global state synchronization creates contention, and hierarchical message passing adds latency proportional to tree depth.

Natural systems that coordinate at scale—ant colonies, immune systems, neural tissue, markets—use a radically different approach. They coordinate through *environment modification* rather than message passing, rely on *local decisions* rather than global planning, and achieve stability through *continuous pressure* rather than explicit goals.

Our contributions:

+ We formalize *gradient-field coordination*: agents observe local quality signals, compute pressure gradients, and take locally-greedy actions. Coordination emerges from shared artifact state, not explicit communication.

+ We introduce *temporal decay* as a mechanism for preventing premature convergence and ensuring continued exploration of high-uncertainty regions.

+ We prove convergence guarantees for this coordination scheme under conditions that commonly hold in practice.

+ We demonstrate empirically that gradient-field coordination scales linearly with agent count on [benchmark], while hierarchical approaches plateau at [N] agents.

= Related Work

== Multi-Agent LLM Systems

Recent work has explored multi-agent architectures for LLM-based task solving. AutoGen @wu2023autogen introduces a conversation-based framework where customizable agents interact through message passing, with support for human-in-the-loop workflows. MetaGPT @hong2023metagpt encodes Standardized Operating Procedures (SOPs) into agent workflows, assigning specialized roles (architect, engineer, QA) in an assembly-line paradigm. CAMEL @li2023camel proposes role-playing between AI assistant and AI user agents, using inception prompting to guide autonomous cooperation. CrewAI @crewai2024 similarly defines agents with roles, goals, and backstories that collaborate on complex tasks.

These frameworks share a common design pattern: explicit orchestration through message passing, role assignment, and hierarchical task decomposition. While effective for structured workflows, this approach faces scaling limitations. Central coordinators become bottlenecks, message-passing overhead grows with agent count, and failures in manager agents cascade to dependents. Our work takes a fundamentally different approach: coordination emerges from shared state rather than explicit communication.

== Swarm Intelligence and Stigmergy

The concept of stigmergy—indirect coordination through environment modification—was introduced by Grassé @grasse1959stigmergie to explain termite nest-building behavior. Termites deposit pheromone-infused material that attracts further deposits, leading to emergent construction without central planning. This principle has proven remarkably powerful: complex structures arise from simple local rules without any agent having global knowledge.

Dorigo and colleagues @dorigo1996ant @dorigo1997acs formalized this insight into Ant Colony Optimization (ACO), where artificial pheromone trails guide search through solution spaces. Key mechanisms include positive feedback (reinforcing good paths), negative feedback (pheromone evaporation), and purely local decision-making. ACO has achieved strong results on combinatorial optimization problems including TSP, vehicle routing, and scheduling.

Our pressure-field coordination directly inherits from stigmergic principles. The artifact serves as the shared environment; pressure gradients are analogous to pheromone concentrations; decay corresponds to evaporation. However, we generalize beyond path-finding to arbitrary artifact refinement and provide formal convergence guarantees through the potential game framework.

== Decentralized Optimization

Potential games, introduced by Monderer and Shapley @monderer1996potential, are games where individual incentives align with a global potential function. A key property is that any sequence of unilateral improvements converges to a Nash equilibrium—greedy local play achieves global coordination. This provides the theoretical foundation for our convergence guarantees: under pressure alignment, the artifact pressure serves as a potential function.

Distributed gradient descent methods @nedic2009distributed @yuan2016convergence address optimization when data or computation is distributed across nodes. The standard approach combines local gradient steps with consensus averaging. While these methods achieve convergence rates matching centralized alternatives, they typically require communication protocols and synchronization. Our approach avoids explicit communication entirely: agents coordinate only through the shared artifact, achieving $O(1)$ coordination overhead.

The connection between multi-agent learning and game theory has been extensively studied @shoham2008multiagent. Our contribution is applying these insights to LLM-based artifact refinement, where the "game" is defined by pressure functions over quality signals rather than explicit reward structures.

= Problem Formulation

We formalize artifact refinement as a dynamical system over a pressure landscape rather than an optimization problem with a target state. The system evolves through local actions and continuous decay, settling into stable basins that represent acceptable artifact states.

== State Space

An *artifact* consists of $n$ regions with content $c_i in cal(C)$ for $i in {1, ..., n}$, where $cal(C)$ is an arbitrary content space (strings, AST nodes, etc.). Each region also carries auxiliary state $h_i in cal(H)$ representing confidence, fitness, and history.

The full system state is:
$ s = ((c_1, h_1), ..., (c_n, h_n)) in (cal(C) times cal(H))^n $

== Pressure Landscape

A *signal function* $sigma: cal(C) -> RR^d$ maps content to measurable features. Signals are *local*: $sigma(c_i)$ depends only on region $i$.

A *pressure function* $phi: RR^d -> RR_(>=0)$ maps signals to scalar "badness." We consider $k$ pressure axes with weights $bold(w) in RR^k_(>0)$. The *region pressure* is:

$ P_i(s) = sum_(j=1)^k w_j phi_j (sigma(c_i)) $

The *artifact pressure* is:

$ P(s) = sum_(i=1)^n P_i(s) $

This defines a landscape over artifact states. Low-pressure regions are "valleys" where the artifact satisfies quality constraints.

== System Dynamics

The system evolves in discrete time steps (ticks). Each tick consists of three phases:

*Phase 1: Decay.* Auxiliary state erodes toward a baseline. For fitness $f_i$ and confidence $gamma_i$ components of $h_i$:

$ f_i^(t+1) = f_i^t dot.c e^(-lambda_f) , quad gamma_i^(t+1) = gamma_i^t dot.c e^(-lambda_gamma) $

where $lambda_f, lambda_gamma > 0$ are decay rates. Decay ensures that stability requires continuous reinforcement.

*Phase 2: Action.* For each region $i$ where pressure exceeds activation threshold ($P_i > tau_"act"$) and the region is not inhibited, an *actor* $a: cal(C) times cal(H) times RR^d -> cal(C)$ proposes a content transformation. The actor observes only local state $(c_i, h_i, sigma(c_i))$.

*Phase 2b: Parallel Validation.* When multiple patches are proposed, each is validated on an independent *fork* of the artifact. Forks are created by cloning artifact state; validation (e.g., compilation, test execution) proceeds in parallel across forks. This addresses a fundamental resource constraint: a single artifact cannot be used to test multiple patches simultaneously without cloning.

*Phase 3: Reinforcement.* Regions where actions were applied receive fitness and confidence boosts, and enter an inhibition period preventing immediate re-modification:

$ f_i^(t+1) = min(f_i^t + Delta_f, 1), quad gamma_i^(t+1) = min(gamma_i^t + Delta_gamma, 1) $

== Stable Basins

#definition(name: "Stability")[
  A state $s^*$ is *stable* if, under the system dynamics with no external perturbation:
  1. All region pressures are below activation threshold: $P_i(s^*) < tau_"act"$ for all $i$
  2. Decay is balanced by residual fitness: the system remains in a neighborhood of $s^*$
]

The central questions are:
1. *Existence*: Under what conditions do stable basins exist?
2. *Quality*: What is the pressure $P(s^*)$ of states in stable basins?
3. *Convergence*: From initial state $s_0$, does the system reach a stable basin? How quickly?
4. *Decentralization*: Can stability be achieved with purely local decisions?

== The Locality Constraint

The key constraint distinguishing our setting from centralized optimization: agents observe only local state. An actor at region $i$ sees $(c_i, h_i, sigma(c_i))$ but not:
- Other regions' content $c_j$ for $j != i$
- Global pressure $P(s)$
- Other agents' actions

This rules out coordinated planning. Stability must emerge from local incentives aligned with global pressure reduction.

= Method

We now present a coordination mechanism that achieves stability through purely local decisions. The key insight is that under appropriate conditions, the artifact pressure $P(s)$ acts as a *potential function*: local improvements by individual agents decrease global pressure, guaranteeing convergence without coordination.

== Pressure Alignment

The locality constraint prohibits agents from observing global state. For decentralized coordination to succeed, we need local incentives to align with global pressure reduction.

#definition(name: "Pressure Alignment")[
  A pressure system is *aligned* if for any region $i$, state $s$, and action $a_i$ that reduces local pressure:
  $ P_i(s') < P_i(s) quad ==> quad P(s') < P(s) $
  where $s' = s[c_i |-> a_i(c_i)]$ is the state after applying $a_i$.
]

Alignment holds automatically when pressure functions are *separable*: each $P_i$ depends only on $c_i$, so $P(s) = sum_i P_i(s)$ and local improvement directly implies global improvement.

More generally, alignment holds when cross-region interactions are bounded:

#definition(name: "Bounded Coupling")[
  A pressure system has *$epsilon$-bounded coupling* if for any action $a_i$ on region $i$:
  $ abs(P_j(s') - P_j(s)) <= epsilon quad forall j != i $
  That is, modifying region $i$ changes other regions' pressures by at most $epsilon$.
]

Under $epsilon$-bounded coupling with $n$ regions, if a local action reduces $P_i$ by $delta > n epsilon$, then global pressure decreases by at least $delta - n epsilon > 0$.

== Connection to Potential Games

The aligned pressure system forms a *potential game* where:
- Players are regions (or agents acting on regions)
- Strategies are content choices $c_i in cal(C)$
- The potential function is $Phi(s) = P(s)$

In potential games, any sequence of improving moves converges to a Nash equilibrium. In our setting, Nash equilibria correspond to stable basins: states where no local action can reduce pressure below the activation threshold.

This connection provides our convergence guarantee without requiring explicit coordination.

== The Coordination Algorithm

The tick loop implements greedy local improvement with decay-driven exploration:

#algorithm(name: "Pressure-Field Tick")[
  *Input:* State $s^t$, signal functions ${sigma_j}$, pressure functions ${phi_j}$, actors ${a_k}$, parameters $(tau_"act", lambda_f, lambda_gamma, Delta_f, Delta_gamma, kappa)$

  *Phase 1: Decay*
  #h(1em) For each region $i$: $quad f_i <- f_i dot.c e^(-lambda_f), quad gamma_i <- gamma_i dot.c e^(-lambda_gamma)$

  *Phase 2: Activation and Proposal*
  #h(1em) $cal(P) <- emptyset$
  #h(1em) For each region $i$ where $P_i(s) >= tau_"act"$ and not inhibited:
  #h(2em) $bold(sigma)_i <- sigma(c_i)$
  #h(2em) For each actor $a_k$:
  #h(3em) $delta <- a_k(c_i, h_i, bold(sigma)_i)$
  #h(3em) $cal(P) <- cal(P) union {(i, delta, hat(Delta)(delta))}$

  *Phase 3: Parallel Validation and Selection*
  #h(1em) For each candidate patch $(i, delta, hat(Delta)) in cal(P)$:
  #h(2em) Fork artifact: $(f_"id", A_f) <- A."fork"()$
  #h(2em) Apply $delta$ to fork $A_f$
  #h(2em) Validate fork (run tests, check compilation)
  #h(1em) Collect validation results ${(i, delta, Delta_"actual", "valid")}$
  #h(1em) Sort validated patches by $Delta_"actual"$
  #h(1em) Greedily select top-$kappa$ non-conflicting patches

  *Phase 4: Application and Reinforcement*
  #h(1em) For each selected patch $(i, delta, dot)$:
  #h(2em) $c_i <- delta(c_i)$
  #h(2em) $f_i <- min(f_i + Delta_f, 1)$, $gamma_i <- min(gamma_i + Delta_gamma, 1)$
  #h(2em) Mark region $i$ inhibited for $tau_"inh"$ ticks

  *Return* updated state $s^(t+1)$
]

The algorithm has three key properties:

*Locality.* Each actor observes only $(c_i, h_i, sigma(c_i))$. No global state is accessed.

*Bounded parallelism.* At most $kappa$ patches per tick prevents thrashing. Inhibition prevents repeated modification of the same region.

*Decay-driven exploration.* Even stable regions eventually decay below confidence thresholds, attracting re-evaluation. This prevents premature convergence to local minima.

== Stability and Termination

The system reaches a stable basin when:
1. All region pressures satisfy $P_i(s) < tau_"act"$
2. Decay is balanced: fitness remains above the threshold needed for stability

Termination is *economic*, not logical. The system stops acting when the cost of action (measured in pressure reduction per patch) falls below the benefit. This matches natural systems: activity ceases when gradients flatten, not when an external goal is declared achieved.

In practice, we also impose budget constraints (maximum ticks or patches) to bound computation.

= Theoretical Analysis

We establish three main results: (1) convergence to stable basins under alignment, (2) bounds on stable basin quality, and (3) scaling properties relative to centralized alternatives.

== Convergence Under Alignment

#theorem(name: "Convergence")[
  Let the pressure system be aligned with $epsilon$-bounded coupling. Let $delta_"min" > 0$ be the minimum pressure reduction from any applied patch, and assume $delta_"min" > n epsilon$ where $n$ is the number of regions. Then from any initial state $s_0$ with pressure $P_0 = P(s_0)$, the system reaches a stable basin within:
  $ T <= P_0 / (delta_"min" - n epsilon) $
  ticks, provided decay rates satisfy $lambda_f, lambda_gamma < delta_"min" "/" tau_"inh"$.
]

*Proof sketch.* Under alignment with $epsilon$-bounded coupling, each applied patch reduces global pressure by at least $delta_"min" - n epsilon > 0$. Since $P(s) >= 0$ and decreases by a fixed minimum per tick (when patches are applied), the system must reach a state where no region exceeds $tau_"act"$ within the stated bound. The decay constraint ensures that stability is maintained once reached: fitness reinforcement from the final patches persists longer than the decay erodes it. $square$

The bound is loose but establishes the key property: convergence time scales with initial pressure, not with state space size or number of possible actions.

== Basin Quality

#theorem(name: "Basin Quality")[
  In any stable basin $s^*$, the artifact pressure satisfies:
  $ P(s^*) < n dot.c tau_"act" $
  where $n$ is the number of regions and $tau_"act"$ is the activation threshold.
]

*Proof.* By definition of stability, $P_i(s^*) < tau_"act"$ for all $i$. Summing over regions: $P(s^*) = sum_i P_i(s^*) < n dot.c tau_"act"$. $square$

This bound is tight: adversarial initial conditions can place the system in a basin where each region has pressure just below threshold. However, in practice, actors typically reduce pressure well below $tau_"act"$, yielding much lower basin pressures.

#theorem(name: "Basin Separation")[
  Under separable pressure (zero coupling), distinct stable basins are separated by pressure barriers of height at least $tau_"act"$.
]

*Proof sketch.* Moving from one basin to another requires some region to exceed $tau_"act"$ (otherwise no action is triggered). The minimum such exceedance defines the barrier height. $square$

This explains why decay is necessary: without decay, the system can become trapped in suboptimal basins. Decay gradually erodes fitness, eventually allowing re-evaluation and potential escape to lower-pressure basins.

== Scaling Properties

#theorem(name: "Linear Scaling")[
  Let $m$ be the number of regions and $n$ be the number of parallel agents. The per-tick complexity is:
  - *Signal computation:* $O(m dot.c d)$ where $d$ is signal dimension
  - *Pressure computation:* $O(m dot.c k)$ where $k$ is the number of pressure axes
  - *Patch proposal:* $O(m dot.c a)$ where $a$ is the number of actors
  - *Selection:* $O(m dot.c a dot.c log(m dot.c a))$ for sorting candidates
  - *Coordination overhead:* $O(1)$ — no inter-agent communication

  Total: $O(m dot.c (d + k + a dot.c log(m a)))$, independent of $n$.
]

The key observation: adding agents increases throughput (more patches proposed per tick) without increasing coordination cost. This contrasts with hierarchical schemes where coordination overhead grows with agent count.

#theorem(name: "Parallel Convergence")[
  Under the same alignment conditions as Theorem 1, with $K$ patches validated in parallel per tick where patches affect disjoint regions, the system reaches a stable basin within:
  $ T <= P_0 / (K dot.c (delta_"min" - n epsilon)) $
  This improves convergence time by factor $K$ while maintaining guarantees.
]

*Proof sketch.* When $K$ non-conflicting patches are applied per tick, each reduces global pressure by at least $delta_"min" - n epsilon$. The combined reduction is $K dot.c (delta_"min" - n epsilon)$ per tick. The bound follows directly. Note that if patches conflict (target the same region), only one is selected per region, and effective speedup is reduced. $square$

== Comparison to Alternatives

We compare against three coordination paradigms:

*Centralized planning.* A global planner evaluates all $(m dot.c a)$ possible actions, selects optimal subset. Per-step complexity: $O(m dot.c a)$ evaluations, but requires global state access. Sequential bottleneck prevents parallelization.

*Hierarchical delegation.* Manager agents decompose tasks, delegate to workers. Communication complexity: $O(n log n)$ for tree-structured delegation with $n$ agents. Latency scales with tree depth. Failure of manager blocks all descendants.

*Message-passing coordination.* Agents negotiate actions through pairwise communication. Convergence requires $O(n^2)$ messages in worst case for $n$ agents. Consensus protocols add latency.

#figure(
  table(
    columns: 4,
    [*Paradigm*], [*Coordination*], [*Parallelism*], [*Fault tolerance*],
    [Centralized], [$O(m dot.c a)$], [None], [Single point of failure],
    [Hierarchical], [$O(n log n)$], [Limited by tree], [Manager failure cascades],
    [Message-passing], [$O(n^2)$], [Consensus-bound], [Partition-sensitive],
    [Pressure-field], [$O(1)$], [Full ($min(n, m, K)$)], [Graceful degradation],
  ),
  caption: [Coordination overhead comparison. $K$ denotes the fork pool size for parallel validation.],
)

Pressure-field coordination achieves $O(1)$ coordination overhead because agents share state only through the artifact itself—a form of stigmergy. Agents can fail, join, or leave without protocol overhead.

= Experiments

== Setup

// TODO: Define benchmarks, baselines, metrics

=== Benchmarks

// TODO: Specific tasks

=== Baselines

// TODO: AutoGen, single-agent iterative, simple hierarchical

=== Metrics

// TODO: Quality, patches applied, wall-clock time, scaling curves

== Main Results

// TODO: Tables and figures

== Ablations

=== Effect of Decay

// TODO: With vs without decay

=== Effect of Inhibition

// TODO: With vs without inhibition window

=== Pressure Weight Sensitivity

// TODO: Vary weights

== Scaling Experiments

// TODO: 1, 2, 4, 8, 16 agents

= Discussion

== Limitations

- Requires well-designed pressure functions (not learned)
- Decay parameters require tuning
- May not suit tasks requiring global planning
- Goodhart's Law: agents may game metrics
- Resource cost of parallel validation: testing $K$ patches in parallel requires $K$ artifact forks, with memory cost $O(K dot.c |A|)$ where $|A|$ is artifact size. For large artifacts, this may require fork pooling or lazy cloning strategies.

== When Hierarchical Coordination Is Appropriate

// TODO: Acknowledge cases where planning matters

== Future Work

- Learning pressure functions from data
- Adversarial robustness
- Extension to multi-artifact coordination

= Conclusion

We presented gradient-field coordination, a decentralized approach to multi-agent systems that achieves coordination through shared state and local pressure gradients rather than explicit orchestration. Our theoretical analysis shows convergence guarantees under alignment conditions, and our experiments demonstrate linear scaling with agent count. This work suggests that constraint-driven emergence, inspired by natural coordination mechanisms, offers a more scalable foundation for multi-agent AI than imported human organizational patterns.

= Appendix: Experimental Protocol

This appendix provides complete reproducibility information for all experiments.

== Hardware and Software

*Hardware:* NVIDIA RTX 4090 GPU, 24GB VRAM

*Software:*
- Rust 1.75+ (edition 2024)
- Ollama v0.5+
- Models: `qwen2.5-coder:1.5b`, `qwen2.5-coder:7b`, `qwen2.5-coder:14b`

== Model Configuration

Custom modelfiles tune the base models for Latin Square solving:

```
FROM qwen2.5-coder:1.5b
SYSTEM """You solve Latin Square puzzles. Given a row with ONE empty cell (_),
return ONLY the single number that fills it.
Example: Row "1 _ 3 4" with available values [2] → Output: 2
Return just the number, nothing else."""
PARAMETER num_predict 8
```

Create variants for 7b and 14b by changing the FROM line.

== Sampling Diversity

The experiment framework overrides default sampling parameters with three exploration bands per LLM call:

#figure(
  table(
    columns: 3,
    [*Band*], [*Temperature*], [*Top-p*],
    [Exploitation], [0.15 - 0.35], [0.80 - 0.90],
    [Balanced], [0.35 - 0.55], [0.85 - 0.95],
    [Exploration], [0.55 - 0.85], [0.90 - 0.98],
  ),
  caption: [Sampling parameter ranges. Each LLM call randomly samples from one band.],
)

This diversity prevents convergence to local optima and enables exploration of the solution space.

== Experiment Commands

*Main Grid (Strategy Comparison):*
```bash
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  --escalation-threshold 10 \
  grid --trials 10 --n 7 --empty 7 --max-ticks 40 --agents 1,2,4,8
```

*Ablation Study:*
```bash
latin-experiment --model-chain "latin-solver" \
  ablation --trials 10 --n 7 --empty 7 --max-ticks 40
```

*Scaling Analysis:*
```bash
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  grid --trials 10 --n 7 --empty 8 --max-ticks 40 --agents 1,2,4,8,16,32
```

*Model Escalation Comparison:*
```bash
# Without escalation
latin-experiment --model-chain "latin-solver" \
  grid --trials 10 --n 7 --empty 8 --max-ticks 40 --agents 2,4,8

# With escalation
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  --escalation-threshold 10 \
  grid --trials 10 --n 7 --empty 8 --max-ticks 40 --agents 2,4,8
```

*Difficulty Scaling:*
```bash
# Run for each (n, empty) pair: (5,5), (6,8), (7,10), (8,14)
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  grid --trials 10 --n {N} --empty {E} --max-ticks 50 --agents 4
```

== Metrics Collected

Each experiment records:
- `solved`: Boolean indicating puzzle completion
- `total_ticks`: Iterations to solve (or max if unsolved)
- `pressure_history`: Pressure value at each tick
- `escalation_events`: Model tier changes (tick, from_model, to_model)
- `final_model`: Which model tier solved the puzzle

== Statistical Analysis

- 10 trials per configuration for statistical significance
- Mann-Whitney U test for pairwise strategy comparisons
- 95% confidence intervals via bootstrap resampling
- Effect sizes reported as Cohen's d

== Estimated Runtime

#figure(
  table(
    columns: 4,
    [*Experiment*], [*Configurations*], [*Trials*], [*Est. Time*],
    [Main Grid], [16], [10], [45 min],
    [Ablation], [8], [10], [20 min],
    [Scaling], [6], [10], [30 min],
    [Escalation], [6], [10], [30 min],
    [Difficulty], [4], [10], [40 min],
    [*Total*], [], [], [*~3 hours*],
  ),
  caption: [Estimated runtime for all experiments on NVIDIA RTX 4090.],
)

#bibliography("references.bib", style: "ieee")
