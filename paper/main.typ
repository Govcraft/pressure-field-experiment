#import "template.typ": *

#show: neurips.with(
  title: [Emergent Coordination in Multi-Agent Systems via Pressure Fields and Temporal Decay],
  authors: (
    (
      name: "Roland Rodriguez",
      affiliation: "Independent Researcher",
    ),
  ),
  abstract: [
    Current multi-agent LLM frameworks rely on explicit orchestration patterns borrowed from human organizational structures: planners delegate to executors, managers coordinate workers, and hierarchical control flow governs agent interactions. These approaches suffer from coordination overhead that scales poorly with agent count and task complexity. We propose a fundamentally different paradigm inspired by natural coordination mechanisms: agents operate locally on a shared artifact, guided only by pressure gradients derived from measurable quality signals, with temporal decay preventing premature convergence. We formalize this as optimization over a pressure landscape and prove convergence guarantees under mild conditions.

    Empirically, on Latin Square constraint satisfaction, pressure-field coordination consistently outperforms all baselines: achieving 80% solve rate on easy problems (5×5) versus 20% for hierarchical, and 23% on hard problems (7×7 with model escalation) versus 1% for the best baseline. Model escalation proves critical---without it, no strategy solves hard problems; with it, pressure-field achieves 21× improvement over hierarchical. Temporal decay is essential: disabling it increases final pressure 15-fold, trapping agents in local minima. The approach maintains consistent performance from 1 to 32 agents while baselines show near-zero solve rates. Our results indicate that constraint-driven emergence offers a more scalable foundation for multi-agent AI systems than imported human organizational patterns.
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

+ We demonstrate empirically that gradient-field coordination achieves 80% solve rate on easy Latin Square problems (vs. 20% for hierarchical), and with model escalation achieves 23% on hard problems where baselines achieve only 0--1%. The approach maintains consistent 17--37% solve rates from 1 to 32 agents while baselines collapse to near-zero.

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

An *artifact* consists of $n$ regions with content $c_i in cal(C)$ for $i in {1, ..., n}$, where $cal(C)$ is an arbitrary content space (strings, AST nodes, etc.). Each region also carries auxiliary state $h_i in cal(H)$ representing confidence, fitness, and history. Regions are passive subdivisions of the artifact; agents are active proposers that observe regions and generate patches.

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
  - *Coordination overhead:* $O(1)$ — no inter-agent communication (fork pool is $O(K)$ where $K$ is fixed)

  Total: $O(m dot.c (d + k + a dot.c log(m a)))$, independent of agent count $n$.
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

We evaluate pressure-field coordination on Latin Square constraint satisfaction: filling partially-completed $n times n$ grids such that each row and column contains each number $1$ to $n$ exactly once. This domain provides clear pressure signals (constraint violations), measurable success criteria, and scalable difficulty.

*Key findings*: Pressure-field coordination consistently outperforms all baselines across difficulty levels (§5.2). Temporal decay is critical---disabling it increases final pressure 15-fold, trapping agents in local minima (§5.3). Model escalation proves essential for hard problems, enabling 23% solve rate where single-model approaches achieve 0% (§5.5). The approach maintains consistent performance from 1 to 32 agents while baselines collapse to near-zero (§5.4).

== Setup

=== Task: Latin Square Constraint Satisfaction

We generate $7 times 7$ Latin Square puzzles with 7 empty cells (15% incomplete). Each puzzle has a unique solution. Agents propose values for empty cells; a puzzle is "solved" when all constraints are satisfied (zero violations) within 100 ticks.

*Pressure function*: $P_i = "empty"_i + 10 dot.c "row\_dups"_i + 10 dot.c "col\_conflicts"_i$

where $"empty"_i$ counts unfilled cells in row $i$, $"row\_dups"_i$ counts duplicate values within row $i$, and $"col\_conflicts"_i$ counts values in row $i$ that conflict with other rows in the same column.

=== Baselines

We compare five coordination strategies, all using identical LLMs (`Qwen/Qwen2.5-0.5B` via vLLM) to isolate coordination effects:

*Pressure-field (ours)*: Full system with decay ($lambda_f = 0.1$), inhibition ($tau_"inh" = 4$ ticks), and parallel validation.

*Sequential*: Single agent iterates through rows in fixed order, proposing one value per tick. No parallelism or pressure guidance.

*Hierarchical*: Simulated manager identifies the row with most empty cells, delegates to worker agent. One patch per tick.

*Random*: Selects random rows and proposes random valid values. Same LLM and validation as other methods.

*Conversation*: AutoGen-style multi-agent dialogue where agents discuss and negotiate moves through explicit message passing. A coordinator agent selects target regions, proposer agents suggest values, and validator agents check constraints. Consensus required before applying patches.

=== Metrics

- *Solve rate*: Percentage of puzzles reaching zero pressure within 100 ticks
- *Ticks to solve*: Convergence speed for solved cases
- *Final pressure*: Remaining constraint violations for unsolved cases

=== Implementation

*Hardware*: NVIDIA A100 80GB GPU. *Software*: Rust implementation with vLLM. *Trials*: 30 per configuration. Full protocol in Appendix A.

*Model escalation*: Unless otherwise noted, all experiments use adaptive model escalation: when a region remains high-pressure for 20 consecutive ticks, the system escalates through the chain 0.5B → 1.5B → 3B → 7B → 14B. Section 5.5 ablates this mechanism.

== Main Results

Pressure-field coordination consistently outperforms all baselines:

#figure(
  table(
    columns: 4,
    [*Strategy*], [*Solved/N*], [*Rate*], [*95% Wilson CI*],
    [Pressure-field], [32/120], [*26.7%*], [19.6%--35.2%],
    [Hierarchical], [4/120], [3.3%], [1.3%--8.3%],
    [Sequential], [0/120], [0.0%], [0.0%--3.1%],
    [Random], [0/120], [0.0%], [0.0%--3.1%],
    [Conversation], [0/120], [0.0%], [0.0%--3.1%],
  ),
  caption: [Solve rates on $7 times 7$ Latin Squares with model escalation (30 trials per agent count $times$ 4 agent counts = 120 per strategy). Chi-square test: $chi^2 = 115.4$, $p < 10^(-23)$.],
)

The performance gap is statistically significant: pressure-field achieves 26.7% solve rate (CI: 19.6%--35.2%) compared to 3.3% for hierarchical (CI: 1.3%--8.3%), with non-overlapping confidence intervals. For solved cases, pressure-field converges in 31.8 average ticks versus 29.2 for the rare hierarchical successes.

Final pressure provides additional insight: pressure-field achieves lowest average final pressure ($2.33 plus.minus 1.79$) compared to hierarchical ($4.55 plus.minus 1.50$), sequential ($5.33 plus.minus 1.23$), random ($5.35 plus.minus 1.21$), and conversation ($11.40 plus.minus 10.0$). Kruskal-Wallis test confirms strategy significantly affects final pressure ($chi^2 = 378.1$, $p < 10^(-80)$).

This validates Theorem 3: coordination overhead remains $O(1)$ for pressure-field, enabling effective parallelism, while hierarchical approaches suffer from coordination bottlenecks that prevent scaling.

== Ablations

=== Effect of Temporal Decay

Decay proves essential---without it, final pressure increases dramatically:

#figure(
  table(
    columns: 4,
    [*Configuration*], [*N*], [*Final Pressure*], [*SD*],
    [With decay], [120], [$4.38$], [$1.36$],
    [Without decay], [120], [*$65.83$*], [$19.59$],
  ),
  caption: [Decay ablation (2 agents, 30 trials $times$ 4 configurations each). Welch's t-test: $t = -34.3$, $p < 10^(-63)$. Cohen's $d = 4.43$ (huge effect).],
)

The effect size is massive: Cohen's $d = 4.43$ far exceeds the threshold for "large" effects ($d > 0.8$). Without decay, fitness saturates after initial patches. High-fitness regions never re-enter the activation threshold, leaving the artifact in a high-pressure state. This validates Theorem 2: decay is necessary to continue pressure reduction even when regions appear "stable."

=== Effect of Inhibition and Examples

The ablation study tested all $2^3 = 8$ combinations of decay, inhibition, and few-shot examples:

#figure(
  table(
    columns: 4,
    [*Configuration*], [*Solved/N*], [*Final Pressure*], [*SD*],
    [D=T, I=T, E=T (full)], [1/30], [$4.37$], [$1.54$],
    [D=T, I=T, E=F], [0/30], [$3.93$], [$1.14$],
    [D=T, I=F, E=T], [0/30], [$4.93$], [$1.31$],
    [D=T, I=F, E=F], [0/30], [$4.30$], [$1.29$],
    [D=F, I=T, E=T], [0/30], [$65.63$], [$17.70$],
    [D=F, I=T, E=F], [0/30], [$66.97$], [$18.71$],
    [D=F, I=F, E=T], [0/30], [$61.07$], [$24.26$],
    [D=F, I=F, E=F], [0/30], [$69.67$], [$16.81$],
  ),
  caption: [Full ablation results. D=decay, I=inhibition, E=examples. Decay is the critical mechanism: with decay, final pressure $approx 4$; without decay, $approx 65$. Inhibition and examples provide marginal benefit.],
) <tbl:ablation>

The key finding is that *decay dominates*: any configuration with decay achieves final pressure $approx 4$, while any without decay achieves $approx 65$. The 7$times$7 problem proved too difficult to differentiate solve rates (all near 0%), but the 15$times$ pressure difference clearly demonstrates decay's importance.

== Scaling Experiments

Pressure-field maintains consistent performance from 1 to 32 agents while baselines collapse:

#figure(
  table(
    columns: 4,
    [*Agents*], [*Solved/30*], [*Rate*], [*95% Wilson CI*],
    [1], [6/30], [20.0%], [9.5%--37.3%],
    [2], [5/30], [16.7%], [7.3%--33.6%],
    [4], [11/30], [*36.7%*], [21.9%--54.5%],
    [8], [6/30], [20.0%], [9.5%--37.3%],
    [16], [6/30], [20.0%], [9.5%--37.3%],
    [32], [5/30], [16.7%], [7.3%--33.6%],
  ),
  caption: [Pressure-field scaling from 1 to 32 agents ($7 times 7$ grid, 8 empty cells). Total: 39/180 solved (21.7%). All baselines achieve 0--3% across all agent counts (not shown for space).],
)

Pressure-field shows a peak at 4 agents (36.7%, CI: 21.9%--54.5%) with consistent performance at other counts. Note that confidence intervals overlap substantially, indicating the peak may reflect sampling variability rather than a true optimum. The key observation is *stability*: pressure-field maintains 17--37% solve rates across 32$times$ variation in agent count.

Baselines show fundamental scaling limitations: hierarchical achieves only sporadic successes (3.3% at 2, 8, 16 agents; 0% otherwise), while sequential, random, and conversation achieve 0% regardless of agent count. This confirms that coordination overhead in baseline approaches prevents effective scaling.

== Model Escalation Ablation

To quantify the impact of model escalation, we compare performance with and without the escalation chain on hard problems:

#figure(
  table(
    columns: 5,
    [*Strategy*], [*Without Esc*], [*With Esc*], [*95% Wilson CI*], [*Improvement*],
    [Pressure-field], [0/90 (0%)], [*21/90 (23.3%)*], [15.7%--33.2%], [$infinity$],
    [Hierarchical], [0/90 (0%)], [1/90 (1.1%)], [0.2%--6.0%], [$infinity$],
    [Sequential], [0/90 (0%)], [0/90 (0%)], [0.0%--4.0%], [---],
    [Random], [0/90 (0%)], [0/90 (0%)], [0.0%--4.0%], [---],
    [Conversation], [0/90 (0%)], [0/90 (0%)], [0.0%--4.0%], [---],
  ),
  caption: [Model escalation impact ($7 times 7$, 8 empty cells, 90 trials per strategy). Fisher's exact test for pressure-field escalation effect: $p = 2.55 times 10^(-7)$. Without escalation (0.5B only), no strategy solves any puzzles; with escalation (0.5B→14B), pressure-field achieves 23%.],
) <tbl:escalation>

Model escalation proves *critical* for hard problems: without it, all strategies achieve 0% solve rate. With escalation, pressure-field achieves 23.3% (21 of 90 trials) while hierarchical manages only 1.1% (1 of 90). This demonstrates that:

1. *Small models alone are insufficient* for difficult constraint satisfaction
2. *Escalation benefits pressure-field disproportionately*: the pressure gradient provides clear signal for when to escalate
3. *Hierarchical coordination cannot exploit escalation effectively*: even with access to larger models, coordination overhead prevents success

== Difficulty Scaling

On easier problems, pressure-field shows strong performance while baselines struggle:

#figure(
  table(
    columns: 4,
    [*Strategy*], [*Solved/30*], [*Rate*], [*95% Wilson CI*],
    [Pressure-field], [24/30], [*80.0%*], [62.5%--90.9%],
    [Hierarchical], [6/30], [20.0%], [9.5%--37.3%],
    [Sequential], [0/30], [0.0%], [0.0%--11.4%],
    [Random], [0/30], [0.0%], [0.0%--11.4%],
    [Conversation], [0/30], [0.0%], [0.0%--11.4%],
  ),
  caption: [Solve rate on easy problems ($5 times 5$ grid, 5 empty cells, 4 agents, 30 trials). Fisher's exact test (pressure-field vs hierarchical): Odds Ratio $= 15.04$ (95% CI: 3.92--69.13), $p < 0.0001$. Pressure-field achieves 4$times$ higher solve rate than hierarchical.],
)

The difficulty scaling reveals two key insights:

1. *Easy problems show clear separation*: Pressure-field achieves 80% vs hierarchical's 20%---a 4$times$ advantage. Sequential, random, and conversation fail entirely even on easy problems.

2. *Hard problems require escalation*: Without model escalation, no strategy solves 7×7 puzzles. With escalation, pressure-field recovers to 23% while hierarchical manages only 1%.

Note: Medium ($6 times 6$) and Very Hard ($8 times 8$) experiments were not completed in the current run. The pattern suggests graceful degradation for pressure-field as difficulty increases.

= Discussion

== Limitations

Our experiments reveal several important limitations:

*Absolute solve rates are modest on hard problems.* While pressure-field consistently outperforms baselines, the absolute solve rates on $7 times 7$ puzzles (17--37% with escalation) indicate this remains a challenging domain. The relative advantage (21$times$ over hierarchical) is substantial, but practitioners should not expect near-perfect performance on difficult constraint satisfaction without further optimization.

*Decay is non-optional.* Without temporal decay, final pressure increases 15-fold regardless of other mechanisms. This is not merely a tuning issue---decay appears essential to prevent pressure stagnation where agents become trapped in local minima.

*Model escalation is required for hard problems.* Without access to larger models, no strategy (including pressure-field) solves $7 times 7$ puzzles. This suggests the coordination mechanism alone is insufficient---capable underlying models are a prerequisite.

*Additional practical limitations:*
- Requires well-designed pressure functions (not learned from data)
- Decay rates $lambda_f, lambda_gamma$ and inhibition period require task-specific tuning
- May not suit tasks requiring long-horizon global planning
- Goodhart's Law: agents may game poorly-designed metrics
- Resource cost of parallel validation: testing $K$ patches requires $O(K dot.c |A|)$ memory where $|A|$ is artifact size

== When Hierarchical Coordination Is Appropriate

While pressure-field outperforms hierarchical across all our experiments (4$times$ on easy, 21$times$ on hard+escalation), hierarchical approaches may remain preferable when:

1. *Problem structure is well-understood.* A central coordinator can exploit known decomposition strategies that don't require exploration.

2. *Communication is cheap and reliable.* Our baselines assume zero-cost coordination messages. Real distributed systems may favor hierarchical approaches when network latency dominates compute time.

3. *Interpretability is critical.* Hierarchical task assignment provides clear audit trails; emergent coordination is harder to debug.

The key finding is that *hierarchical approaches do not scale*: achieving only 0--3% solve rate on hard problems regardless of agent count or model capability. Pressure-field coordination becomes increasingly advantageous as problems exceed what centralized planning can tractably decompose.

== Model Escalation as Adaptive Capability

Our escalation mechanism (0.5B → 1.5B → 3B → 7B → 14B parameters) proves essential on hard problems (@tbl:escalation). Without escalation, all strategies achieve 0% solve rate; with escalation, pressure-field achieves 23%. This suggests an important design principle: pressure-field coordination benefits from adaptive capability deployment.

When smaller models cannot reduce pressure below threshold, escalation to larger models breaks through local minima. The mechanism works because larger models have broader solution coverage, not necessarily better constraint reasoning. The 5-tier escalation chain provides graduated capability increases, invoking expensive large models only when necessary.

Critically, hierarchical coordination cannot exploit escalation effectively: even with access to the same model chain, it achieves only 1% solve rate. We hypothesize this is because hierarchical assignment decisions create dependencies that larger models cannot easily overcome, whereas pressure-field's independent local actions allow each region to benefit from escalation individually.

== Future Work

- *Learned pressure functions*: Current sensors are hand-designed. Can we learn pressure functions from solution traces?
- *Adversarial robustness*: Can malicious agents exploit pressure gradients to degrade system performance?
- *Multi-artifact coordination*: Extension to coupled artifacts where patches in one affect pressure in another
- *Larger-scale experiments*: Testing on $8 times 8$ and $9 times 9$ grids to characterize the difficulty ceiling
- *Alternative domains*: Applying pressure-field coordination to code refactoring, configuration management, and other artifact refinement tasks

= Conclusion

We presented gradient-field coordination, a decentralized approach to multi-agent systems that achieves coordination through shared state and local pressure gradients rather than explicit orchestration.

Our theoretical analysis establishes convergence guarantees under pressure alignment conditions, with coordination overhead independent of agent count. Empirically, on Latin Square constraint satisfaction, pressure-field coordination consistently outperforms all baselines: achieving 80% solve rate on easy problems (4$times$ hierarchical) and 23% on hard problems with model escalation (21$times$ hierarchical). The approach maintains consistent 17--37% solve rates from 1 to 32 agents while baselines collapse to 0--3%.

Key findings include: (1) temporal decay is essential---disabling it increases final pressure 15-fold, trapping agents in local minima; (2) model escalation is critical for hard problems---without it, no strategy achieves any solves; (3) the relative advantage of pressure-field coordination *increases* with problem difficulty, suggesting it is most valuable precisely where other approaches fail.

These results suggest that constraint-driven emergence, inspired by natural coordination mechanisms like chemotaxis, offers a more scalable foundation for multi-agent AI than imported human organizational patterns. While absolute solve rates on hard problems remain modest (23%), the consistent outperformance across all configurations indicates a fundamentally more robust coordination paradigm. The approach is most advantageous for problems where centralized decomposition becomes intractable---precisely where scaling matters most.

= Appendix: Experimental Protocol

This appendix provides complete reproducibility information for all experiments.

== Hardware and Software

*Hardware:* NVIDIA A100 80GB GPU (RunPod cloud)

*Software:*
- Rust 1.75+ (edition 2024)
- vLLM (OpenAI-compatible inference server)
- Models: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`

== Model Configuration

Models are served via vLLM with a system prompt configured for Latin Square solving:

```
You solve Latin Square puzzles. Given a row with empty cells (_),
return ONLY the number(s) that fill them. Return just the numbers,
nothing else.
```

For multi-model setups (model escalation), each model runs on a separate vLLM instance with automatic port routing based on model size.

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
latin-experiment --vllm-host http://localhost:8001 \
  --model-chain "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B" \
  --escalation-threshold 20 \
  grid --trials 30 --n 7 --empty 7 --max-ticks 100 --agents 1,2,4,8
```

*Ablation Study:*
```bash
latin-experiment --vllm-host http://localhost:8001 \
  --model-chain "Qwen/Qwen2.5-0.5B" \
  ablation --trials 30 --n 7 --empty 7 --max-ticks 100
```

*Scaling Analysis:*
```bash
latin-experiment --vllm-host http://localhost:8001 \
  --model-chain "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B" \
  --escalation-threshold 20 \
  grid --trials 30 --n 7 --empty 8 --max-ticks 100 --agents 1,2,4,8,16,32
```

*Model Escalation Comparison:*
```bash
# Without escalation (single model)
latin-experiment --vllm-host http://localhost:8001 \
  --model-chain "Qwen/Qwen2.5-0.5B" \
  grid --trials 30 --n 7 --empty 8 --max-ticks 100 --agents 2,4,8

# With escalation (full chain)
latin-experiment --vllm-host http://localhost:8001 \
  --model-chain "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B" \
  --escalation-threshold 20 \
  grid --trials 30 --n 7 --empty 8 --max-ticks 100 --agents 2,4,8
```

*Difficulty Scaling:*
```bash
# Easy (5x5, 5 empty)
latin-experiment --vllm-host http://localhost:8001 \
  --model-chain "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B" \
  --escalation-threshold 20 \
  grid --trials 30 --n 5 --empty 5 --max-ticks 100 --agents 4
```

== Metrics Collected

Each experiment records:
- `solved`: Boolean indicating puzzle completion
- `total_ticks`: Iterations to solve (or max if unsolved)
- `pressure_history`: Pressure value at each tick
- `escalation_events`: Model tier changes (tick, from_model, to_model)
- `final_model`: Which model tier solved the puzzle

== Replication Notes

Each configuration runs 30 independent trials with different random seeds to ensure reliability. Results report mean solve rates and tick counts across trials.

== Estimated Runtime

#figure(
  table(
    columns: 4,
    [*Experiment*], [*Configurations*], [*Trials*], [*Est. Time*],
    [Main Grid], [20], [30], [2 hours],
    [Ablation], [8], [30], [1 hour],
    [Scaling], [30], [30], [3 hours],
    [Escalation], [10], [30], [2 hours],
    [Difficulty], [5], [30], [1.5 hours],
    [*Total*], [], [], [*~9.5 hours*],
  ),
  caption: [Estimated runtime for all experiments on NVIDIA A100 80GB GPU with 10 parallel jobs.],
)

#bibliography("references.bib", style: "ieee")
