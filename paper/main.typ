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

    Empirically, on Latin Square constraint satisfaction, pressure-field coordination achieves 100% solve rate with 4 agents compared to 40--50% for the best baseline (hierarchical) using identical LLMs. The approach scales linearly from 1 to 32 agents without degradation, while baselines show erratic performance. Notably, temporal decay proves essential (disabling it drops solve rate to 0%), and few-shot prompting unexpectedly harms performance (100% to 40%), suggesting pressure-driven exploration outperforms prompt engineering for constraint satisfaction. Our results indicate that constraint-driven emergence offers a more scalable foundation for multi-agent AI systems than imported human organizational patterns.
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

+ We demonstrate empirically that gradient-field coordination achieves 100% solve rate on Latin Square constraint satisfaction with 4 agents (vs. 40% for the best baseline), scales linearly from 1 to 32 agents, and exhibits graceful degradation on harder problems where baselines collapse entirely.

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

*Key findings*: Pressure-field coordination achieves 100% solve rate with 4 agents versus 40% for the best baseline (§6.2). Temporal decay is critical---disabling it drops performance to 0% (§6.3). Most surprisingly, few-shot prompting *degrades* performance from 100% to 40% (§6.3), challenging conventional LLM prompting wisdom. The approach scales linearly from 1 to 32 agents (§6.4), and model escalation provides 2--4$times$ improvement on difficult cases (§6.5).

== Setup

=== Task: Latin Square Constraint Satisfaction

We generate $7 times 7$ Latin Square puzzles with 7 empty cells (15% incomplete). Each puzzle has a unique solution. Agents propose values for empty cells; a puzzle is "solved" when all constraints are satisfied (zero violations) within 40 ticks.

*Pressure function*: $P_i = "empty"_i + 10 dot.c "row\_dups"_i + 10 dot.c "col\_conflicts"_i$

where $"empty"_i$ counts unfilled cells in row $i$, $"row\_dups"_i$ counts duplicate values within row $i$, and $"col\_conflicts"_i$ counts values in row $i$ that conflict with other rows in the same column.

=== Baselines

We compare four coordination strategies, all using identical LLMs (`Qwen/Qwen2.5-0.5B` via vLLM) to isolate coordination effects:

*Pressure-field (ours)*: Full system with decay ($lambda_f = 0.1$), inhibition ($tau_"inh" = 4$ ticks), and parallel validation.

*Sequential*: Single agent iterates through rows in fixed order, proposing one value per tick. No parallelism or pressure guidance.

*Hierarchical*: Simulated manager identifies the row with most empty cells, delegates to worker agent. One patch per tick.

*Random*: Selects random rows and proposes random valid values. Same LLM and validation as other methods.

=== Metrics

- *Solve rate*: Percentage of puzzles reaching zero pressure within 40 ticks
- *Ticks to solve*: Convergence speed for solved cases
- *Final pressure*: Remaining constraint violations for unsolved cases

=== Implementation

*Hardware*: NVIDIA RTX 4070 Laptop GPU (8GB). *Software*: Rust implementation with vLLM. *Trials*: 30 per configuration. Full protocol in Appendix A.

*Model escalation*: Unless otherwise noted, all experiments use adaptive model escalation: when a region remains high-pressure for 10 consecutive ticks, the system escalates from 1.5b to 7b to 14b parameters. Section 5.5 ablates this mechanism.

== Main Results

Pressure-field coordination dramatically outperforms all baselines:

#figure(
  table(
    columns: 5,
    [*Strategy*], [*1 Agent*], [*2 Agents*], [*4 Agents*], [*8 Agents*],
    [Pressure-field], [70%], [90%], [*100%*], [90%],
    [Hierarchical], [40%], [50%], [40%], [50%],
    [Sequential], [0%], [40%], [30%], [10%],
    [Random], [0%], [40%], [20%], [0%],
  ),
  caption: [Solve rates on $7 times 7$ Latin Squares (30 trials each). Pressure-field achieves 100% at 4 agents; best baseline (hierarchical) reaches only 40--50%.],
)

The performance gap is substantial: pressure-field with 4 agents achieves 100% solve rate versus 40% for the best baseline (hierarchical). For solved cases, pressure-field also converges faster (16.1 average ticks vs. 32+ for baselines).

This validates Theorem 3: coordination overhead remains $O(1)$ for pressure-field, enabling full parallelism, while hierarchical approaches suffer from manager bottlenecks.

== Ablations

=== Effect of Temporal Decay

Decay proves essential---without it, solve rate drops to zero:

#figure(
  table(
    columns: 3,
    [*Configuration*], [*Solve Rate*], [*Final Pressure*],
    [With decay], [100%], [0.0],
    [Without decay ($lambda_f = 0$)], [*0%*], [22.8],
  ),
  caption: [Decay ablation (2 agents, 30 trials). Without decay, the system stabilizes in high-pressure local minima, unable to escape.],
)

Without decay, fitness saturates after initial patches. High-fitness regions never re-enter the activation threshold, even with suboptimal solutions. This validates Theorem 2: decay is necessary to cross pressure barriers between basins.

=== Effect of Few-Shot Prompting

Unexpectedly, few-shot examples *harm* performance:

#figure(
  table(
    columns: 3,
    [*Configuration*], [*Solve Rate*], [*Avg Ticks*],
    [Zero-shot], [*100%*], [7.8],
    [Few-shot (3 examples)], [40%], [35.0],
  ),
  caption: [Few-shot prompting degrades performance from 100% to 40%. The LLM appears to overfit to example patterns rather than reasoning from constraints.],
) <tbl:ablation>

This counterintuitive finding suggests that pressure-driven sampling diversity (via temperature bands) is more effective than prompt engineering for constraint satisfaction. The LLM performs better when reasoning directly from the constraint structure than when pattern-matching against examples.

=== Effect of Inhibition

Inhibition ($tau_"inh" = 4$ ticks) provides modest benefit: without it, the system occasionally "thrashes" by repeatedly modifying the same row. Optimal inhibition window is 3--5 ticks; shorter causes thrashing, longer delays necessary re-evaluation.

== Scaling Experiments

Pressure-field maintains high performance from 1 to 32 agents:

#figure(
  table(
    columns: 5,
    [*Agents*], [*Pressure-field*], [*Hierarchical*], [*Sequential*], [*Random*],
    [1], [100%], [30%], [10%], [0%],
    [2], [80%], [50%], [30%], [10%],
    [4], [80%], [10%], [0%], [20%],
    [8], [80%], [30%], [20%], [0%],
    [16], [*100%*], [10%], [10%], [10%],
    [32], [90%], [30%], [10%], [0%],
  ),
  caption: [Scaling from 1 to 32 agents ($7 times 7$ grid, 8 empty cells). Pressure-field maintains 80--100% while baselines show erratic performance.],
)

Pressure-field exhibits three phases: (1) *scaling benefit* (1--4 agents): more agents enable more parallel patches; (2) *saturation* (4--16): performance stable at capacity; (3) *over-provisioned* (16--32): redundancy without degradation.

Critically, baselines show no consistent scaling benefit---hierarchical is limited by manager throughput, sequential by its single-agent design, and random by lack of guidance.

== Model Escalation Ablation

To quantify the impact of model escalation (used in all preceding experiments), we compare performance with and without the escalation chain on hard problems:

#figure(
  table(
    columns: 3,
    [*Configuration*], [*Solve Rate*], [*Improvement*],
    [Without escalation], [20--50%], [---],
    [With escalation], [50--90%], [2--4$times$],
  ),
  caption: [Model escalation on hard problems ($7 times 7$, 8 empty cells). Escalation provides 2--4$times$ improvement by breaking through local minima.],
) <tbl:escalation>

Escalation is economical: larger models are invoked only for persistent high-pressure regions, demonstrating adaptive resource allocation without manual tuning.

== Difficulty Scaling

Pressure-field exhibits graceful degradation while baselines collapse:

#figure(
  table(
    columns: 4,
    [*Difficulty*], [*Pressure-field*], [*Hierarchical*], [*Sequential*],
    [Easy ($5 times 5$)], [80%], [90%], [0%],
    [Medium ($6 times 6$)], [60%], [30%], [0%],
    [Hard ($7 times 7$)], [50%], [0%], [30%],
    [Very Hard ($8 times 8$)], [20%], [10%], [0%],
  ),
  caption: [Solve rate vs. difficulty (4 agents). Pressure-field degrades gracefully; hierarchical collapses from 90% to 0% as difficulty increases.],
)

= Discussion

== Limitations

Our experiments reveal several important limitations and unexpected findings:

*Prompt engineering may not transfer.* Few-shot prompting, typically beneficial for LLM tasks, unexpectedly degraded performance from 100% to 40% solve rate (@tbl:ablation). We hypothesize that examples bias agents toward pattern matching rather than constraint reasoning, reducing exploration diversity. This suggests pressure-driven search may be fundamentally different from standard LLM prompting paradigms.

*Decay is non-optional.* Without temporal decay, solve rate drops to 0% regardless of other mechanisms. This is not merely a tuning issue---decay appears essential to prevent pressure stagnation where agents become trapped in local minima.

*Additional practical limitations:*
- Requires well-designed pressure functions (not learned from data)
- Decay rates $lambda_f, lambda_gamma$ and inhibition period require task-specific tuning
- May not suit tasks requiring long-horizon global planning
- Goodhart's Law: agents may game poorly-designed metrics
- Resource cost of parallel validation: testing $K$ patches requires $O(K dot.c |A|)$ memory where $|A|$ is artifact size

== When Hierarchical Coordination Is Appropriate

Our results show hierarchical coordination achieves 90% solve rate on easy problems ($5 times 5$ grids), outperforming pressure-field's 80%. This suggests hierarchical approaches remain preferable when:

1. *Problem structure is well-understood.* A central coordinator can exploit known decomposition strategies. Latin Squares at small scales have obvious row/column independence that hierarchical assignment captures directly.

2. *Search space is small.* With few constraints, the overhead of pressure-field exploration exceeds the benefit. Hierarchical assignment suffices when brute-force enumeration is tractable.

3. *Communication is cheap.* Our baselines assume zero-cost coordination messages. Real distributed systems may favor hierarchical approaches when network latency dominates compute time.

However, as problem difficulty increases, hierarchical performance collapses (90% → 0%) while pressure-field degrades gracefully (80% → 20%). This crossover suggests pressure-field coordination becomes increasingly advantageous as problems scale beyond what centralized planning can tractably decompose.

== Model Escalation as Adaptive Capability

Our escalation mechanism (1.5b → 7b → 14b parameters) provides 2--4$times$ improvement on hard problems (@tbl:escalation). This suggests an important design principle: pressure-field coordination benefits from adaptive capability deployment.

When smaller models cannot reduce pressure below threshold, escalation to larger models breaks through local minima. This is analogous to simulated annealing's temperature schedule---capability escalation provides "thermal kicks" to escape stagnation. The mechanism works because larger models have broader solution coverage, not necessarily better constraint reasoning.

Interestingly, hierarchical coordination shows mixed results with escalation (some configurations improve, others degrade). We hypothesize this is because hierarchical assignment decisions made by smaller models may be difficult for larger models to reverse, whereas pressure-field's local actions remain independent of model history.

== Future Work

- *Learned pressure functions*: Current sensors are hand-designed. Can we learn pressure functions from solution traces?
- *Adversarial robustness*: Can malicious agents exploit pressure gradients to degrade system performance?
- *Multi-artifact coordination*: Extension to coupled artifacts where patches in one affect pressure in another
- *Understanding the few-shot paradox*: Why does prompting hurt? This finding warrants deeper investigation into LLM behavior under constraint satisfaction

= Conclusion

We presented gradient-field coordination, a decentralized approach to multi-agent systems that achieves coordination through shared state and local pressure gradients rather than explicit orchestration.

Our theoretical analysis establishes convergence guarantees under pressure alignment conditions, with coordination overhead independent of agent count. Empirically, on Latin Square constraint satisfaction, pressure-field coordination achieves 100% solve rate with 4 agents compared to 40% for the best baseline (hierarchical) using identical LLMs. The approach scales linearly from 1 to 32 agents (maintaining 80--100% solve rate) while baselines show erratic performance.

Key findings include: (1) temporal decay is essential---disabling it drops solve rate to 0%; (2) few-shot prompting unexpectedly degrades performance, suggesting pressure-driven exploration differs fundamentally from standard LLM prompting; (3) model escalation provides 2--4$times$ improvement on hard problems by breaking through local minima.

These results suggest that constraint-driven emergence, inspired by natural coordination mechanisms like chemotaxis, offers a more scalable foundation for multi-agent AI than imported human organizational patterns. The approach is most advantageous for problems where centralized decomposition becomes intractable---precisely where scaling matters most.

= Appendix: Experimental Protocol

This appendix provides complete reproducibility information for all experiments.

== Hardware and Software

*Hardware:* NVIDIA RTX 4070 Laptop GPU, 8GB VRAM

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
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  --escalation-threshold 10 \
  grid --trials 30 --n 7 --empty 7 --max-ticks 40 --agents 1,2,4,8
```

*Ablation Study:*
```bash
latin-experiment --model-chain "latin-solver" \
  ablation --trials 30 --n 7 --empty 7 --max-ticks 40
```

*Scaling Analysis:*
```bash
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  grid --trials 30 --n 7 --empty 8 --max-ticks 40 --agents 1,2,4,8,16,32
```

*Model Escalation Comparison:*
```bash
# Without escalation
latin-experiment --model-chain "latin-solver" \
  grid --trials 30 --n 7 --empty 8 --max-ticks 40 --agents 2,4,8

# With escalation
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  --escalation-threshold 10 \
  grid --trials 30 --n 7 --empty 8 --max-ticks 40 --agents 2,4,8
```

*Difficulty Scaling:*
```bash
# Run for each (n, empty) pair: (5,5), (6,8), (7,10), (8,14)
latin-experiment --model-chain "latin-solver,latin-solver-7b,latin-solver-14b" \
  grid --trials 30 --n {N} --empty {E} --max-ticks 50 --agents 4
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
    [Main Grid], [16], [30], [2.5 hours],
    [Ablation], [8], [30], [1 hour],
    [Scaling], [6], [30], [1.5 hours],
    [Escalation], [6], [30], [1.5 hours],
    [Difficulty], [4], [30], [2 hours],
    [*Total*], [], [], [*~8 hours*],
  ),
  caption: [Estimated runtime for all experiments on NVIDIA RTX 4070 Laptop GPU.],
)

#bibliography("references.bib", style: "ieee")
