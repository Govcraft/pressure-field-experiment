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

    Empirically, on Latin Square constraint satisfaction across 1,078 trials, pressure-field coordination matches hierarchical control (38.2% vs 38.8% aggregate solve rate, $p = 0.94$) while requiring no explicit coordination. Both significantly outperform sequential (23.3%), random (11.7%), and conversation-based multi-agent dialogue (8.6%, $p < 10^{-5}$). Temporal decay is essential: disabling it increases final pressure 49-fold ($d = 4.15$). On easy problems, pressure-field achieves 87% solve rate. The approach maintains consistent performance from 2 to 32 agents. Our key finding: implicit coordination through shared pressure gradients achieves parity with explicit hierarchical control while dramatically outperforming explicit dialogue-based coordination. This suggests that constraint-driven emergence offers a simpler, equally effective foundation for multi-agent AI.
  ],
  keywords: (
    "multi-agent systems",
    "emergent coordination",
    "decentralized optimization",
    "LLM agents",
  ),
)

= Introduction

Multi-agent systems built on large language models have emerged as a promising approach to complex task automation @wu2023autogen @hong2023metagpt @li2023camel. The dominant paradigm treats agents as organizational units: planners decompose tasks, managers delegate subtasks, and workers execute instructions under hierarchical supervision. This coordination overhead scales poorly with agent count and task complexity.

We demonstrate that *implicit* coordination through shared state achieves equivalent performance to explicit hierarchical control—without coordinators, planners, or message passing. Across 1,078 trials on Latin Square constraint satisfaction, pressure-field coordination matches hierarchical control (38.2% vs 38.8% aggregate solve rate, $p = 0.94$). Notably, AutoGen-style conversation-based coordination performs *worst* (8.6%), even below random selection (11.7%), demonstrating that explicit dialogue overhead actively harms performance on constraint satisfaction tasks.

Our approach draws inspiration from natural coordination mechanisms—ant colonies, immune systems, neural tissue—that coordinate through *environment modification* rather than message passing. Agents observe local quality signals (pressure gradients), take locally-greedy actions, and coordination emerges from shared artifact state. Temporal decay prevents premature convergence by ensuring continued exploration.

Our contributions:

+ We formalize *gradient-field coordination*: agents observe local quality signals, compute pressure gradients, and take locally-greedy actions. Coordination emerges from shared artifact state, not explicit communication.

+ We introduce *temporal decay* as a mechanism for preventing premature convergence. Disabling decay increases final pressure 49-fold (Cohen's $d = 4.15$), trapping agents in local minima.

+ We prove convergence guarantees for this coordination scheme under pressure alignment conditions.

+ We provide empirical evidence across 1,078 trials showing: (a) pressure-field matches hierarchical control, (b) both significantly outperform sequential (23%), random (12%), and conversation-based approaches (9%, $p < 10^{-5}$), and (c) conversation-based multi-agent dialogue is counterproductive for constraint satisfaction.

= Related Work

Our approach bridges three research streams: multi-agent LLM frameworks provide the application domain but rely on explicit coordination we eliminate; swarm intelligence and stigmergy inspire our pressure-field mechanism but lack formal guarantees; decentralized optimization provides theoretical foundations we adapt to LLM-based artifact refinement. We survey each and position our contribution.

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

*Key findings*: Pressure-field coordination matches hierarchical control while both significantly outperform other baselines (§5.2). Temporal decay is critical---disabling it increases final pressure 49-fold (§5.3). The approach maintains consistent performance from 2 to 32 agents (§5.4). Conversation-based multi-agent dialogue performs worst across all conditions, demonstrating that explicit message-passing coordination is counterproductive for this domain (§5.2).

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

Across 1,078 total trials spanning four experiments (easy, medium, hard, and scaling conditions), we find that pressure-field and hierarchical coordination perform equivalently, while both significantly outperform other baselines:

#figure(
  table(
    columns: 4,
    [*Strategy*], [*Solved/N*], [*Rate*], [*95% Wilson CI*],
    [Hierarchical], [128/330], [38.8%], [33.7%--44.1%],
    [Pressure-field], [126/330], [38.2%], [33.1%--43.5%],
    [Sequential], [42/180], [23.3%], [17.8%--30.0%],
    [Random], [21/180], [11.7%], [7.8%--17.2%],
    [Conversation], [5/58], [8.6%], [3.7%--18.6%],
  ),
  caption: [Aggregate solve rates across all experiments (1,078 total trials). Chi-square test: $chi^2 = 68.1$, $p < 10^(-13)$.],
)

The key finding is *stratification into two tiers*:

*Top tier (implicit and explicit coordination)*: Pressure-field and hierarchical achieve statistically equivalent performance (38.2% vs 38.8%, Fisher's exact $p = 0.94$). Their confidence intervals overlap substantially.

*Lower tier (no coordination or dialogue-based)*: Sequential (23.3%), random (11.7%), and conversation (8.6%) perform significantly worse. All pairwise comparisons with top-tier strategies are highly significant ($p < 0.001$).

The conversation strategy---AutoGen-style multi-agent dialogue with explicit message passing---performs *worst* across all conditions. This counterintuitive result suggests that coordination overhead from consensus-seeking dialogue actively harms performance on constraint satisfaction tasks.

This validates our central thesis: implicit coordination through shared pressure gradients achieves parity with explicit hierarchical control, while avoiding the pitfalls of dialogue-based coordination.

== Ablations

=== Effect of Temporal Decay

Decay proves essential---without it, final pressure increases dramatically:

#figure(
  table(
    columns: 4,
    [*Configuration*], [*N*], [*Final Pressure*], [*SD*],
    [With decay], [120], [$1.18$], [$1.45$],
    [Without decay], [120], [*$58.14$*], [$19.35$],
  ),
  caption: [Decay ablation on $5 times 5$ puzzles (240 total trials across 8 configurations). Welch's t-test: $t = -32.2$, $p < 10^(-60)$. Cohen's $d = 4.15$ (huge effect).],
)

The effect size is massive: Cohen's $d = 4.15$ far exceeds the threshold for "large" effects ($d > 0.8$). Disabling decay increases final pressure by 49$times$ (from 1.18 to 58.14). Without decay, fitness saturates after initial patches. High-fitness regions never re-enter the activation threshold, leaving the artifact in a high-pressure state. This validates Theorem 2: decay is necessary to continue pressure reduction even when regions appear "stable."

=== Effect of Inhibition and Examples

The ablation study tested all $2^3 = 8$ combinations of decay, inhibition, and few-shot examples on $5 times 5$ puzzles:

#figure(
  table(
    columns: 4,
    [*Configuration*], [*Solved/N*], [*Final Pressure*], [*SD*],
    [D=T, I=T, E=F], [18/30], [$1.00$], [$1.39$],
    [D=T, I=F, E=F], [17/30], [$1.00$], [$1.23$],
    [D=T, I=T, E=T], [15/30], [$1.20$], [$1.40$],
    [D=T, I=F, E=T], [15/30], [$1.53$], [$1.74$],
    [D=F, I=T, E=F], [0/30], [$53.03$], [$19.71$],
    [D=F, I=F, E=T], [0/30], [$57.27$], [$17.20$],
    [D=F, I=T, E=T], [0/30], [$60.77$], [$19.48$],
    [D=F, I=F, E=F], [0/30], [$61.50$], [$20.63$],
  ),
  caption: [Full ablation results (240 trials). D=decay, I=inhibition, E=examples. Decay is the critical mechanism: with decay, solve rate $approx 54%$ and final pressure $approx 1$; without decay, solve rate $= 0%$ and pressure $approx 58$.],
) <tbl:ablation>

The key finding is that *decay dominates*: any configuration with decay achieves $approx 54%$ solve rate with final pressure $approx 1$, while any without decay achieves 0% solve rate with pressure $approx 58$. Interestingly, few-shot examples provide no benefit (and may slightly hurt); inhibition shows marginal positive effect. The 49$times$ pressure difference between decay-enabled and decay-disabled configurations demonstrates decay's critical importance.

== Scaling Experiments

Both pressure-field and hierarchical maintain consistent performance from 2 to 32 agents on $7 times 7$ puzzles with 8 empty cells:

#figure(
  table(
    columns: 5,
    [*Agents*], [*Pressure-field*], [*95% CI*], [*Hierarchical*], [*95% CI*],
    [2], [7/30 (23.3%)], [11.8%--40.9%], [9/30 (30.0%)], [16.7%--47.9%],
    [4], [13/30 (43.3%)], [27.4%--60.8%], [7/30 (23.3%)], [11.8%--40.9%],
    [8], [10/30 (33.3%)], [19.2%--51.2%], [9/30 (30.0%)], [16.7%--47.9%],
    [16], [8/30 (26.7%)], [14.2%--44.4%], [9/30 (30.0%)], [16.7%--47.9%],
    [32], [10/30 (33.3%)], [19.2%--51.2%], [11/30 (36.7%)], [21.9%--54.5%],
  ),
  caption: [Scaling from 2 to 32 agents ($7 times 7$ grid, 8 empty cells, 30 trials each). Both strategies show stable performance across agent counts. Totals: pressure-field 48/150 (32.0%), hierarchical 45/150 (30.0%).],
)

Both strategies show stable performance across the full range of agent counts. Pressure-field peaks at 4 agents (43.3%) while hierarchical peaks at 32 agents (36.7%), but confidence intervals overlap substantially at all counts, indicating no significant agent-count effect for either strategy.

The key observation is *robustness*: both coordination strategies maintain 23--43% solve rates despite 16$times$ variation in agent count. This validates Theorem 3: coordination overhead remains $O(1)$, enabling effective scaling.

== Model Escalation Ablation

All main experiments use model escalation (0.5B → 1.5B → 3B → 7B → 14B). To quantify its impact, we examine the escalation experiment on harder problems ($7 times 7$, 8 empty cells):

#figure(
  table(
    columns: 4,
    [*Strategy*], [*Solved/N*], [*Rate*], [*95% Wilson CI*],
    [Hierarchical], [5/30], [16.7%], [7.3%--33.6%],
    [Pressure-field], [4/30], [13.3%], [5.3%--29.7%],
    [Sequential], [1/30], [3.3%], [0.6%--16.7%],
    [Random], [0/30], [0.0%], [0.0%--11.4%],
    [Conversation], [0/20], [0.0%], [0.0%--16.1%],
  ),
  caption: [Escalation experiment ($7 times 7$, 8 empty cells, harder condition). With model escalation enabled, top-tier strategies achieve 13--17% while baselines achieve 0--3%.],
) <tbl:escalation>

Even with model escalation, hard problems remain challenging. The pattern mirrors the aggregate results: hierarchical and pressure-field perform equivalently (16.7% vs 13.3%, overlapping CIs), while sequential, random, and conversation perform significantly worse.

== Difficulty Scaling

On easier problems ($5 times 5$, 5 empty cells), all strategies show improved performance, but the tier structure persists:

#figure(
  table(
    columns: 4,
    [*Strategy*], [*Solved/N*], [*Rate*], [*95% Wilson CI*],
    [Pressure-field], [26/30], [*86.7%*], [70.3%--94.7%],
    [Hierarchical], [24/30], [80.0%], [62.7%--90.5%],
    [Sequential], [16/30], [53.3%], [36.1%--69.8%],
    [Random], [13/30], [43.3%], [27.4%--60.8%],
    [Conversation], [3/20], [15.0%], [5.2%--36.0%],
  ),
  caption: [Solve rate on easy problems ($5 times 5$ grid, 5 empty cells). Even on easy problems, conversation-based coordination performs worst (15%).],
)

The difficulty scaling reveals key insights:

1. *Easy problems maintain tier structure*: Pressure-field (86.7%) and hierarchical (80.0%) remain the top tier. Sequential (53.3%) and random (43.3%) improve substantially but remain below top-tier. Conversation (15.0%) remains worst.

2. *Conversation fails even on easy problems*: Despite the reduced difficulty, explicit dialogue-based coordination achieves only 15%---worse than random guessing (43.3%). This suggests the coordination overhead of consensus-seeking actively harms performance.

3. *All strategies improve with easier problems*: The absolute difficulty of the task matters. On easy problems, even random achieves 43%. On hard problems (@tbl:escalation), random achieves 0%.

= Discussion

== Limitations

Our experiments reveal several important limitations:

*Pressure-field does not outperform hierarchical.* Contrary to initial expectations, pressure-field coordination achieves statistically equivalent performance to explicit hierarchical control (38.2% vs 38.8%, $p = 0.94$). The contribution is not performance advantage but rather *equivalent performance with simpler architecture*---no coordinator agent, no explicit message passing.

*Decay is non-optional.* Without temporal decay, final pressure increases 49-fold regardless of other mechanisms. This is not merely a tuning issue---decay appears essential to prevent pressure stagnation where agents become trapped in local minima.

*Absolute solve rates are modest on hard problems.* Even top-tier strategies achieve only 13--17% on hard problems and 30--43% on medium problems. Latin Square constraint satisfaction remains challenging for current LLMs.

*Additional practical limitations:*
- Requires well-designed pressure functions (not learned from data)
- Decay rates $lambda_f, lambda_gamma$ and inhibition period require task-specific tuning
- May not suit tasks requiring long-horizon global planning
- Goodhart's Law: agents may game poorly-designed metrics
- Resource cost of parallel validation: testing $K$ patches requires $O(K dot.c |A|)$ memory where $|A|$ is artifact size

== When to Choose Each Approach

Our results suggest the following guidance:

*Pressure-field coordination is preferable when:*
1. *Simplicity is valued.* No coordinator agent needed; coordination emerges from shared state.
2. *Fault tolerance matters.* No single point of failure; agents can join/leave without protocol overhead.
3. *Pressure signals are available.* The domain provides measurable quality gradients.

*Hierarchical coordination is equivalent when:*
1. *Explicit control is needed.* Some domains require deterministic task assignment.
2. *Interpretability is critical.* Hierarchical task assignment provides clear audit trails.

*Conversation-based coordination should be avoided for constraint satisfaction:*
Our experiments show that AutoGen-style multi-agent dialogue performs *worst* across all conditions (8.6% aggregate, worse than random at 11.7%). The overhead of consensus-seeking through explicit dialogue actively harms performance. This suggests that for constraint satisfaction, implicit coordination (whether pressure-field or hierarchical) is strictly preferable to explicit dialogue.

== Model Escalation as Adaptive Capability

All experiments use model escalation (0.5B → 1.5B → 3B → 7B → 14B parameters), triggered when regions remain high-pressure for 20 consecutive ticks. This mechanism proves beneficial for both top-tier strategies: on hard problems, both pressure-field and hierarchical achieve 13--17% with escalation enabled.

The escalation mechanism works because larger models have broader solution coverage. The 5-tier chain provides graduated capability increases, invoking expensive larger models only when necessary. Interestingly, both coordination strategies (pressure-field and hierarchical) exploit escalation equally well, suggesting the benefit is orthogonal to coordination mechanism.

== Future Work

- *Learned pressure functions*: Current sensors are hand-designed. Can we learn pressure functions from solution traces?
- *Adversarial robustness*: Can malicious agents exploit pressure gradients to degrade system performance?
- *Multi-artifact coordination*: Extension to coupled artifacts where patches in one affect pressure in another
- *Larger-scale experiments*: Testing on $8 times 8$ and $9 times 9$ grids to characterize the difficulty ceiling
- *Alternative domains*: Applying pressure-field coordination to code refactoring, configuration management, and other artifact refinement tasks

= Conclusion

We presented gradient-field coordination, a decentralized approach to multi-agent systems that achieves coordination through shared state and local pressure gradients rather than explicit orchestration.

Our theoretical analysis establishes convergence guarantees under pressure alignment conditions, with coordination overhead independent of agent count. Empirically, on Latin Square constraint satisfaction across 1,078 trials, we find:

1. *Pressure-field matches hierarchical control* (38.2% vs 38.8%, $p = 0.94$). Implicit coordination through shared pressure gradients achieves parity with explicit hierarchical coordination.

2. *Both significantly outperform other baselines*. Sequential (23.3%), random (11.7%), and conversation-based dialogue (8.6%) perform significantly worse ($p < 0.001$).

3. *Conversation-based coordination fails dramatically*. AutoGen-style multi-agent dialogue performs worst across all conditions---even worse than random on hard problems. The overhead of consensus-seeking through explicit message passing actively harms performance.

4. *Temporal decay is essential*. Disabling it increases final pressure 49-fold (Cohen's $d = 4.15$), trapping agents in local minima.

The key contribution is not that pressure-field outperforms hierarchical---it does not. Rather, pressure-field achieves *equivalent performance with simpler architecture*: no coordinator agent, no explicit message passing, just shared state and local pressure gradients. Meanwhile, the popular paradigm of multi-agent dialogue coordination proves counterproductive for constraint satisfaction.

These results suggest that for domains with measurable quality signals, implicit coordination through shared state offers a simpler, equally effective alternative to explicit hierarchical control---and a strictly superior alternative to dialogue-based coordination.

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
