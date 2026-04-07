# Prompt for Gemini — IEEE System Architecture Figure (TikZ)

---

## Your Role

You are an expert in LaTeX/TikZ figure design for IEEE academic papers. Your task is to generate a **publication-quality system architecture diagram** in TikZ for a research paper titled *"Graph-Matcher-PPO: Adaptive Task Graph Shaping for Heterogeneous UAV Swarms via Deep Reinforcement Learning"*.

---

## Paper Background (Read Carefully)

The paper proposes a reinforcement learning framework that optimizes DAG task scheduling in a heterogeneous UAV swarm. The **core novelty** is that the task graph topology is *not fixed* — the RL agent adaptively **reshapes** it (Split or Merge nodes) at each step based on real-time UAV resource feedback.

### System Components

**1. Task Graph** $\mathcal{G}_T = (\mathcal{V}_T, \mathcal{E}_T)$
- Nodes $v_i$: each has CPU demand $c_i$ and data volume $s_i$
- Edges $(v_i, v_j)$: carry transfer volume $d_{ij}$ (MB)
- Topology changes at each epoch via Split/Merge

**2. Resource Graph** $\mathcal{G}_R = (\mathcal{U}, \mathcal{L})$
- 8 heterogeneous UAVs $\{u_1, \ldots, u_8\}$, fully connected
- Each UAV has: CPU capacity $\phi_k$, battery $\beta_k(t)$, bandwidth
- Feature vector: $\mathbf{x}_k^R(t) = [\tilde\phi_k(t)/\phi^{\max},\ \bar B_k(t)/B_{\max},\ \beta_k(t)]$

**3. GNN-Based RL Policy (PPO)**
- Encodes both graphs with GraphSAGE layers
- Cross-graph attention: task nodes attend to resource nodes
- Actor outputs action $a_t$; Critic estimates $V(s_t)$

**4. Three Shaping Actions** $a_t \in \mathcal{A}$:
- `noop`: leave graph unchanged
- `Split(v_i)`: decompose node → $c_i \leftarrow c_i/2$, add sibling $v_i^+$ in parallel
- `Merge(v_i, v_j)`: fuse two topologically independent nodes → $c_i \leftarrow c_i + c_j$, remove $v_j$

**5. Deterministic Greedy Scheduler** (environment component, NOT learned)
- Assigns tasks in topological order to UAV with highest residual CPU
- Execution latency: $L_j^{\mathrm{cmp}} = \kappa \cdot c_j / \tilde\phi_l(t)$
- Transfer latency: $L_j^{\mathrm{com}}(t) = \sum_{v_i \in \Pi(v_j)} d_{ij} / B_{\sigma(i)l}(t)$

**6. UAV Swarm Dynamics**
- Mobility: Random Waypoint model in $[0, 600\text{m}]^2$ arena
- Bandwidth decay: $B_{kl}(t) = B_{\min} + (B_{\max} - B_{\min})\exp(-r_{kl}(t)/\delta_H)$
- Energy: $\beta_k(t+1) = \beta_k(t) - \alpha_c\rho_k(t) - \alpha_t\tau_k(t)$
- CPU derating: $\tilde\phi_k(t) = 0$ if depleted; $\gamma\phi_k$ if low-power; $\phi_k$ otherwise

**7. Scheduling Feedback Vector** $\mathbf{f}(t) \in \mathbb{R}^6$:
$$\mathbf{f}(t) = [\lambda(t),\ \sigma(t),\ \varrho(t),\ \bar u(t),\ \hat B_{\min}(t),\ \hat\beta_{\min}(t)]$$
- $\lambda = T_{\mathrm{span}}/T_D$: normalised makespan
- $\sigma$: task success rate
- $\varrho$: reschedule count
- $\bar u$: mean fleet CPU utilisation
- $\hat B_{\min}$: normalised minimum link bandwidth
- $\hat\beta_{\min}$: minimum battery fraction

---

## Closed-Loop Flow (The Key Logic)

```
Observation O(t) = {G_T,  G_R,  f(t)}
         ↓
  GNN Policy (PPO)
         ↓  action a_t
  Shaping Ops [Split / Merge / noop]
         ↓  G_T'
  Greedy Scheduler ← B(t), φ̃(t) ← UAV Swarm
         ↓  schedule result
  Energy update: β_k(t+1) = β_k(t) - Δβ_k(t)
         ↓
  New feedback f(t+1) ──────────────────────────→ back to top
```

---

## Figure Requirements

### Layout (3-row vertical flow, double-column IEEE `figure*`)

```
ROW 1 — Observation O(t)          [light gray dashed background]
  ┌───────────────┐  ┌─────────────────────┐  ┌──────────────────┐
  │  Task DAG     │  │  Resource Graph G_R  │  │  Feedback f(t)   │
  │  G_T=(V,E)    │  │  x_k^R=[φ̃,B̄,β]    │  │  [λ,σ,ϱ,ū,B̂,β̂]│
  └───────────────┘  └─────────────────────┘  └──────────────────┘

ROW 2 — GNN-Based RL Policy (PPO) [light blue background, thick border]
  Task-GNN ⊕ Res-GNN  →[cross-graph attention]→  Actor→a_t ; Critic→V(s)

  [action strip, white box]:  a_t ∈ A: noop | Split(v_i) | Merge(v_i,v_j)

ROW 3 — Environment Step           [light orange dashed background]
  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
  │  Shaping Ops │→ │ Modified DAG │→ │ Greedy Scheduler │  │  UAV Swarm Dyn. │
  │  Split/Merge │  │  G_T'        │  │  (deterministic) │  │  Mobility: RWP  │
  │              │  │  |V'|=N±1    │  │  topo order      │  │  B(t), φ̃_k(t)  │
  └──────────────┘  └──────────────┘  │  + BW latency    │  │  β_k(t+1)       │
                                      └──────────────────┘  └─────────────────┘
```

### Arrows

| Arrow | Type | Label |
|-------|------|-------|
| G_T, G_R, f(t) → Agent | solid → | (no label, three lines) |
| Agent → action strip | solid → | `action a_t` |
| action strip → Shaping Ops | solid → | (L-bend) |
| Shaping → ModDAG → Scheduler | solid → | — |
| UAV Swarm → Scheduler | dashed gray → | $\mathbf{B}(t),\ \tilde\phi_k(t)$ (above) |
| Scheduler → UAV Swarm | dashed gray → | $\rho_k(t),\ \tau_k(t)$ (below) |
| UAV Swarm → Feedback f(t+1) | **thick solid →** routed along right side | `f(t+1)` with components listed |

### Visual Style

- Font: `\footnotesize` everywhere, `\scriptsize` inside boxes
- Node fills: gray!9 (observation), blue!5 (agent, thick border), orange!7 (environment), white (action strip)
- Section backgrounds: dashed border + very light fill (gray!3, orange!3)
- Section labels: `\scriptsize\itshape` above/below background boxes
- Line widths: 0.55pt default, 0.65pt for main arrows, 0.80pt for feedback loop arrow
- Arrow tips: `Stealth` style

### Required TikZ Libraries
```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, fit, backgrounds, calc}
\usepackage{amsmath, amssymb}
```

### Output Format
- Wrap in `\begin{figure*}[t] ... \end{figure*}`
- Include a complete IEEE-style `\caption{...}` summarizing the closed-loop
- Include `\label{fig:framework}`
- Must compile cleanly in a standard IEEE LaTeX template (`IEEEtran.cls`)

---

## Reference TikZ Draft (Improve This)

The following is an existing draft. It has the correct structure but may have rendering issues or layout imprecision. Please **fix any bugs, improve alignment, and make it publication-ready**:

```latex
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  font=\footnotesize,
  >=Stealth,
  line width=0.55pt,
  sbox/.style={
    draw, rectangle, rounded corners=2pt,
    fill=gray!9, text width=#1,
    minimum height=0.90cm, align=center
  }, sbox/.default=3.0cm,
  ebox/.style={
    draw, rectangle, rounded corners=2pt,
    fill=orange!7, text width=#1,
    minimum height=0.90cm, align=center
  }, ebox/.default=2.7cm,
  abox/.style={
    draw=black!80, rectangle, rounded corners=3pt,
    fill=blue!5, line width=0.9pt,
    text width=11.4cm, minimum height=1.18cm, align=center
  },
  actbox/.style={
    draw, rectangle, rounded corners=2pt,
    fill=white, text width=7.2cm,
    minimum height=0.62cm, align=center
  },
  arr/.style={->, line width=0.65pt, rounded corners=3pt},
  darr/.style={->, dashed, line width=0.58pt, gray!70, rounded corners=3pt},
  fbarr/.style={->, line width=0.80pt, rounded corners=6pt},
]

\node[sbox=3.1cm] (gt) at (2.0, 6.50) {
  \textbf{Task DAG}\; $\mathcal{G}_T$\\[-1pt]
  \scriptsize $v_i{:}\,(c_i,\,s_i)$;\enspace $(v_i,v_j){:}\,d_{ij}$
};
\node[sbox=3.5cm] (gr) at (6.10, 6.50) {
  \textbf{Resource Graph}\; $\mathcal{G}_R$\\[-1pt]
  \scriptsize $\mathbf{x}_k^R {=} [\tilde\phi_k/\phi^{\max},\; \bar B_k/B_{\max},\; \beta_k]$
};
\node[sbox=3.1cm] (fb) at (10.2, 6.50) {
  \textbf{Feedback}\; $\mathbf{f}(t)\!\in\!\mathbb{R}^6$\\[-1pt]
  \scriptsize $[\lambda,\,\sigma,\,\varrho,\, \bar u,\,\hat B_{\min},\,\hat\beta_{\min}]$
};

\begin{scope}[on background layer]
  \node[draw, dashed, rounded corners=5pt, fill=gray!3, inner sep=8pt,
        fit=(gt)(gr)(fb)] (obsbox) {};
\end{scope}
\node[above=2pt of obsbox, font=\scriptsize\itshape] {Observation $\mathcal{O}(t)$};

\node[abox] (agent) at (6.10, 4.90) {
  \textbf{GNN-Based RL Policy}\enspace(PPO)\\[2pt]
  \scriptsize
  Task-GNN\enspace$\oplus$\enspace Res-GNN
  \enspace$\xrightarrow{\;\text{cross-graph attention}\;}$\enspace
  Actor $\!\rightarrow\! a_t$\enspace;\enspace Critic $\!\rightarrow\! V(s_t)$
};

\draw[arr] (gt.south)  -- ++(0,-.28) -| ([xshift=-2.6cm]agent.north);
\draw[arr] (gr.south)  -- ++(0,-.28) --  (agent.north);
\draw[arr] (fb.south)  -- ++(0,-.28) -| ([xshift=2.4cm]agent.north);

\node[actbox] (act) at (6.10, 3.68) {
  \scriptsize $a_t \!\in\! \mathcal{A}$:\enspace
  $\mathtt{noop}$\enspace$\mid$\enspace
  \textsc{Split}$(v_i)$\enspace$\mid$\enspace
  \textsc{Merge}$(v_i,\,v_j)$
};
\draw[arr] (agent.south) -- (act.north)
  node[midway, right=3pt, font=\scriptsize] {action $a_t$};

\node[ebox=2.70cm] (shp) at ( 1.50, 2.30) {
  \textbf{Shaping Ops}\\[-1pt]
  \scriptsize Split:\enspace$c_i\!\leftarrow\!c_i/2$\\[-2pt]
  Merge:\enspace$c_i\!\leftarrow\!c_i\!+\!c_j$
};
\node[ebox=2.30cm] (gtp) at ( 4.45, 2.30) {
  \textbf{Modified DAG}\\[1pt] $\mathcal{G}_T'$\\[-2pt]
  \scriptsize $|\mathcal{V}_T'|\!=\!N\!\pm\!1$
};
\node[ebox=2.75cm] (sch) at ( 7.30, 2.30) {
  \textbf{Greedy Scheduler}\\[-1pt]
  \scriptsize\itshape(deterministic)\\[-1pt]
  \scriptsize topological order\\[-2pt] $+$ BW-aware latency
};
\node[ebox=2.95cm] (uav) at (10.70, 2.30) {
  \textbf{UAV Swarm Dyn.}\\[-1pt]
  \scriptsize Mobility: RWP\\[-2pt]
  $\mathbf{B}(t)$,\enspace $\tilde{\phi}_k(t)$\\[-2pt]
  $\beta_k(t{+}1){=}\beta_k(t){-}\Delta\beta_k(t)$
};

\begin{scope}[on background layer]
  \node[draw, dashed, rounded corners=5pt, fill=orange!3, inner sep=8pt,
        fit=(shp)(gtp)(sch)(uav)] (envbox) {};
\end{scope}
\node[below=2pt of envbox, font=\scriptsize\itshape] {Environment Step};

\draw[arr] (act.south) -- ++(0,-.30) -| (shp.north);
\draw[arr] (shp.east)  -- (gtp.west);
\draw[arr] (gtp.east)  -- (sch.west);

\draw[darr] (uav.north) -- ++(0,+.55) -| (sch.north)
  node[pos=0.30, above, font=\scriptsize] {$\mathbf{B}(t),\;\tilde\phi_k(t)$};
\draw[darr] (sch.south) -- ++(0,-.42) -| (uav.south)
  node[pos=0.30, below, font=\scriptsize] {$\rho_k(t),\;\tau_k(t)$};

\draw[fbarr]
  (uav.east) -- ++(0.55, 0) -- ++(0, 4.90) -- (fb.east)
  node[pos=0.50, right=4pt, align=left, font=\scriptsize]
    {$\mathbf{f}(t{+}1)$\\[-1pt]
     \scriptsize$\!=\![\lambda,\sigma,\varrho,$\\[-1pt]
     \scriptsize$\;\bar u,\hat B_{\min},\hat\beta_{\min}]$};

\end{tikzpicture}
\caption{System architecture of the proposed \textsc{Graph-Matcher-PPO} framework.
At each epoch $t$, the GNN-based PPO policy observes $\mathcal{O}(t) = \{\mathcal{G}_T, \mathcal{G}_R, \mathbf{f}(t)\}$
and selects action $a_t$. The shaping operation transforms $\mathcal{G}_T{\to}\mathcal{G}_T'$, which the
deterministic greedy scheduler maps to a UAV assignment using $\mathbf{B}(t)$ and $\tilde\phi_k(t)$;
the resulting workload $(\rho_k, \tau_k)$ depletes $\beta_k(t{+}1)$, and the scheduling outcome
is encoded in $\mathbf{f}(t{+}1)$, closing the adaptive loop.}
\label{fig:framework}
\end{figure*}
```

---

## What To Deliver

1. **A corrected, improved, and compilable TikZ figure** that matches all requirements above.
2. The figure should fit cleanly inside an **IEEE `figure*` double-column** environment.
3. All math symbols must use proper LaTeX (`\mathcal`, `\tilde`, `\hat`, `\mathbf`, etc.).
4. Use **passive academic English** only in the caption (no "we/our").
5. Optionally suggest one alternative layout (e.g., horizontal flow or circular) with a brief rationale.
