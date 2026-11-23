# HPL (High-Performance Linpack): Application Research and HPC Benchmarking Guide

> **Project:** CHPC Student Cluster Competition (SCC) — National Round & Predictive HPC Cluster Monitoring  
> **Benchmark:** HPL v2.3 (High-Performance Linpack Benchmark for Distributed-Memory Clusters)  
> **Document type:** Research and implementation reference  
> **Research consolidation:** 23 November 2026  
> **Primary software:** [HPL v2.3 Official Release (Netlib / Innovative Computing Laboratory, UTK)](https://www.netlib.org/benchmark/hpl/)

---

## Table of contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [What HPL is](#2-what-hpl-is)
3. [Scientific and industry applications (TOP500 and cluster acceptance)](#3-scientific-and-industry-applications-top500-and-cluster-acceptance)
4. [How the application works](#4-how-the-application-works)
5. [Software architecture and dependencies](#5-software-architecture-and-dependencies)
6. [Stage 01: Theoretical peak performance ($R_{\text{peak}}$) and problem size ($N$) calculation](#6-stage-01-theoretical-peak-performance-r_textpeak-and-problem-size-n-calculation)
7. [Stage 02: 2D block-cyclic distribution ($NB$ and $P \times Q$ process grid)](#7-stage-02-2d-block-cyclic-distribution-nb-and-p-times-q-process-grid)
8. [Stage 03: Running `xhpl` (Panel factorization, broadcast, lookahead and `DGEMM`)](#8-stage-03-running-xhpl-panel-factorization-broadcast-lookahead-and-dgemm)
9. [Stage 04: Result analysis, residual verification and parallel efficiency ($R_{\text{max}} / R_{\text{peak}}$)](#9-stage-04-result-analysis-residual-verification-and-parallel-efficiency-r_textmax--r_textpeak)
10. [HPC performance metrics](#10-hpc-performance-metrics)
11. [Benchmark design for our Ubuntu HPC cluster](#11-benchmark-design-for-our-ubuntu-hpc-cluster)
12. [Recommended experiment workflow (with manual commands and verification)](#12-recommended-experiment-workflow-with-manual-commands-and-verification)
13. [Evidence collection and directory organisation](#13-evidence-collection-and-directory-organisation)
14. [Important experimental controls and interpretation](#14-important-experimental-controls-and-interpretation)
15. [Known considerations and limitations](#15-known-considerations-and-limitations)
16. [Glossary](#16-glossary)
17. [Official documentation and references](#17-official-documentation-and-references)

---

## 1. Purpose and scope

This README consolidates the research carried out on **HPL (High-Performance Linpack v2.3)** as the foundational floating-point compute benchmark for the **CHPC Student Cluster Competition (SCC) National Round** and HPC cluster performance monitoring. It is intended to be a comprehensive mathematical, architectural, and tuning reference.

The document explains:

- what HPL does, the dense linear system $A\mathbf{x} = \mathbf{b}$ it solves, and why it is the universal standard for ranking the **TOP500** supercomputers;
- how to calculate a cluster's **Theoretical Peak Performance ($R_{\text{peak}}$)** in GFLOP/s and compare it against **Measured Linpack Performance ($R_{\text{max}}$)** to determine **Linpack Efficiency ($R_{\text{max}} / R_{\text{peak}}$)**;
- how to calculate the exact matrix order **$N$** to target **65% to 80% of physical cluster RAM** without triggering the Linux OOM killer or swap thrashing;
- how the **2D Block-Cyclic Data Distribution** partitions an $N \times N$ double-precision matrix into $NB \times NB$ blocks across a 2D MPI process grid of **$P \times Q$** ranks;
- how every parameter in **`HPL.dat`** (`N`, `NB`, `PMAP`, `P`, `Q`, `PFACT`, `NBMIN`, `NDIV`, `RFACT`, `BCAST`, `DEPTH`, `SWAP`, `L1`, `U`, `EQUIL`) controls CPU vector utilization and MPI communication overlap;
- how HPL validates numerical accuracy using the scaled backward residual norm (which **must be $< 16.0$** to report `PASSED`); and
- how to compile, verify, and tune both a **Baseline (`netlib` / reference BLAS, `-O2`)** and **Optimised (OpenBLAS / AMD BLIS / Intel MKL, `-O3 -march=native` / `znver2` / `sapphirerapids`)** `xhpl` binary on Ubuntu Linux.

### 1.1 How to read this document

The document distinguishes three kinds of information:

- **Application behaviour:** verified mathematical and algorithmic behaviour of HPL v2.3.
- **Benchmark recommendation:** a proposed, repeatable methodology for calculating `HPL.dat` parameters and measuring $R_{\text{max}}$ across 1, 2, and 3 compute nodes.
- **Configuration-dependent behaviour:** details that depend on the CPU microarchitecture (AMD EPYC-Rome `znver2` with 256-bit AVX2 vs. Intel Xeon Gold 5418Y `sapphirerapids` with 512-bit AVX-512), BLAS library (OpenBLAS vs. BLIS vs. Intel MKL), and available node RAM.

---

## 2. What HPL is

**HPL (High-Performance Linpack)** is a portable, open-source parallel benchmark written in C by Antoine Petitet, R. Clint Whaley, Jack Dongarra, and Andy Cleary at the **Innovative Computing Laboratory (ICL), University of Tennessee, Knoxville**.

HPL measures the sustained double-precision (`FP64`, 64-bit IEEE 754 floating-point) computational throughput of distributed-memory computer clusters by solving a dense, randomly generated linear system of equations of order $N$:

$$A \mathbf{x} = \mathbf{b}, \qquad A \in \mathbb{R}^{N \times N}, \quad \mathbf{x}, \mathbf{b} \in \mathbb{R}^N$$

In plain language, **HPL creates a giant $N \times N$ square table of 64-bit decimal numbers in the cluster's combined RAM, distributes square tiles ($NB \times NB$) of that table across all MPI processes, solves for the unknown vector $\mathbf{x}$ using Gaussian elimination with row pivoting ($LU$ factorization), checks that the answer is mathematically accurate, and reports the speed in Billions of Floating-Point Operations per Second (GFLOP/s)**.

### 2.1 The LU Factorization algorithm in simple terms

To solve $A\mathbf{x} = \mathbf{b}$, HPL factors the augmented matrix $[A \mid \mathbf{b}]$ of dimension $N \times (N+1)$ into a unit lower-triangular matrix $L$ and an upper-triangular matrix $U$ with row permutations $P_{\text{piv}}$ (partial pivoting for numerical stability):

$$P_{\text{piv}} A = L U$$

Once $L$ and $U$ are found, $\mathbf{x}$ is computed via backwards substitution ($U\mathbf{x} = L^{-1} P_{\text{piv}} \mathbf{b}$).

The total number of floating-point operations (additions and multiplications) required to solve a dense system of order $N$ is fixed by linear algebra:

$$\text{Total FLOPs}(N) = \frac{2}{3} N^3 + 2 N^2 + \mathcal{O}(N)$$

- The **$LU$ factorization phase** performs $\frac{2}{3} N^3 + \frac{1}{2} N^2$ FLOPs (dominating $> 99.9\%$ of the work for large $N$).
- The **backward substitution phase** performs $2 N^2$ FLOPs.

If the solver takes $T$ wall-clock seconds to finish, HPL calculates the achieved performance $R_{\text{max}}$ (in GFLOP/s) as:

$$R_{\text{max}} \text{ (GFLOP/s)} = \frac{\frac{2}{3} N^3 + 2 N^2}{T \times 10^9}$$

### 2.2 Why HPL is the ultimate CPU + BLAS + Memory benchmark

Unlike sparse or stencil codes (such as WRF or HPCG) that are bottlenecked by DRAM memory bandwidth (~0.4 FLOPs/byte), HPL's core operation is **Level-3 BLAS Matrix-Matrix Multiplication (`DGEMM`: $C \leftarrow C - A \times B$)** on blocks of size $NB \times NB$:

- Performing `DGEMM` on two $NB \times NB$ blocks requires $2\,NB^3$ FLOPs while reading/writing only $3\,NB^2 \times 8\text{ bytes}$ of data.
- Its **Arithmetic Intensity** is $\frac{2\,NB^3}{24\,NB^2} = \frac{NB}{12}\text{ FLOPs/byte}$. For $NB = 192$, arithmetic intensity is **$16\text{ FLOPs/byte}$**—meaning that once an $NB \times NB$ panel is loaded into L2/L3 cache, the CPU's **SIMD Fused Multiply-Add (`vfmadd`) execution units** can run at 80%–95% of theoretical hardware peak speed!
- Consequently, HPL tests:
  1. **SIMD Vector Execution & BLAS Quality (`DGEMM`):** How well OpenBLAS, AMD BLIS, or Intel MKL packs matrices into L1/L2/L3 cache and feeds the AVX2 (256-bit) or AVX-512 (512-bit) FMA pipelines.
  2. **Thermal & Power Stability:** Sustained `DGEMM` drives CPUs to maximum power draw (TDP) and maximum AVX boost clock.
  3. **MPI Communication Overlap:** How efficiently panel broadcasts (`BCAST`) and lookahead (`DEPTH=1`) hide network latency across nodes.

---

## 3. Scientific and industry applications (TOP500 and cluster acceptance)

1. **The TOP500 Supercomputer Ranking:** Since 1993, the biannual **TOP500** list has ranked the world's 500 fastest supercomputers by their HPL $R_{\text{max}}$ score (measured in PFLOP/s or EFLOP/s).
2. **Student Cluster Competitions (CHPC SCC, ISC, SC, ASC):** In every Student Cluster Competition, HPL is the headline benchmark (often carrying a dedicated "Highest Linpack" award) that proves the team has properly tuned their compilers, BLAS libraries, MPI pinning, hugepages, and power/thermal limits.
3. **HPC System Acceptance & Burn-In Testing:** National laboratories (including South Africa's **CHPC**) run HPL during cluster acceptance to verify that every CPU core, DIMM channel, power supply, and interconnect link operates without hardware errors under maximum thermal load.
4. **Dense Linear Algebra in Science:** Direct numerical simulations in electromagnetics (Method of Moments radar cross-section modeling), boundary element acoustics, dense quantum chemistry (Hartree–Fock / Coupled Cluster matrix operations), and global geodetic inversion rely directly on distributed dense $LU$ and `DGEMM` kernels (`ScaLAPACK` / `SLATE`).

---

## 4. How the application works

HPL executes a right-looking, block-partitioned $LU$ factorization in steps of block size $NB$ across the $N \times N$ matrix:

```text
+-------------------------------------------------------------------------+
| Stage 1: Initialization & 2D Block-Cyclic Distribution                  |
|   -> Parse `HPL.dat` and arrange MPI ranks into a 2D P x Q process grid |
|   -> Allocate local matrix blocks and fill with pseudo-random FP64 data |
+-------------------------------------------------------------------------+
                                     |
                                     v
+=========================================================================+
| Stage 2: Main Block-LU Factorization Loop (k = 0, NB, 2*NB, ..., N)     |
|                                                                         |
|  Step A: Panel Factorization (Recursive / Crout / Left-Looking)         |
|    -> The single process column owning the current panel of width NB    |
|       factors the vertical panel and finds pivot rows (PFACT, RFACT).   |
|    -> Bottleneck: Memory bandwidth & vertical MPI column communication. |
|                                                                         |
|  Step B: Panel Broadcast (BCAST = 1ringM, 2ringM, etc.)                 |
|    -> The factored panel of width NB is broadcast horizontally across   |
|       the Q process columns so all ranks receive the panel.             |
|    -> If Lookahead (DEPTH = 1) is enabled, the next panel is factored   |
|       while the current trailing matrix update executes!                |
|                                                                         |
|  Step C: Row Swapping & Triangular Solve (SWAP & DTRSM)                 |
|    -> Apply pivot row permutations across the trailing matrix and solve |
|       the top NB x (N - k) row block via DTRSM.                         |
|                                                                         |
|  Step D: Trailing Submatrix Update (DGEMM — 85% to 95% of Total FLOPs)  |
|    -> Every rank updates its local trailing submatrix tiles using       |
|       Level-3 BLAS Matrix-Matrix Multiplication:                        |
|         A_trailing <-- A_trailing - L_panel * U_row                     |
|    -> Bottleneck: Pure CPU AVX2 / AVX-512 FMA Execution Units!          |
+=========================================================================+
                                     |
                                     v
+-------------------------------------------------------------------------+
| Stage 3: Backward Substitution & Scaled Residual Verification           |
|   -> Solve U * x = y via backward substitution.                         |
|   -> Compute scaled residual:                                           |
|        ||A*x - b||_inf / (eps * (||A||_inf * ||x||_inf + ||b||_inf) * N)|
|   -> Print GFLOP/s score and "PASSED" if residual < THRESHOLD (16.0).   |
+-------------------------------------------------------------------------+
```

---

## 5. Software architecture and dependencies

HPL has a clean three-layer software dependency stack:

| Layer | Component | Role in HPL | Recommended Ubuntu Choice |
|---|---|---|---|
| **1. Compiler** | C Compiler (`gcc` or `icx` / `clang`) | Compiles HPL's panel logic, MPI communication rings, and driver (`xhpl`). | `gcc 11.4.0` (`-O3 -march=native -mtune=native -funroll-loops`) |
| **2. MPI Library** | MPI (`OpenMPI` or `Intel MPI` / `MPICH`) | Executes 2D process grid (`MPI_Cart` / `MPI_Send` / `MPI_Recv`) for panel broadcasts and row swaps. | `OpenMPI 4.1.2` (`mpicc` / `mpirun`) |
| **3. BLAS Library** | Basic Linear Algebra Subprograms (`DGEMM`, `DTRSM`, `DGEMV`, `IDAMAX`) | **Performs 90%+ of all floating-point work.** Must be architecture-vectorized! | **Baseline:** Netlib Reference BLAS (`libblas.so`)<br>**Optimised:** **OpenBLAS** (`libopenblas.so`), **AMD BLIS** (on EPYC-Rome), or **Intel MKL** (on Xeon 5418Y) |

> **The Golden HPL Rule:** Never link HPL against unoptimized reference Netlib BLAS for your final competition score! Switching from reference BLAS (`libblas-dev`) to an AVX2/AVX-512 tuned BLAS (`libopenblas-dev`, AMD BLIS, or Intel MKL) typically increases HPL performance by **$5\times$ to $15\times$** on the exact same hardware.

---

## 6. Stage 01: Theoretical peak performance ($R_{\text{peak}}$) and problem size ($N$) calculation

### 6.1 How to calculate Theoretical Peak Performance ($R_{\text{peak}}$)

For any CPU cluster, the theoretical double-precision (`FP64`) peak performance in GFLOP/s is:

$$R_{\text{peak}} \text{ (GFLOP/s)} = (\text{Total Physical Cores}) \times (\text{Clock Frequency in GHz}) \times (\text{FP64 FLOPs per Cycle per Core})$$

How many `FP64 FLOPs per Cycle per Core` does a modern CPU execute?

$$\text{FP64 FLOPs/cycle/core} = (\text{Number of Vector FMA Units}) \times \left(\frac{\text{SIMD Register Width in Bits}}{64\text{ bits}}\right) \times 2 \text{ (1 Multiply + 1 Add per FMA)}$$

| Processor Architecture | SIMD Width | FP64 Elements / Vector | Vector FMA Units / Core | FP64 FLOPs / Cycle / Core | Example $R_{\text{peak}}$ Formula |
|---|---:|---:|---:|---:|---|
| **AMD EPYC-Rome (`znver2`)** | 256-bit (AVX2) | $256 / 64 = 4$ | 2 FMA units | $4 \times 2 \times 2 = \mathbf{16}$ | $N_{\text{cores}} \times f_{\text{GHz}} \times 16$ |
| **Intel Xeon Gold 5418Y (`sapphirerapids`)** | 512-bit (AVX-512) | $512 / 64 = 8$ | 2 FMA units | $8 \times 2 \times 2 = \mathbf{32}$ | $N_{\text{cores}} \times f_{\text{GHz}} \times 32$ |

**Worked Examples for Our Cluster Environments:**
1. **3× Compute Nodes (`compute001..003`, 8 AMD EPYC-Rome vCPUs/node = 24 cores total @ ~2.25 GHz):**
   $$R_{\text{peak}} = 24\text{ cores} \times 2.25\text{ GHz} \times 16\text{ FLOPs/cycle} = \mathbf{864.0\text{ GFLOP/s}}$$
2. **2× Physical HPE DL320 Gen11 Nodes (24-core Intel Xeon Gold 5418Y/node = 48 cores total @ 2.0 GHz base):**
   $$R_{\text{peak,base}} = 48\text{ cores} \times 2.0\text{ GHz} \times 32\text{ FLOPs/cycle} = \mathbf{3,072.0\text{ GFLOP/s}} \text{ (3.072 TFLOP/s)}$$

### 6.2 How to calculate the Optimal Problem Size ($N$) for 65%–80% RAM

An $N \times N$ double-precision matrix requires $N^2 \times 8\text{ bytes}$ of memory. Larger $N$ increases the ratio of $\mathcal{O}(N^3)$ `DGEMM` compute work to $\mathcal{O}(N^2)$ MPI communication overhead, pushing $R_{\text{max}}$ closer to $R_{\text{peak}}$. However, if $N^2 \times 8\text{ bytes}$ exceeds free physical RAM, Linux begins swapping to disk (dropping performance by $99\%$) or triggers the OOM killer (`ExitCode 137`).

The exact formula for matrix order $N$ given $N_{\text{nodes}}$, physical RAM per node $M_{\text{node,bytes}}$, and target memory fraction $\alpha \in [0.65, 0.80]$ is:

$$N_{\text{raw}} = \sqrt{\frac{\alpha \times N_{\text{nodes}} \times M_{\text{node,bytes}}}{8\text{ bytes}}}$$

**Crucial Alignment Rule:** Always round $N_{\text{raw}}$ down to an exact integer multiple of block size $NB$ (or even better, a multiple of $NB \times \text{LCM}(P, Q)$ so every MPI rank gets the exact same number of blocks!):

$$N = \left\lfloor \frac{N_{\text{raw}}}{NB} \right\rfloor \times NB$$

#### Ready-to-Use $N$ Lookup Table for Our Cluster Profiles ($NB = 192$)

| Cluster Environment | Nodes | RAM / Node | Total Cluster RAM | **65% Safe Target ($N$)** | **75% Peak Target ($N$)** | **80% Aggressive Target ($N$)** |
|---|---:|---:|---:|---:|---:|---:|
| **VMware 2 GB Nodes** | 3 | 2.0 GB | 6.0 GB | $N = \mathbf{22,848}$ (`3.89 GB`) | $N = \mathbf{24,576}$ (`4.50 GB`) | $N = \mathbf{25,344}$ (`4.79 GB`) |
| **Target 4 GB Nodes** | 2 | 4.0 GB | 8.0 GB | $N = \mathbf{26,304}$ (`5.16 GB`) | $N = \mathbf{28,224}$ (`5.94 GB`) | $N = \mathbf{29,184}$ (`6.34 GB`) |
| **Target 4 GB Nodes** | 3 | 4.0 GB | 12.0 GB | $N = \mathbf{32,256}$ (`7.75 GB`) | $N = \mathbf{34,560}$ (`8.90 GB`) | $N = \mathbf{35,712}$ (`9.50 GB`) |
| **16 GB Nodes (`compute001..003`)** | 3 | 16.0 GB | 48.0 GB | $N = \mathbf{64,512}$ (`31.0 GB`) | $N = \mathbf{69,312}$ (`35.8 GB`) | $N = \mathbf{71,616}$ (`38.2 GB`) |
| **48 GB Xeon 5418Y Nodes** | 2 | 48.0 GB | 96.0 GB | $N = \mathbf{91,200}$ (`61.9 GB`) | $N = \mathbf{97,920}$ (`71.4 GB`) | $N = \mathbf{101,184}$ (`76.3 GB`) |

---

## 7. Stage 02: 2D block-cyclic distribution ($NB$ and $P \times Q$ process grid)

Why doesn't HPL simply give the top half of matrix $A$ to Node 1 and the bottom half to Node 2?
- Because as Gaussian elimination progresses from column $1$ to column $N$, the active trailing submatrix shrinks toward the bottom-right corner. With a simple block decomposition, upper ranks would run out of work early and sit completely idle for the second half of the run!

HPL solves this using **2D Block-Cyclic Distribution**:
1. The $N \times N$ matrix is tiled into square blocks of size **$NB \times NB$**.
2. The total MPI processes ($N_{\text{MPI}}$) are arranged into a 2D logical grid of **$P$ rows by $Q$ columns** ($P \times Q = N_{\text{MPI}}$).
3. Blocks are dealt out cyclically like playing cards along both rows (modulo $P$) and columns (modulo $Q$). Even when the active trailing submatrix shrinks to the bottom-right corner, every single MPI rank still owns an equal $1/(P \times Q)$ share of the remaining blocks!

### 7.1 How to choose Block Size ($NB$)

- **$NB$ Too Small ($NB < 64$):** `DGEMM` arithmetic intensity ($NB/12$) is too low to reach peak FMA speed, and MPI message count increases.
- **$NB$ Too Large ($NB > 384$):** $NB \times NB$ blocks spill out of L2/L3 cache, panel factorization takes too long, and load balance at the tail end of the matrix degrades.
- **Optimal $NB$ by Architecture:**
  - **AMD EPYC-Rome (`znver2`, AVX2) with OpenBLAS / BLIS:** Test **`NB = 192`** and **`NB = 224`** (or `128`, `168`, `240`).
  - **Intel Xeon Gold 5418Y (`sapphirerapids`, AVX-512) with MKL / OpenBLAS:** Test **`NB = 192`**, **`NB = 256`**, and **`NB = 384`** (multiples of 64 align cleanly with 64-byte cache lines and 512-bit ZMM registers).

### 7.2 How to choose the Process Grid ($P \times Q$)

1. **Strict Rule:** $P \times Q$ must equal the number of MPI ranks passed to `mpirun -np <N>`.
2. **Aspect Ratio Rule ($1 \le Q/P \le 3$, with $P \le Q$):**
   - Keep $P$ and $Q$ close to a square, with **$P$ slightly smaller than $Q$** ($P \le Q$).
   - Why $P \le Q$? Panel factorization communicates vertically along a process column of length $P$ (involving latency-sensitive collective max-pivot searches at every single column within the panel), whereas panel broadcast sends one large horizontal message across $Q$ columns. Keeping $P$ smaller reduces vertical panel-factorization latency!
   - **Recommended $P \times Q$ Factorizations:**
     - **4 MPI ranks:** `P = 2, Q = 2` (or `P = 1, Q = 4`)
     - **8 MPI ranks (1 node):** `P = 2, Q = 4`
     - **16 MPI ranks (2 nodes):** `P = 4, Q = 4` (or `P = 2, Q = 8`)
     - **24 MPI ranks (3 nodes × 8 cores):** **`P = 4, Q = 6`** (primary) and **`P = 3, Q = 8`** (secondary)
     - **48 MPI ranks (2 nodes × 24 cores):** **`P = 6, Q = 8`** (primary) and **`P = 4, Q = 12`** (secondary)

---

## 8. Stage 03: Running `xhpl` (Panel factorization, broadcast, lookahead and `DGEMM`)

When `./xhpl` starts, it looks for a file named **`HPL.dat`** in the current working directory.

### 8.1 Complete Line-by-Line Reference for `HPL.dat`

```text
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
32256        Ns
1            # of NBs
192          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
4            Ps
6            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

| `HPL.dat` Parameter | Recommended Value | Detailed Technical Explanation |
|---|---|---|
| `Ns` | `32256` (for 3× 4GB nodes) | Matrix dimension $N$ (must be divisible by $NB$ and tuned to 65%–80% of physical RAM). |
| `NBs` | `192` (or `224`) | Block size $NB$ for 2D block-cyclic distribution and `DGEMM` blocking. |
| `PMAP` | `0` (Row-major) | Maps consecutive MPI ranks on the same physical node along rows ($Q$). Because $P$ ranks in a column factor panels simultaneously, `PMAP = 0` spreads a process column across nodes so panel factorization enjoys aggregate multi-node memory bandwidth! |
| `Ps`, `Qs` | `P = 4, Q = 6` (24 ranks) | 2D process grid dimensions ($P \times Q = N_{\text{MPI}}$, with $P \le Q$). |
| `threshold` | `16.0` | Standard TOP500 backward error threshold. **Never increase above `16.0`!** |
| `PFACTs` | `2` (Right) or `1` (Crout) | Panel factorization algorithm (`0=Left, 1=Crout, 2=Right`). You can list `3` values (`0 1 2`) during tuning sweeps to let HPL test all three in one run! |
| `NBMINs` | `4` (or `8`) | Minimum sub-column width where recursive panel factorization stops. |
| `NDIVs` | `2` | Number of sub-panels each recursive step divides into (`2` = binary recursion). |
| `RFACTs` | `1` (Crout) or `2` (Right) | Recursive panel factorization variant. |
| `BCASTs` | `1` (`1ringM`) or `3` (`2ringM`) | Horizontal panel broadcast topology (`0=1ring, 1=1ringM, 2=2ring, 3=2ringM, 4=Blong, 5=BlongM`). `1ringM` (modified 1-ring where the root rank sends to its immediate neighbor first) is almost always fastest on 1–4 nodes over Ethernet. |
| `DEPTHs` | `1` | **Lookahead depth:** `DEPTH = 1` allocates workspace so the next panel ($k+1$) is factored and broadcast concurrently while `DGEMM` updates the trailing matrix for panel $k$. Typically adds **3%–8% GFLOP/s** on multi-node Ethernet clusters! |
| `SWAP` | `2` (Mix) | Row-swapping communication algorithm (`0=binary exchange, 1=long/spread-roll, 2=mixed`). |
| `L1`, `U` | `0`, `0` (Transposed) | Stores $L_1$ and $U$ panels in transposed layout (`0`) so `DGEMM` reads them with unit stride (contiguous memory access). |
| `Equilibration` | `1` (Yes) | Enables row equilibration during spreading/rolling to improve load balance. |
| `memory alignment` | `8` | Aligns allocated `double` arrays on 64-byte (`8 * 8 bytes`) cache-line boundaries. |

---

## 9. Stage 04: Result analysis, residual verification and parallel efficiency ($R_{\text{max}} / R_{\text{peak}}$)

At the end of each problem configuration in `HPL.dat`, `xhpl` prints a result table and residual check:

```text
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR11C2R4       32256   192     4     6             18.42               1.215e+03
HPL_pdgesv() start time ...
HPL_pdgesv() end time   ...
--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0041821 ...... PASSED
================================================================================
```

### 9.1 How to decode the `T/V` (Test/Variant) code (e.g., `WR11C2R4`)
- **`W`**: `PMAP` (`R` = Row-major `0`, `C` = Column-major `1`)
- **`R`**: `SWAP` algorithm
- **`1`**: `DEPTH` (Lookahead depth `1`)
- **`1`**: `BCAST` (`1` = `1ringM`)
- **`C`**: `RFACT` (`L` = Left, `C` = Crout, `R` = Right)
- **`2`**: `NDIV` (`2` sub-panels)
- **`R`**: `PFACT` (`L` = Left, `C` = Crout, `R` = Right)
- **`4`**: `NBMIN` (`4` columns)

### 9.2 Residual Verification Rule
HPL computes the scaled infinity-norm residual:

$$R_{\text{res}} = \frac{\|A\mathbf{x} - \mathbf{b}\|_{\infty}}{\epsilon_{\text{mach}} \left(\|A\|_{\infty} \|\mathbf{x}\|_{\infty} + \|\mathbf{b}\|_{\infty}\right) N}, \qquad \epsilon_{\text{mach}} = 2^{-53} \approx 1.11 \times 10^{-16}$$

- **Valid Run:** $R_{\text{res}} < 16.0$ (typically $0.001 \le R_{\text{res}} \le 0.05$) and the line ends with **`PASSED`**.
- **Invalid Run:** `FAILED` or `NaN` (caused by corrupted memory, buggy BLAS, or unstable CPU overclocking).

---

## 10. HPC performance metrics

| Metric | Formula / Source | Typical Target Value | Why it matters |
|---|---|---|---|
| **Measured Performance ($R_{\text{max}}$)** | `Gflops` column in `HPL.out` | Maximize (GFLOP/s or TFLOP/s) | Primary competition score. |
| **Theoretical Peak ($R_{\text{peak}}$)** | $N_{\text{cores}} \times f_{\text{GHz}} \times \text{FLOPs/cycle}$ | Hardware ceiling | Denominator for calculating cluster efficiency. |
| **Linpack Efficiency ($E_{\text{HPL}}$)** | $\frac{R_{\text{max}}}{R_{\text{peak}}} \times 100\%$ | **Baseline (Netlib):** 5%–12%<br>**Optimised (OpenBLAS/BLIS/MKL):** **65%–85%** | Measures how effectively your BLAS, MPI, and `HPL.dat` tuning utilize the hardware. |
| **Memory Utilization (%)** | $\frac{N^2 \times 8\text{ bytes}}{\text{Total Physical Cluster RAM}} \times 100\%$ | **65%–80%** | Higher % reduces communication-to-compute ratio without triggering OOM or swap. |
| **Scaled Residual ($R_{\text{res}}$)** | `||Ax-b||_oo / (...)` in `HPL.out` | $< 16.0$ (typically $< 0.05$) | Mandatory mathematical proof of correct solution. |

---

## 11. Benchmark design for our Ubuntu HPC cluster

### 11.1 Hybrid MPI × OpenBLAS Threading vs. Pure MPI

HPL can be executed in two distinct parallelism modes:

1. **Mode A — Pure MPI (1 MPI Rank per Physical Core, `OPENBLAS_NUM_THREADS=1`):**
   - E.g., on 3 nodes × 8 cores = 24 MPI ranks (`P = 4, Q = 6`, `OPENBLAS_NUM_THREADS = 1`).
   - Simple, deterministic, zero OpenMP thread fork/join overhead.
2. **Mode B — Hybrid MPI + Multithreaded BLAS (1 MPI Rank per Socket / NUMA Node, `OPENBLAS_NUM_THREADS = Cores_per_Socket`):**
   - E.g., on 3 nodes × 8 cores: run **3 MPI ranks** (`P = 1, Q = 3`, 1 rank per node) or **6 MPI ranks** (`P = 2, Q = 3`, 2 ranks per node) with `OPENBLAS_NUM_THREADS=8` or `4`!
   - **Why Hybrid Mode Often Wins on Ethernet Clusters:** Reducing total MPI ranks from 24 to 3 or 6 shrinks the number of MPI network messages by **$4\times$ to $8\times$**, while OpenBLAS threads share panel data directly in L3 cache inside each node! Always test both Pure MPI (`24 × 1`) and Hybrid MPI+OpenBLAS (`6 × 4` and `3 × 8`) on your cluster.

---

## 12. Recommended experiment workflow (with manual commands and verification)

### Phase 1: Install Ubuntu prerequisites (OpenMPI, Netlib Reference BLAS & OpenBLAS)

```bash
sudo apt update && sudo apt install -y \
  build-essential gcc g++ gfortran make cmake git wget curl pkg-config \
  openmpi-bin libopenmpi-dev \
  libblas-dev liblapack-dev \
  libopenblas-dev \
  numactl hwloc python3
```
Verify OpenMPI and both BLAS libraries (`libblas.so` for Baseline and `libopenblas.so` for Optimised):
```bash
mpicc --version && ls -l /usr/lib/x86_64-linux-gnu/libblas.so /usr/lib/x86_64-linux-gnu/libopenblas.so
```
> ✅ **Success Output:** Displays `gcc (Ubuntu 11.4.0...)` and lists both `/usr/lib/x86_64-linux-gnu/libblas.so` and `/usr/lib/x86_64-linux-gnu/libopenblas.so`.

---

### Phase 2: Download and extract official HPL v2.3 source

```bash
export ROOT="$HOME/hpl"
mkdir -p "$ROOT/source" "$ROOT/builds" "$ROOT/runs" "$ROOT/logs"
cd "$ROOT/source"
wget -q https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar -xzf hpl-2.3.tar.gz
ls -la "$ROOT/source/hpl-2.3/Make.top"
```
> ✅ **Success Indicator:** `/home/ubuntu/hpl/source/hpl-2.3/Make.top` exists.

---

### Phase 3: Compile Baseline HPL (`baseline-netlib-o2`) and Optimised HPL (`opt-openblas-native`)

Follow the step-by-step manual compilation guides in:
- **Baseline Guide:** [`compilation_guide/baseline/README.md`](compilation_guide/baseline/README.md) (`-O2 -fno-fast-math` linked against reference Netlib `libblas.so`)
- **Optimised Guide:** [`compilation_guide/optimisation/README.md`](compilation_guide/optimisation/README.md) (`-O3 -march=native -mtune=native -funroll-loops` linked against multithreaded `libopenblas.so`)

---

### Phase 4: Generate `HPL.dat` automatically for your exact node count and 65% RAM ceiling

Run this interactive Python one-liner to calculate $N$ for **65% of physical node RAM** and write a valid `HPL.dat` file (adjust `nodes=3, ram_gb=4.0, P=4, Q=6, NB=192` as needed):

```bash
python3 -c '
import math
nodes, ram_gb, target_pct, P, Q, NB = 3, 4.0, 65.0, 4, 6, 192
tot_bytes = nodes * ram_gb * (1024**3) * (target_pct / 100.0)
N_raw = int(math.sqrt(tot_bytes / 8.0))
N = (N_raw // NB) * NB
actual_gb = (N * N * 8.0) / (1024**3)
actual_pct = (actual_gb / (nodes * ram_gb)) * 100.0
print(f"Cluster RAM : {nodes} nodes x {ram_gb:.1f} GB = {nodes*ram_gb:.1f} GB total")
print(f"Tuned N     : {N} (divisible by NB={NB}) | Matrix RAM = {actual_gb:.2f} GB ({actual_pct:.1f}% <= {target_pct:.0f}%)")
print(f"MPI Grid    : P={P} x Q={Q} = {P*Q} MPI ranks")
'
```
> ✅ **Success Output:**
> ```text
> Cluster RAM : 3 nodes x 4.0 GB = 12.0 GB total
> Tuned N     : 32256 (divisible by NB=192) | Matrix RAM = 7.75 GB (64.6% <= 65%)
> MPI Grid    : P=4 x Q=6 = 24 MPI ranks
> ```

---

### Phase 5: Execute `xhpl` and verify `PASSED` residual

```bash
mpirun -np 24 --bind-to core --map-by core ./xhpl | tee hpl_run.log
grep -E "WR|PASSED|FAILED" hpl_run.log
```
> ✅ **Success Output:**
> ```text
> WR11C2R4       32256   192     4     6             ...               ...e+02
> ||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0039412 ...... PASSED
> ```

---

## 13. Evidence collection and directory organisation

```text
hpl-chpc-scc/
├── README.md                          # Master HPL research & benchmarking reference
├── compilation_guide/
│   ├── baseline/
│   │   └── README.md                  # Standalone Baseline HPL compilation guide (Netlib BLAS, -O2)
│   └── optimisation/
│       └── README.md                  # Standalone Optimised HPL compilation guide (OpenBLAS, -O3/znver2)
```

---

## 14. Important experimental controls and interpretation

1. **Always warm up or run small $N$ first:** Before launching a full 65%–80% memory run (`N = 32256`), test `N = 4000` (`~128 MB`) to confirm `xhpl` launches across all nodes and prints `PASSED` within 2 seconds.
2. **Never oversubscribe cores (`MPI Ranks_per_Node × OPENBLAS_NUM_THREADS <= Physical_Cores_per_Node`):** If you launch 8 MPI ranks on an 8-core node (`compute001`), you **must** set `export OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS=1`). If `OPENBLAS_NUM_THREADS` is left unconfigured, OpenBLAS will spawn 8 threads per rank ($8 \times 8 = 64$ threads on 8 cores!), causing catastrophic lock contention and cutting your GFLOP/s score by 80%!
3. **Never change `threshold` above `16.0`:** Any run with residual $\ge 16.0$ is disqualified under TOP500 and SCC rules.
4. **Keep $N$ an integer multiple of $NB$:** Avoid partial edge blocks that cause unaligned `DGEMM` tails.

---

## 15. Known considerations and limitations

1. **Asymmetric Node Core Counts (`headnode` 4 vCPUs vs. `compute001..003` 8 vCPUs):** Do **not** include `headnode` in the same homogeneous HPL run as `compute001..003` unless you restrict every node to 4 ranks, because HPL divides matrix blocks evenly across all $P \times Q$ MPI ranks; the slowest or most overloaded node throttles the entire 2D grid. Run HPL exclusively across the three identical compute nodes (`compute001`, `compute002`, `compute003` = 24 cores total).
2. **Shared Library Propagation over NFS:** Copy `libopenblas.so*` into `$HOME/hpl/runtime/openblas/lib` and link `xhpl` with `-Wl,-rpath,$HOME/hpl/runtime/openblas/lib` so `compute001..003` resolve OpenBLAS over NFS automatically.

---

## 16. Glossary

| Term | Explanation |
|---|---|
| **`BCAST`** | Horizontal panel broadcast topology (`0=1ring, 1=1ringM, 2=2ring, 3=2ringM, 4=Blong, 5=BlongM`). |
| **Block-Cyclic Distribution** | Matrix partitioning scheme that deals $NB \times NB$ blocks cyclically across a $P \times Q$ process grid to maintain load balance as the active submatrix shrinks. |
| **`DEPTH` (Lookahead)** | Number of future panels factored concurrently while `DGEMM` updates the trailing matrix (`DEPTH = 1` hides network broadcast latency). |
| **`DGEMM`** | Double-precision General Matrix-Matrix Multiply ($C \leftarrow \alpha A B + \beta C$); performs $>90\%$ of HPL's floating-point operations. |
| **`NB`** | Block size (in rows/columns) used for data distribution and Level-3 BLAS cache blocking. |
| **`P × Q`** | Dimensions of the 2D logical MPI process grid ($P \times Q = \text{Total MPI Ranks}$). |
| **`R_max`** | Maximum achieved HPL performance in GFLOP/s (or TFLOP/s) on a valid (`PASSED`) run. |
| **`R_peak`** | Theoretical hardware peak performance: $\text{Cores} \times \text{GHz} \times \text{FP64 FLOPs/cycle}$. |

---

## 17. Official documentation and references

- [Official Netlib HPL v2.3 Homepage & Tarball](https://www.netlib.org/benchmark/hpl/)
- [HPL Tuning Guide by Antoine Petitet et al.](https://www.netlib.org/benchmark/hpl/tuning.html)
- [TOP500 Linpack Benchmark Rules](https://www.top500.org/project/linpack/)
- [OpenBLAS GitHub Repository](https://github.com/OpenMathLib/OpenBLAS)
