# HPL v2.3 — Optimised CPU Compilation Guide (`opt-avx2-openblas`, `opt-native-openblas` & `opt-lto-openblas`)

**Purpose:** Reproduce the three performance-optimised CPU builds of HPL v2.3 prepared for the CHPC Student Cluster Competition environment on AMD EPYC-Rome (`znver2`) / x86_64 clusters, complete with interactive node inspection, OpenBLAS NFS staging, SIMD instruction audits, and multi-node `HPL.dat` tuning verification.

**Pinned source:** Netlib `hpl-2.3.tar.gz` (`https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz`)  
**Build environment recorded:** Ubuntu 22.04.5 LTS, GCC 11.4.0, OpenMPI 4.1.2, OpenBLAS 0.3.20 (`libopenblas-dev`), GNU Make 4.3  
**Target CPU:** AMD EPYC-Rome Processor (`znver2`; `headnode`: 4 vCPUs, `compute001..003`: 8 vCPUs each = 24 compute cores total; also auto-detects AVX-512 on Intel Xeon Gold 5418Y if present)  
**Optimised build variants:** `opt-avx2-openblas` (Explicit AVX2/FMA + OpenBLAS), `opt-native-openblas` (Host Native `znver2` + OpenBLAS), and `opt-lto-openblas` (Interprocedural LTO + OpenBLAS)

> **Command provenance:** Each optimised variant is built in its own isolated directory under `$HOME/hpl/builds/` and links against the AVX2/Zen2-optimized **OpenBLAS (`libopenblas.so.0`)** library staged in the NFS-shared `$HOME/hpl/runtime/openblas/lib` directory with `-Wl,-rpath`. Every command block is immediately followed by **`What it does:`**, **`Expected output:`**, and a **`✅ Success Output:`** / **`✅ Success Indicator:`** check.

---

## 1. Compilation overview

| Build | Compiler & Vector Flags (`CFLAGS`) | BLAS Backend | Purpose |
|---|---|---|---|
| `opt-avx2-openblas` | `-O3 -mavx2 -mfma -mbmi2 -mf16c -funroll-loops -fomit-frame-pointer -fno-math-errno` | OpenBLAS (`libopenblas.so.0`) | **Explicit 256-bit AVX2/FMA build + OpenBLAS** |
| `opt-native-openblas` | `-O3 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno` (resolves to `znver2` on AMD EPYC-Rome) | OpenBLAS (`libopenblas.so.0`) | **Host CPU-specific optimized build (`znver2`) + OpenBLAS** |
| `opt-lto-openblas` | `-O3 -march=native -mtune=native -flto=auto -funroll-loops -fomit-frame-pointer -fno-math-errno` | OpenBLAS (`libopenblas.so.0`) | **Whole-program Link-Time Optimization (LTO) + OpenBLAS** inlining panel factorization & swap routines |

---

## Inspect the head node and compute nodes

On `headnode`, copy and execute this entire block to verify the AMD EPYC-Rome SIMD flags, resolved `-march=native` target, and OpenBLAS presence:

```bash
echo "=========================================="
echo "HEAD NODE ENVIRONMENT (HPL OPTIMISED CHECK)"
echo "=========================================="

hostname
cat /etc/os-release | grep PRETTY_NAME
lscpu | grep -E 'Architecture|Model name|Socket|Core|Thread|NUMA'

echo "CPU SIMD instruction sets:"
grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' |
grep -E '^(sse4_2|avx|avx2|fma|f16c|bmi2|avx512f)$' |
sort -u

echo ""
echo "=========================================="
echo "NATIVE COMPILER TARGET (-march=native)"
echo "=========================================="

gcc -march=native -Q --help=target 2>/dev/null | grep -E 'march=|mtune=' | head -n 4

echo ""
echo "=========================================="
echo "OPENBLAS DEVELOPMENT FILES"
echo "=========================================="

dpkg-query -W -f='${Status} ${Version}\n' libopenblas-dev 2>/dev/null || echo "Not installed"
ls -l /usr/lib/x86_64-linux-gnu/libopenblas*.so* 2>/dev/null || true
```

**What it does:** Verifies the AMD EPYC-Rome SIMD instruction sets (`avx`, `avx2`, `bmi2`, `f16c`, `fma`, `sse4_2`), confirms `-march=native` resolves to `-march=znver2 -mtune=znver2`, and checks `libopenblas-dev`.  
**Expected output:**
> ✅ **Success Indicator:** Lists `avx`, `avx2`, `bmi2`, `f16c`, `fma`, `sse4_2`, prints `-march=znver2` and `-mtune=znver2`, and shows `libopenblas.so` installed.

---

## 2. Stage OpenBLAS into NFS-shared `$HOME/hpl/runtime/openblas/lib`

Because `compute001`, `compute002`, and `compute003` mount `/home/ubuntu` via NFS, we copy the OpenBLAS shared libraries (`libopenblas*.so*`) and `libgfortran.so*` into `$HOME/hpl/runtime/openblas/lib/` and embed `-Wl,-rpath` during linking so all three compute nodes load OpenBLAS directly over NFS:

```bash
export ROOT="$HOME/hpl"
export SRC_DIR="$ROOT/source/hpl-2.3"
export BUILDS_DIR="$ROOT/builds"
export RUNS_DIR="$ROOT/runs"
export LOG_DIR="$ROOT/logs"
export OPENBLAS_LIB_DIR="$ROOT/runtime/openblas/lib"
export BUILD_JOBS="$(nproc)"

mkdir -p "$BUILDS_DIR" "$RUNS_DIR/opt-avx2-openblas" \
         "$RUNS_DIR/opt-native-openblas" "$RUNS_DIR/opt-lto-openblas" \
         "$LOG_DIR" "$OPENBLAS_LIB_DIR"

rm -rf "$OPENBLAS_LIB_DIR/"*
cp -a /usr/lib/x86_64-linux-gnu/libopenblas*.so* "$OPENBLAS_LIB_DIR/"
cp -a /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas*.so* "$OPENBLAS_LIB_DIR/" 2>/dev/null || true
ln -sf "$OPENBLAS_LIB_DIR/libopenblas.so.0" "$OPENBLAS_LIB_DIR/libopenblas.so"
ls -la "$OPENBLAS_LIB_DIR/"
```

**What it does:** Copies `libopenblas.so.0` and its linker symlink `libopenblas.so` into the NFS-shared `$HOME/hpl/runtime/openblas/lib/` directory.  
**Expected output:**
> ✅ **Success Indicator:** Lists `libopenblas.so` and `libopenblas.so.0` inside `/home/ubuntu/hpl/runtime/openblas/lib/`.

---

## 3. Build 1 — `opt-avx2-openblas` (Explicit 256-bit AVX2 + FMA + OpenBLAS)

### 3.1 Configure and compile `opt-avx2-openblas`

```bash
export BUILD_DIR="$BUILDS_DIR/opt-avx2-openblas"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR/bin"
cd "$BUILD_DIR"

set -o pipefail
CC=mpicc MPICC=mpicc \
CFLAGS="-O3 -mavx2 -mfma -mbmi2 -mf16c -funroll-loops -fomit-frame-pointer -fno-math-errno" \
LDFLAGS="-L$OPENBLAS_LIB_DIR -Wl,-rpath,$OPENBLAS_LIB_DIR" \
LIBS="-lopenblas -lm" \
"$SRC_DIR/configure" \
  --prefix="$BUILD_DIR" \
  2>&1 | tee "$LOG_DIR/configure-opt-avx2-openblas.log"

make -j "$BUILD_JOBS" 2>&1 | tee "$LOG_DIR/build-opt-avx2-openblas.log"
make install 2>&1 | tee -a "$LOG_DIR/build-opt-avx2-openblas.log"
ls -lh "$BUILD_DIR/bin/xhpl"
```

**What it does:** Configures and compiles HPL v2.3 with explicit 256-bit AVX2/FMA flags linked against the staged OpenBLAS library.  
**Expected output:**
> ✅ **Success Output:** Lists `/home/ubuntu/hpl/builds/opt-avx2-openblas/bin/xhpl` with executable permissions.

---

## 4. Build 2 — `opt-native-openblas` (Host Native `znver2` + OpenBLAS)

### 4.1 Configure and compile `opt-native-openblas`

```bash
export BUILD_DIR="$BUILDS_DIR/opt-native-openblas"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR/bin"
cd "$BUILD_DIR"

if grep -q "avx512f" /proc/cpuinfo; then
    export HPL_NATIVE_CFLAGS="-O3 -march=native -mtune=native -mprefer-vector-width=512 -funroll-loops -fomit-frame-pointer -fno-math-errno"
else
    export HPL_NATIVE_CFLAGS="-O3 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno"
fi
printf 'Selected HPL_NATIVE_CFLAGS: %s\n' "$HPL_NATIVE_CFLAGS"

set -o pipefail
CC=mpicc MPICC=mpicc \
CFLAGS="$HPL_NATIVE_CFLAGS" \
LDFLAGS="-L$OPENBLAS_LIB_DIR -Wl,-rpath,$OPENBLAS_LIB_DIR" \
LIBS="-lopenblas -lm" \
"$SRC_DIR/configure" \
  --prefix="$BUILD_DIR" \
  2>&1 | tee "$LOG_DIR/configure-opt-native-openblas.log"

make -j "$BUILD_JOBS" 2>&1 | tee "$LOG_DIR/build-opt-native-openblas.log"
make install 2>&1 | tee -a "$LOG_DIR/build-opt-native-openblas.log"
ls -lh "$BUILD_DIR/bin/xhpl"
```

**What it does:** Configures and compiles HPL v2.3 tuned specifically for the host CPU (`znver2` on AMD EPYC-Rome, or 512-bit AVX-512 on Xeon Gold 5418Y) and links against OpenBLAS.  
**Expected output:**
> ✅ **Success Output:** Lists `/home/ubuntu/hpl/builds/opt-native-openblas/bin/xhpl`.

---

## 5. Build 3 — `opt-lto-openblas` (Interprocedural LTO + OpenBLAS)

### 5.1 Configure and compile `opt-lto-openblas`

```bash
export BUILD_DIR="$BUILDS_DIR/opt-lto-openblas"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR/bin"
cd "$BUILD_DIR"

set -o pipefail
CC=mpicc MPICC=mpicc \
CFLAGS="$HPL_NATIVE_CFLAGS -flto=auto" \
LDFLAGS="-flto=auto -L$OPENBLAS_LIB_DIR -Wl,-rpath,$OPENBLAS_LIB_DIR" \
LIBS="-lopenblas -lm" \
"$SRC_DIR/configure" \
  --prefix="$BUILD_DIR" \
  2>&1 | tee "$LOG_DIR/configure-opt-lto-openblas.log"

make -j "$BUILD_JOBS" 2>&1 | tee "$LOG_DIR/build-opt-lto-openblas.log"
make install 2>&1 | tee -a "$LOG_DIR/build-opt-lto-openblas.log"
ls -lh "$BUILD_DIR/bin/xhpl"
```

**What it does:** Compiles HPL v2.3 with whole-program Link-Time Optimization (`-flto=auto`) so panel recursion, pivoting, and MPI ring communication routines inline across translation units.  
**Expected output:**
> ✅ **Success Output:** Lists `/home/ubuntu/hpl/builds/opt-lto-openblas/bin/xhpl`.

---

## 6. Verify all four builds (`baseline-netlib-o2` vs. Optimised OpenBLAS Variants)

### 6.1 Verify binaries, BLAS linkage (`ldd`), multi-node accessibility, and 4-rank smoke test speedup

```bash
for build in baseline-netlib-o2 opt-avx2-openblas opt-native-openblas opt-lto-openblas; do
  echo "===== $build ====="
  BIN="$BUILDS_DIR/$build/bin/xhpl"
  if [ -x "$BIN" ]; then
    echo "Binary  : OK ($BIN)"
    ldd "$BIN" | grep -E "blas|openblas"
    ldd "$BIN" | grep 'not found' || echo "Libraries: OK (No missing shared libraries)"
  else
    echo "MISSING : $BIN"
  fi
done

for node in compute001 compute002 compute003; do
  echo "===== Verifying opt-lto-openblas on $node ====="
  ssh "$node" "ldd $BUILDS_DIR/opt-lto-openblas/bin/xhpl | grep -E 'openblas|not found'"
done
```

**What it does:** Confirms that `baseline-netlib-o2` links to `netlib-blas/lib/libblas.so.3` while all three optimised builds link to `openblas/lib/libopenblas.so.0` across `headnode` and `compute001..003`.  
**Expected output:**
> ✅ **Success Output:** Every node resolves `libopenblas.so.0 => /home/ubuntu/hpl/runtime/openblas/lib/libopenblas.so.0` with zero missing libraries.

### 6.2 Compare 4-Rank Smoke Test Speedup (`baseline-netlib-o2` vs. `opt-lto-openblas`)

```bash
cd "$RUNS_DIR/opt-lto-openblas"
cp -f "$RUNS_DIR/baseline-netlib-o2/HPL.dat" ./HPL.dat

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mpirun -np 4 "$BUILDS_DIR/opt-lto-openblas/bin/xhpl" | tee "$LOG_DIR/smoke-opt-lto-openblas.log"

echo "=== Baseline (Netlib BLAS -O2) vs Optimised (OpenBLAS + LTO -O3) ==="
grep -E "WR11C2R4|PASSED" "$LOG_DIR/smoke-baseline-netlib-o2.log"
grep -E "WR11C2R4|PASSED" "$LOG_DIR/smoke-opt-lto-openblas.log"
```

**What it does:** Runs the exact same `N = 2000` 4-rank test on `opt-lto-openblas` with `OPENBLAS_NUM_THREADS=1` and prints the GFLOP/s comparison against `baseline-netlib-o2`.  
**Expected output:**
> ✅ **Success Indicator:** Both runs print `PASSED`, and `opt-lto-openblas` achieves a **$5\times$ to $15\times$ higher Gflops score** than `baseline-netlib-o2`!

---

## 7. Multi-Node Production Benchmark across `compute001..003` (24 Cores Total)

### 7.1 Generate a 65% RAM `HPL.dat` tuned for `compute001`, `compute002`, and `compute003`

First, check the exact physical RAM on `compute001`:
```bash
ssh compute001 "awk '/MemTotal/ {printf \"%.2f\n\", \$2/1024/1024}' /proc/meminfo"
```

Then generate `HPL.dat` for 3 compute nodes (24 cores total: `P = 4, Q = 6`, `NB = 192`) targeting **65% of physical node RAM** (set `NODE_RAM_GB` below to the value printed by `compute001`, e.g. `4.0` or `16.0`):

```bash
cd "$RUNS_DIR/opt-lto-openblas"
export NODE_RAM_GB=4.0

python3 -c '
import math, os
nodes, ram_gb, pct, P, Q, NB = 3, float(os.environ.get("NODE_RAM_GB", 4.0)), 65.0, 4, 6, 192
N = (int(math.sqrt(nodes * ram_gb * (1024**3) * (pct/100.0) / 8.0)) // NB) * NB
hpl_dat = f"""HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
{N:<12d} Ns
1            # of NBs
{NB:<12d} NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
{P:<12d} Ps
{Q:<12d} Qs
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
"""
open("HPL.dat", "w").write(hpl_dat)
print(f"Generated HPL.dat: N={N}, NB={NB}, P={P}, Q={Q} ({P*Q} ranks across {nodes} nodes, 65% of {nodes*ram_gb:.1f} GB)")
'
```

### 7.2 Launch across `compute001`, `compute002`, and `compute003` (24 Ranks, Core-Pinned)

```bash
cd "$RUNS_DIR/opt-lto-openblas"
mpirun -np 24 \
  --host compute001:8,compute002:8,compute003:8 \
  --bind-to core --map-by core \
  -x OPENBLAS_NUM_THREADS=1 \
  -x OMP_NUM_THREADS=1 \
  -x LD_LIBRARY_PATH="$OPENBLAS_LIB_DIR:${LD_LIBRARY_PATH:-}" \
  "$BUILDS_DIR/opt-lto-openblas/bin/xhpl" | tee "$LOG_DIR/prod-24ranks-opt-lto-openblas.log"
```

**What it does:** Launches 24 core-pinned MPI ranks across `compute001`, `compute002`, and `compute003` (8 ranks per node, `OPENBLAS_NUM_THREADS=1` to prevent thread oversubscription) and logs the final GFLOP/s score and residual check.  
**Expected output:**
> ✅ **Success Output:** Prints the `WR11C2R4` GFLOP/s score and `||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= ... PASSED`.
