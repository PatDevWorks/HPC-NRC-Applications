# HPL v2.3 — Baseline CPU Compilation Guide (`baseline-netlib-o2` Golden Reference)

**Purpose:** Reproduce the conservative Golden Reference CPU build (`baseline-netlib-o2`, `-O2 -fno-fast-math`, linked against reference Netlib BLAS `libblas.so`) of HPL v2.3 for the CHPC Student Cluster Competition environment.

**Pinned source:** Netlib `hpl-2.3.tar.gz` (`https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz`)  
**Build environment recorded:** Ubuntu 22.04.5 LTS, GCC 11.4.0, OpenMPI 4.1.2, Netlib Reference BLAS (`libblas-dev`), GNU Make 4.3  
**Target CPU:** AMD EPYC-Rome Processor (`znver2`; `headnode`: 4 vCPUs, `compute001..003`: 8 vCPUs each = 24 compute cores total)  
**Build variant:** `baseline-netlib-o2` (`-O2 -fno-fast-math` + unoptimized reference `libblas.so`)

> **Command provenance:** This guide uses HPL v2.3's modern Autotools `./configure` + `make` build system (introduced in HPL 2.3) in an isolated build directory (`$HOME/hpl/builds/baseline-netlib-o2`), staging `libblas.so*` into an NFS-shared runtime directory so `compute001..003` can run `xhpl` without needing local package installations.

---

## 1. Compilation overview

| Build | Compiler & Flags | BLAS Backend | Purpose |
|---|---|---|---|
| `baseline-netlib-o2` | `mpicc` (`-O2 -g -fno-fast-math -fomit-frame-pointer`) | Netlib Reference BLAS (`libblas.so.3`) | **Conservative Golden Reference build:** establishes baseline GFLOP/s ($1.00\times$) and verifies residual accuracy (`PASSED`) before BLAS & ISA tuning |

---

## Inspect the head node and compute nodes

On `headnode`, copy and execute this entire block:

```bash
echo "=========================================="
echo "HEAD NODE ENVIRONMENT (HPL BASELINE CHECK)"
echo "=========================================="

hostname
cat /etc/os-release | grep PRETTY_NAME
lscpu | grep -E 'Architecture|Model name|Socket|Core|Thread|NUMA'

for cmd in gcc mpicc mpirun make wget tar; do
    printf "Checking %-10s: " "$cmd"
    command -v "$cmd" || echo "MISSING"
done

gcc --version | head -1
mpicc --version | head -1
make --version | head -1
df -h "$HOME"
```

**What it does:** Audits the head node OS, AMD EPYC-Rome CPU topology, `gcc`, `mpicc`, `mpirun`, `make`, and available disk space on `$HOME`.  
**Expected output:**
> ✅ **Success Indicator:** Identifies `Ubuntu 22.04.5 LTS`, `AMD EPYC-Rome Processor`, `gcc 11.4.0`, and resolves all tools in `/usr/bin/...`.

## Inspect all three compute nodes

Still on `headnode`, verify `compute001`, `compute002`, and `compute003`:

```bash
for node in compute001 compute002 compute003; do
    echo ""
    echo "=========================================="
    echo "CHECKING $node"
    echo "=========================================="

    ssh "$node" '
        hostname
        lscpu | grep -E "Architecture|Model name|Socket|Core|Thread|NUMA"
        free -h
        df -h "$HOME"
    '
done
```

**What it does:** Confirms over SSH that all three compute nodes are online, expose 8 AMD EPYC-Rome vCPUs (`0-7`), report their physical RAM (`free -h`), and mount `headnode:/home/ubuntu`.  
**Expected output:**
> ✅ **Success Indicator:** All three compute nodes report `8` vCPUs, available RAM, and the shared `/home/ubuntu` mount.

---

## 2. Install prerequisites BEFORE staging workspace libraries

### 2.1 Install compilers, OpenMPI, Netlib Reference BLAS & OpenBLAS

```bash
sudo apt update
sudo apt install -y \
  build-essential gcc g++ gfortran make cmake git wget curl pkg-config \
  openmpi-bin libopenmpi-dev \
  libblas-dev liblapack-dev \
  libopenblas-dev \
  numactl hwloc python3
```

**What it does:** Installs OpenMPI (`mpicc`), Netlib Reference BLAS (`libblas-dev`, used for this baseline build), and OpenBLAS (`libopenblas-dev`, used for the optimised build).  
**Expected output:** `apt` completes with zero errors.

### 2.2 Verify installed BLAS libraries and MPI wrapper

```bash
echo "========== PACKAGE CHECK =========="
for pkg in build-essential openmpi-bin libopenmpi-dev libblas-dev libopenblas-dev; do
    printf "%-18s: " "$pkg"
    dpkg-query -W -f='${Status} ${Version}\n' "$pkg" 2>/dev/null || echo "Not installed"
done

ls -l /usr/lib/x86_64-linux-gnu/blas/libblas.so* /usr/lib/x86_64-linux-gnu/libblas.so* 2>/dev/null || true
mpicc --showme
```

**What it does:** Confirms that `libblas-dev` is installed and prints the `mpicc` compiler wrapper flags.  
**Expected output:**
> ✅ **Success Output:** All packages show `install ok installed` and `libblas.so` is listed under `/usr/lib/x86_64-linux-gnu/blas/`.

---

## 3. Prepare the Workspace and stage Netlib Reference BLAS for NFS compute nodes

```text
/home/ubuntu/hpl/
├── source/
│   └── hpl-2.3/                   # Clean extracted HPL v2.3 source tree
├── runtime/
│   └── netlib-blas/
│       └── lib/                   # Staged reference Netlib libblas.so* (shared via NFS)
├── builds/
│   └── baseline-netlib-o2/        # Isolated Baseline (-O2, Netlib BLAS) build
│       └── bin/
│           └── xhpl               # Baseline Golden Reference HPL executable
├── runs/
│   └── baseline-netlib-o2/        # Baseline execution directory (HPL.dat & HPL.out)
└── logs/                          # Configure, build, ldd, and provenance logs
```

### 3.1 Set the main paths and stage Netlib Reference BLAS

```bash
export ROOT="$HOME/hpl"
export SRC_DIR="$ROOT/source/hpl-2.3"
export BUILDS_DIR="$ROOT/builds"
export BUILD_DIR="$BUILDS_DIR/baseline-netlib-o2"
export RUNS_DIR="$ROOT/runs"
export LOG_DIR="$ROOT/logs"
export NETLIB_BLAS_DIR="$ROOT/runtime/netlib-blas/lib"
export BUILD_JOBS="$(nproc)"

mkdir -p "$ROOT/source" "$BUILD_DIR/bin" "$RUNS_DIR/baseline-netlib-o2" \
         "$LOG_DIR" "$NETLIB_BLAS_DIR"

# Copy reference Netlib BLAS (NOT OpenBLAS) into $NETLIB_BLAS_DIR so compute001..003 load it over NFS
rm -rf "$NETLIB_BLAS_DIR/"*
cp -a /usr/lib/x86_64-linux-gnu/blas/libblas.so* "$NETLIB_BLAS_DIR/"
ln -sf "$NETLIB_BLAS_DIR/libblas.so.3" "$NETLIB_BLAS_DIR/libblas.so"
ls -la "$NETLIB_BLAS_DIR/"
```

**What it does:** Creates the `$HOME/hpl` directory structure and stages the unoptimized Netlib Reference `libblas.so.3` explicitly into `$HOME/hpl/runtime/netlib-blas/lib` (bypassing Ubuntu's `/etc/alternatives` symlink so the baseline build uses pure Netlib reference BLAS even when `libopenblas-dev` is also installed on the system!).  
**Expected output:**
> ✅ **Success Indicator:** Lists `libblas.so` and `libblas.so.3` inside `/home/ubuntu/hpl/runtime/netlib-blas/lib/`.

---

## 4. Download and verify HPL v2.3 source

```bash
cd "$ROOT/source"
if [ ! -f "hpl-2.3.tar.gz" ]; then
    wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
fi
rm -rf "$SRC_DIR"
tar -xzf hpl-2.3.tar.gz
sha256sum hpl-2.3.tar.gz
test -x "$SRC_DIR/configure" && echo "HPL v2.3 source tree verified"
```

**What it does:** Downloads the official `hpl-2.3.tar.gz` archive from Netlib, extracts it to `$HOME/hpl/source/hpl-2.3`, prints its SHA256 checksum, and confirms `./configure` exists.  
**Expected output:**
> ✅ **Success Output:** Prints the SHA256 checksum (`32c5c17d22330e6f2337b681be2163...`) and `HPL v2.3 source tree verified`.

---

## 5. Configure and compile Build 1 — `baseline-netlib-o2`

### 5.1 Configure `baseline-netlib-o2`

```bash
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR/bin"
cd "$BUILD_DIR"

set -o pipefail
CC=mpicc MPICC=mpicc \
CFLAGS="-O2 -g -fno-fast-math -fomit-frame-pointer" \
LDFLAGS="-L$NETLIB_BLAS_DIR -Wl,-rpath,$NETLIB_BLAS_DIR" \
LIBS="-lblas -lm" \
"$SRC_DIR/configure" \
  --prefix="$BUILD_DIR" \
  2>&1 | tee "$LOG_DIR/configure-baseline-netlib-o2.log"
```

**What it does:** Configures HPL v2.3 out-of-source in `$HOME/hpl/builds/baseline-netlib-o2` with `-O2 -g -fno-fast-math` and embeds `-Wl,-rpath,$NETLIB_BLAS_DIR` so `xhpl` always loads reference Netlib `libblas.so.3` across all nodes.  
**Expected output:**
> ✅ **Success Output:** Ends with `config.status: creating Makefile` and `config.status: creating testing/Makefile` with zero errors.

### 5.2 Compile and install `baseline-netlib-o2`

```bash
cd "$BUILD_DIR"
set -o pipefail
make -j "$BUILD_JOBS" 2>&1 | tee "$LOG_DIR/build-baseline-netlib-o2.log"
make install 2>&1 | tee -a "$LOG_DIR/build-baseline-netlib-o2.log"
ls -lh "$BUILD_DIR/bin/xhpl"
```

**What it does:** Compiles `xhpl` and installs the binary and default `HPL.dat` into `$HOME/hpl/builds/baseline-netlib-o2/bin/`.  
**Expected output:**
> ✅ **Success Output:** Lists `/home/ubuntu/hpl/builds/baseline-netlib-o2/bin/xhpl` with executable permissions.

---

## 6. Verify the Baseline executable and run a 4-rank smoke test

### 6.1 Verify shared-library linkage (`ldd`) on `headnode` and `compute001..003`

```bash
ldd "$BUILD_DIR/bin/xhpl" | grep -E "blas|mpi"
ldd "$BUILD_DIR/bin/xhpl" | grep 'not found' || echo "No missing shared libraries on headnode"

for node in compute001 compute002 compute003; do
  echo "===== Verifying baseline xhpl on $node ====="
  ssh "$node" "test -x $BUILD_DIR/bin/xhpl && ldd $BUILD_DIR/bin/xhpl | grep -E 'blas|not found'"
done
```

**What it does:** Confirms that `baseline-netlib-o2/bin/xhpl` resolves `libblas.so.3` directly to `/home/ubuntu/hpl/runtime/netlib-blas/lib/libblas.so.3` on both `headnode` and `compute001..003`.  
**Expected output:**
> ✅ **Success Output:** Every node resolves `libblas.so.3 => /home/ubuntu/hpl/runtime/netlib-blas/lib/libblas.so.3` with zero `not found` lines.

### 6.2 Run a quick 4-rank Baseline smoke test (`N = 2000`)

```bash
cd "$RUNS_DIR/baseline-netlib-o2"
cat > HPL.dat <<'HPL_SMOKE_EOF'
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
2000         Ns
1            # of NBs
128          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
2            Ps
2            Qs
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
HPL_SMOKE_EOF

mpirun -np 4 "$BUILD_DIR/bin/xhpl" | tee "$LOG_DIR/smoke-baseline-netlib-o2.log"
```

**What it does:** Executes a fast 4-rank smoke test (`N = 2000`, `P = 2, Q = 2`) using the baseline binary and verifies the residual calculation.  
**Expected output:**
> ✅ **Success Output:** Prints the GFLOP/s result row (`WR11C2R4 2000 128 2 2 ...`) followed by `||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= ... PASSED` and `1 tests completed and passed residual checks`.

---

## 7. Preserve the Baseline compilation evidence

```bash
{
  echo "===== HPL v2.3 Baseline Provenance ====="
  sha256sum "$ROOT/source/hpl-2.3.tar.gz"
  echo
  echo "===== Operating System & CPU ====="
  cat /etc/os-release | grep PRETTY_NAME
  lscpu | grep -E 'Architecture|Model name|Socket|Core|Thread|NUMA'
  echo
  echo "===== Compiler & MPI ====="
  gcc --version | head -n 1
  mpicc --showme
  echo
  echo "===== Baseline xhpl SHA256 & ldd ====="
  sha256sum "$BUILD_DIR/bin/xhpl"
  ldd "$BUILD_DIR/bin/xhpl"
} > "$LOG_DIR/compilation-environment-baseline.txt"

cat "$LOG_DIR/compilation-environment-baseline.txt"
```

**What it does:** Saves the source checksum, OS, CPU topology, compiler/MPI wrapper flags, and `xhpl` SHA256 checksum to `~/hpl/logs/compilation-environment-baseline.txt`.  
**Expected output:**
> ✅ **Success Indicator:** Prints the complete baseline provenance record.
