# System-wide Lmod setup

Installs Lmod to `/opt/lmod` on the head node so every user gets it automatically, and configures a hierarchical module tree so compiler and MPI combinations that were never actually built together can't be loaded together by accident.

## What Lmod actually does

Environment modules exist because HPC nodes usually have several versions of the same software stack installed side by side (GCC 13, 15, two versions of OpenMPI, a couple of Python builds), and a user's shell can only sanely point at one of each at a time. A module file is just a script that edits environment variables like `PATH`, `LD_LIBRARY_PATH`, and `MODULEPATH` when you load it, and undoes those edits when you unload it.

Lmod is a specific implementation of that idea, written in Lua, and the two things it adds over older tools (like the original Tcl-based Environment Modules) are:

- **Hierarchical visibility.** Modules can be hidden until a prerequisite is loaded. This is the mechanism this setup relies on.
- **Dependency tracking.** Lmod knows which modules were loaded to satisfy which other modules, so unloading a compiler will also warn about or unload anything built against it.

## Why hierarchy, specifically

Flat module systems let you `module load gcc/15.2` and `module load openmpi/5.0` independently, with nothing stopping you from loading an OpenMPI build that was actually compiled against a completely different compiler. That combination will often still load without error and then fail, or worse, run with subtly wrong behavior, somewhere deep in a parallel job.

A hierarchical layout avoids this by physically not listing the MPI modules until a compiler is loaded. The mechanism is `MODULEPATH`: Lmod only looks in directories currently listed in that variable, and a compiler's module file adds a new directory to `MODULEPATH` as one of its side effects. Load a different compiler and that path gets swapped for a different one. The user never sees an MPI option that wasn't actually built for their currently loaded compiler, because that option isn't on the search path at all.

## 1. Dependencies

```bash
sudo dnf install git gcc make tcl-devel tcl tcllib bc -y
sudo dnf install lua lua-posix lua-term lua-filesystem lua-devel -y
sudo dnf group install "Development Tools" -y
```

Lmod is written in Lua but its module files can be either Lua (`.lua`) or Tcl (`.tcl`), which is why both language toolchains are pulled in here even though this setup only uses Lua-format modules going forward.

## 2. Build and install

```bash
mkdir -p ~/downloads && cd ~/downloads
git clone https://www.github.com/TACC/Lmod.git
cd Lmod
./configure --prefix=/opt/lmod
make -j$(nproc)
sudo make install
```

`sudo` is only needed for the final `make install` step, since `/opt/lmod` isn't writable by a regular user. The build itself doesn't need root.

This installed Lmod 9.4.2 to `/opt/lmod/lmod/9.4.2`. Worth confirming the version landed where you expect, since a version-specific path here gets hardcoded into the profile script in the next step:

```bash
ls /opt/lmod/lmod/
```

## 3. System-wide profile script

`/etc/profile.d/lmod.sh` runs for every login shell on the system, which is what makes this "system-wide" rather than a per-user setup each person has to opt into:

```bash
# Lmod System-Wide Initialization
export PATH=/opt/lmod/lmod/lmod/libexec:$PATH
source /opt/lmod/lmod/lmod/init/bash
module use --append /opt/lmod/lmod/modulefiles
module use --append /opt/modulefiles/core
module unuse /opt/modulefiles # Remove parent directory to avoid duplicate core/ paths
```

The `module unuse` / `module use` pair at the bottom matters more than it looks. Without it, both `/opt/modulefiles` and `/opt/modulefiles/core` end up on `MODULEPATH`, and since `core` sits inside `/opt/modulefiles`, Lmod would show top-level modules twice under slightly different listings. Restricting `MODULEPATH` to just `core` keeps `module avail` output clean, and the compiler-specific directories still get added correctly later since that's handled inside the compiler module files themselves, not here.

File placement: this file needs root ownership and should be readable by everyone (`644` is standard for `/etc/profile.d/` scripts). Anything in this directory runs on every login, so a syntax error here breaks shell startup for the whole cluster, not just Lmod. Test changes in a subshell before trusting them system-wide:

```bash
bash -x /etc/profile.d/lmod.sh
```

## 4. Directory layout

```
/opt/modulefiles/
  core/
    gcc/15.2.lua
    icx/2025.3.lua
  compiler/
    gcc/15.2/
      openmpi/5.0.lua
      mpich/5.0.lua
    icx/2025.3/
      intelmpi/2021.17.lua
```

`core` holds the modules visible with nothing else loaded, in this case just the three compilers. `compiler` holds everything that's only visible once a specific compiler is active, organized by compiler and version so different compiler versions don't accidentally share an MPI directory.

## 5. Compiler modules (level 1)

These are the entry point. Loading one adds the matching subdirectory of `compiler/` to `MODULEPATH`, which is what makes the level-2 modules appear.

`/opt/modulefiles/core/gcc/15.2.lua`:

```lua
whatis("Name: gcc")
whatis("Version: 15.2")
whatis("Description: GNU Compiler Collection")
family("compiler")
prepend_path("MODULEPATH", "/opt/modulefiles/compiler/gcc/15.2")
```

`family("compiler")` is doing real work here, not just documentation. It tells Lmod that only one module in the `compiler` family can be loaded at a time, so loading `icx` while `gcc` is active automatically swaps it out rather than leaving both loaded and both sets of MPI options visible at once. That's what stops a user from ending up in an inconsistent state without needing to remember to unload anything themselves.

The `icx` and `nvc` module files follow the same pattern, each prepending its own compiler-specific path (`/opt/modulefiles/compiler/icx/2025.3` and `/opt/modulefiles/compiler/nvc/26.1` respectively).

## 6. MPI modules (level 2)

`/opt/modulefiles/compiler/gcc/15.2/openmpi/5.0.lua`:

```lua
whatis("Name: openmpi")
whatis("Version: 5.0")
whatis("Description: Open MPI built with GCC")
family("mpi")
depends_on("gcc/15.2")
prepend_path("MODULEPATH", "/opt/modulefiles/compiler/gcc/15.2/openmpi/5.0")
```

`depends_on("gcc/15.2")` is a second layer of protection on top of the directory-hiding trick. Even if a module file at this level somehow got loaded directly (say, a user hardcoded a full path), Lmod checks the dependency and will refuse, or auto-load the dependency, rather than silently allowing an MPI build to run without its matching compiler active. `family("mpi")` gives the same one-at-a-time behavior as the compiler family, so switching MPI implementations swaps cleanly instead of stacking.

## 7. Verification

```bash
ml purge
ml av          # should list only gcc, icx, nvc
ml load gcc/15.2
ml av          # openmpi and mpich now appear, nested under gcc/15.2
ml load openmpi/5.0
ml list        # confirms both gcc/15.2 and openmpi/5.0 are active
ml purge
```

If a module you expect isn't showing up, `module spider <name>` searches the full tree regardless of what's currently loaded and tells you what prerequisite would make it visible:

```bash
module spider openmpi
```

This is the actual troubleshooting tool for "why can't I see this module," since `module avail` only ever shows what's currently reachable through `MODULEPATH`.

## 8. Rollback and recovery

**Disable Lmod without removing it**, if it's causing login problems cluster-wide:

```bash
sudo mv /etc/profile.d/lmod.sh /etc/profile.d/lmod.sh.disabled
```

New shells stop loading Lmod immediately. Nothing else on the system is touched.

**Remove a single broken module file** without affecting the rest of the tree — deleting the file is enough, Lmod rescans `MODULEPATH` on demand rather than caching a fixed list:

```bash
sudo rm /opt/modulefiles/compiler/gcc/15.2/openmpi/5.0.lua
```

**Full uninstall:**

```bash
sudo rm -rf /opt/lmod
sudo rm /etc/profile.d/lmod.sh
```

Users already logged in will keep their current session's environment until they log out and back in, so this is safe to run without kicking anyone off first.

**Before making structural changes** to the module tree on a production head node, keep a copy:

```bash
sudo cp -r /opt/modulefiles /opt/modulefiles.bak-$(date +%Y%m%d)
```

Restoring is just swapping the directory back, and since Lmod reads the tree live, no service restart is needed either way.
