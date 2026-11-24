# Ansible user management

Automates creation of standard cluster accounts and passwordless `sudo` access across the head node and compute nodes, so you don't SSH into every box by hand every time someone joins the team.

Target environment: head node `192.168.0.1`, compute nodes on the same subnet, RHEL/Rocky-family OS (uses `dnf`).

## 1. Install prerequisites on the head node

Ansible only needs to run from the head node. It talks to the compute nodes over SSH, nothing extra gets installed on them for this step.

```bash
sudo dnf install epel-release -y
sudo crb enable
sudo dnf install ansible-core -y
```

Verify the install before doing anything else:

```bash
ansible --version
```

## 2. File placement

Keep the inventory and playbooks together under your home directory so they're easy to find and re-run later:

```
~/inventory
~/playbooks/create_sudo_users.yml
```

If multiple admins share this cluster, put this under version control (a private git repo, or at minimum `/opt/admin-playbooks` owned by the `wheel` group) rather than leaving it in one person's home directory. Otherwise the playbook disappears the day that account gets deleted.

## 3. Inventory

`~/inventory`:

```ini
[head]
192.168.0.1

[compute]
192.168.0.2
192.168.0.3

[cluster:children]
head
compute
```

Add new compute nodes here as they come online. `cluster:children` is just a convenience group so you can target everything at once with `-l cluster`.

## 4. The playbook

`~/playbooks/create_sudo_users.yml` creates three accounts, adds them to `wheel`, and grants the whole group passwordless `sudo`:

```yaml
---
- hosts: all
  become: true
  vars:
    users_to_create:
      - thabo
      - bosvark
      - khuphuka

  tasks:
    - name: Ensure sudo users are present on system
      ansible.builtin.user:
        name: "{{ item }}"
        state: present
        groups: wheel
        append: true
        create_home: true
      loop: "{{ users_to_create }}"

    - name: Configure passwordless sudo for wheel group
      ansible.builtin.copy:
        dest: /etc/sudoers.d/99-wheel-nopasswd
        content: "%wheel ALL=(ALL) NOPASSWD: ALL\n"
        owner: root
        group: root
        mode: '0440'
        validate: 'visudo -cf %s'
```

A couple of things worth knowing about this playbook rather than just running it blind:

- `append: true` on the user task means existing group memberships are kept, not overwritten. Drop this and a re-run could silently strip a user out of other groups.
- `validate: 'visudo -cf %s'` runs syntax validation on the sudoers file *before* it's written. If the content were malformed, the task fails and nothing gets copied, so you can't lock yourself out with a broken sudoers file this way. Don't remove this line.
- Mode `0440` (read-only, owner and group) is the correct permission for anything under `/etc/sudoers.d/`. Anything more permissive and sudo will refuse to read the file.

## 5. Running it

```bash
ansible-playbook -i inventory ~/playbooks/create_sudo_users.yml -K
```

`-K` prompts for the become (sudo) password on the machine you're running Ansible from.

Before running against the real cluster, do a dry run:

```bash
ansible-playbook -i inventory ~/playbooks/create_sudo_users.yml -K --check --diff
```

`--check` simulates the run without changing anything, and `--diff` shows you what would change. Worth doing every time you edit the playbook, not just the first time.

## 6. Security notes

- Passwordless `sudo` for the entire `wheel` group is convenient but broad. Anyone added to `wheel` (by this playbook or by hand later) gets full root, no password prompt, no audit friction. If that's more access than the team actually needs day to day, consider scoping `sudoers` entries to specific commands instead of `ALL`.
- These accounts have no password set by this playbook, which means login is SSH-key-only unless you set one separately. Confirm SSH key distribution is handled before you rely on this for onboarding.
- `/etc/sudoers.d/99-wheel-nopasswd` is a cluster-wide file. A mistake here affects every node in the inventory, not just one machine, so run `--check` first and read the diff.

## 7. Rollback

If something goes wrong after applying this:

**Remove passwordless sudo, keep the accounts:**

```bash
ansible all -i inventory -b -m ansible.builtin.file -a "path=/etc/sudoers.d/99-wheel-nopasswd state=absent"
```

**Remove a specific user entirely (run per host, or loop it):**

```bash
ansible all -i inventory -b -m ansible.builtin.user -a "name=khuphuka state=absent remove=yes"
```

`remove=yes` also deletes the home directory. Leave it off if you just want the account gone but the files kept.

**Full revert:** since this playbook is idempotent, the safest rollback for anything more than a single mistake is usually to fix the playbook and re-run it, rather than hand-editing nodes. Keep the previous version of `create_sudo_users.yml` in git so you always have something to diff against or revert to.

## 8. Verification checklist

After running, confirm on at least one compute node:

```bash
ssh 192.168.0.2 "id thabo"
ssh 192.168.0.2 "sudo -l -U thabo"
```

The first confirms the account and group membership exist. The second lists what that user is permitted to run under sudo, which should show `(ALL) NOPASSWD: ALL` if the sudoers change took.
