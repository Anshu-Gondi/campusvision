# Contributing to CampusVision

Thank you for your interest in CampusVision. Before contributing, please read
this document carefully — it explains how contributions work under a proprietary
license and what you can expect from the process.

---

## Important: This is Proprietary Software

CampusVision is **not open source**. It is licensed under a proprietary license
(see [LICENSE](LICENSE)) where all rights are reserved by the Owner (Anshu Gondi).

By submitting any contribution — a pull request, patch, issue, suggestion, or
code snippet — you agree that:

> **The Owner retains full and exclusive copyright over all contributions merged
> into this repository.** You do not acquire any ownership, co-authorship claim,
> or license rights to the Software as a result of contributing.

If you are not comfortable with this, please do not submit contributions.

---

## What Counts as a Contribution

- Bug reports and issue descriptions
- Feature suggestions or enhancement proposals
- Pull requests with code, tests, or documentation fixes
- Feedback on existing functionality
- Typo fixes or documentation improvements

---

## How to Contribute

### 1. Open an Issue First

Before writing any code, **open an issue** to discuss the problem or
feature you want to address. This avoids wasted effort and makes sure your
contribution aligns with the project's direction.

Use clear, descriptive titles. Include:
- What the problem is or what you'd like to see
- Steps to reproduce (for bugs)
- Your environment (OS, Python version, Rust version, Node version) if relevant

### 2. Fork and Branch

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/campus_vision_project.git
cd campus_vision_project

# Create a focused branch — one concern per branch
git checkout -b fix/qr-session-expiry
git checkout -b feat/export-attendance-csv
```

Branch naming conventions:

| Prefix | Use for |
|---|---|
| `fix/` | Bug fixes |
| `feat/` | New features |
| `docs/` | Documentation only |
| `refactor/` | Code cleanup without behavior change |
| `test/` | Adding or improving tests |
| `chore/` | Tooling, dependencies, config |

### 3. Make Your Changes

Follow the code style and patterns already present in the codebase.
Keep changes focused — one pull request should do one thing.

**Backend (Python/Django):**
- Follow PEP 8
- Write or update tests in the relevant `tests.py`
- Run `python manage.py test` before submitting

**Vision Engine (Rust):**
- Run `cargo fmt` and `cargo clippy` before submitting
- Add tests for any new logic
- Run `cargo test` — all tests must pass

**Frontend (React/Vite):**
- Keep components focused and reusable
- No `console.log` left in submitted code
- Run `pnpm lint` before submitting

**Mobile (Expo/React Native):**
- Test on both Android and iOS if possible
- Follow existing component and hook patterns

### 4. Commit Message Format

Use clear, imperative commit messages:

```
fix: resolve QR session expiry edge case on timezone change
feat: add CSV export for attendance reports
docs: clarify Rust vision engine setup in README
refactor: extract geo validation into shared utility
```

Avoid vague messages like `fix stuff`, `update`, or `changes`.

### 5. Open a Pull Request

Push your branch and open a pull request against `main`:

```bash
git push origin your-branch-name
```

In your PR description, include:
- **What** this PR does
- **Why** it's needed (link to the related issue)
- **How** to test it
- Any known limitations or follow-up work

PRs will be reviewed by the Owner. Feedback may be given before merging.
Please respond to review comments promptly.

---

## What Will Not Be Accepted

To keep the review process efficient, the following will be closed without merge:

- PRs with no associated issue or prior discussion
- PRs that significantly change the project's architecture without alignment
- Code that doesn't follow existing style conventions
- Contributions that include third-party code with incompatible licenses
- Anything that introduces security vulnerabilities

---

## Reporting Security Issues

**Do not open a public issue for security vulnerabilities.**

If you find a security issue, please report it privately by contacting the Owner
directly via GitHub: [https://github.com/Anshu-Gondi](https://github.com/Anshu-Gondi)

Provide a clear description of the vulnerability and steps to reproduce it.
You will receive a response as quickly as possible.

---

## Licensing Inquiries

If you want to use CampusVision in a commercial product, fork it for your own
project, or discuss any other licensing arrangement, please see the
[LICENSE](LICENSE) file and reach out via GitHub.

---

## Code of Conduct

Be professional and respectful in all interactions — issues, PRs, and discussions.
Harassment, spam, or bad-faith contributions will result in being blocked from
the repository.

---

*Thank you for taking the time to improve CampusVision.*
*— Anshu Gondi*