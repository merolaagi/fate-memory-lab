# Changelog

## 0.2.0
- Multiple seeds per arm, with mean and standard deviation in every table column
- Steps-to-settle metric: median time for a random starting state to come to rest
- Stop a running or queued run; finished parts are kept
- Download any run as CSV or JSON
- Health endpoint at /api/health
- Default port 47431
- update.sh installs a newer package from ~/Downloads while keeping runs, .venv and logs
- Tests and GitHub Actions workflow; MIT license

## 0.1.0
- First web bench: flipflop, xor and parity tasks; four arms; plate view of attractors; weight-drift stress test
