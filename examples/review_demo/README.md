# Code review demo fixtures

This directory contains intentionally unsafe code for a code review demonstration.
Do not merge or deploy this pull request. The module is not imported by the
application, exposed through an endpoint, or included in the installed package.

The sample models support-user input flowing into diagnostic, build-search,
and log-reading helpers, together with build-summary utilities. Existing tests
cover ordinary inputs. The review agent should identify security problems,
correctness bugs, and missing regression tests in the pull request diff.

Run the happy-path tests locally:

```bash
python3 -m unittest discover -s examples/review_demo -p 'test_*.py' -v
```

Do not invoke the diagnostic helper with real user input. Tests use temporary
files and an in-memory database and do not run system diagnostics.
