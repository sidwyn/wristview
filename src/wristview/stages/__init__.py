"""Pipeline stages.

Each module exposes one `run(ctx)` function that reads files from the run
directory and writes files back. No stage imports or calls another stage. The
driver in `wristview.cli` decides the order.
"""
