"""Swappable implementations behind each stage.

Every backend here answers the same question: is the preferred implementation
installable on this machine, and if not, what runs instead. Each one records
the answer so `meta.json` says which code actually produced the artifact.
"""
