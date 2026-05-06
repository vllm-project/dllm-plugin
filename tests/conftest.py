# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for dllm-plugin tests."""

from __future__ import annotations


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark tests (can be slow)",
    )
