"""Opt-in conversation relationship memory.

PostgreSQL owns consent and lifecycle state. Neo4j implementations in this
package are projections only and are deliberately hidden behind a narrow,
parameterized interface.
"""

from .routes import router

__all__ = ["router"]
