"""Framework adapters — import only the adapter you use."""

# Do NOT import adapters here. Users import directly:
#   from memshield.adapters.openai_provider import OpenAIProvider
#
# This avoids pulling in framework dependencies for users
# who only use one adapter.
