#!/usr/bin/env python
"""
FDQC v4.0 - Simple Example

The simplest possible usage of the FDQC AI system.
Copy and run this to get started!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdqc_ai import FDQC_AI

# Create AI
print("Creating AI...")
ai = FDQC_AI(name="SimpleAI", verbose=True)

# Process some thoughts
print("\n1. Processing thoughts...")
ai.think("I am learning about the world")
ai.learn(reward=0.7, success=0.8)

ai.think("This is interesting")
ai.learn(reward=0.8, success=0.9)

ai.think("I wonder what will happen next")
ai.learn(reward=0.6, success=0.75)

# Make a decision
print("\n2. Making a decision...")
action = ai.decide("What should I do?")
print(f"Decision: {action}")

# Remember something
print("\n3. Retrieving memories...")
memories = ai.remember("interesting", k=3)

# Check internal state
print("\n4. Introspecting...")
state = ai.introspect()

# Get statistics
print("\n5. Getting statistics...")
stats = ai.get_statistics()

print("\n✓ Simple example complete!")

