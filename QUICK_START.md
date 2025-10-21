# FDQC v4.0 - Quick Start Guide

## 5-Minute Getting Started

### 1. Install Dependencies

```bash
pip install numpy matplotlib scipy
```

### 2. Create Your First AI

```python
from fdqc_ai import FDQC_AI

# Create AI
ai = FDQC_AI(name="MyFirstAI", verbose=True)

# It's alive!
print("✓ AI System Ready")
```

### 3. Basic Interactions

```python
# Think about something
ai.think("I am learning about the world")

# Provide feedback (reward and success rate)
ai.learn(reward=0.8, success=0.9)

# Make a decision
action = ai.decide("What should I do next?")
print(f"Decision: {action}")

# Check internal state
ai.introspect()
```

### 4. Run Demo

```bash
python fdqc_ai.py
```

---

## Common Use Cases

### Use Case 1: Learning from Experience

```python
from fdqc_ai import FDQC_AI
import numpy as np

ai = FDQC_AI(name="Learner", verbose=False)

# Simulate learning experience
for episode in range(100):
    # Process experience
    ai.think(f"experience {episode}")
    
    # Provide feedback (success improves over time)
    success = 0.5 + 0.5 * (episode / 100)
    reward = success - 0.2  # Small energy cost
    
    ai.learn(reward=reward, success=success)
    
    if episode % 20 == 0:
        print(f"Episode {episode}: Success={success:.2f}, Reward={reward:.2f}")

# Check what was learned
stats = ai.get_statistics()
print(f"\nMean reward: {stats['mean_reward']:.2f}")
print(f"Crisis rate: {stats['crisis_rate']:.1%}")
```

### Use Case 2: Visual Perception

```python
ai = FDQC_AI(name="Vision", verbose=False)

# Process images
for i in range(10):
    # Generate random image (replace with real images)
    image = np.random.randn(28, 28, 3)
    
    # Perceive image
    result = ai.perceive_image(image, label=f"image_{i}")
    
    # Provide feedback
    ai.learn(reward=np.random.uniform(0, 1))
    
    print(f"Image {i}: "
          f"Salience={result['attention']:.2f}, "
          f"Novelty={result['novelty']:.2f}")
```

### Use Case 3: Memory & Recall

```python
ai = FDQC_AI(name="Memory", verbose=False)

# Create memorable experiences
experiences = [
    ("exciting discovery", 0.9, 0.95),
    ("boring routine", 0.2, 0.5),
    ("challenging problem", 0.7, 0.8),
    ("pleasant surprise", 0.85, 0.9),
    ("frustrating failure", -0.5, 0.3)
]

# Encode experiences
for text, reward, success in experiences:
    ai.think(text)
    ai.learn(reward=reward, success=success)

# Retrieve memories
print("\nRecalling 'exciting' experiences:")
memories = ai.remember("exciting", k=3)

print("\nRecalling 'failure' experiences:")
memories = ai.remember("failure", k=3)
```

### Use Case 4: Decision Making

```python
ai = FDQC_AI(name="Decider", verbose=False)

# Decision-making scenarios
scenarios = [
    "safe but boring option",
    "risky but rewarding opportunity",
    "familiar comfortable choice",
    "unknown challenging path"
]

for scenario in scenarios:
    print(f"\nScenario: {scenario}")
    action = ai.decide(scenario)
    print(f"Decision: {action}")
    
    # Simulate outcome
    if "risky" in scenario:
        outcome = np.random.choice([0.1, 0.9])  # High variance
    else:
        outcome = np.random.uniform(0.5, 0.7)    # Low variance
    
    ai.learn(reward=outcome)
```

### Use Case 5: Emotional Dynamics

```python
ai = FDQC_AI(name="Emotional", verbose=False)

# Emotional journey
events = [
    ("success", 0.9),
    ("another success", 0.85),
    ("unexpected failure", -0.6),
    ("recovery", 0.3),
    ("breakthrough", 0.95)
]

for event, reward in events:
    result = ai.think(event)
    print(f"{event:20s}: Valence={result['valence']:+.2f}, Arousal={result['arousal']:.2f}")
    ai.learn(reward=reward)
```

---

## Visualization

```python
from utils.visualization import visualize_all

# After running your AI for a while...
visualize_all(ai, output_dir="my_results")
```

This generates:
- `cognitive_timeline.png`: Time-series of all cognitive variables
- `memory_landscape.png`: Episodic memory structure
- `action_distribution.png`: Action selection patterns
- `system_report.txt`: Detailed text report

---

## Advanced: Direct Core Access

```python
from core.fdqc_core import FDQCCore

# Create core system
core = FDQCCore(n_global=60, n_wm_min=4, n_wm_max=15, f_c=10)

# Process single cognitive cycle
result = core.process_cycle(
    stimulus=np.random.randn(28, 28),
    stimulus_modality='visual',
    goals={'relevance': 0.8, 'success': 0.9}
)

# Access all details
print(f"Capacity: {result['capacity']}")
print(f"Dopamine: {result['dopamine']:.3f}")
print(f"TD Error: {result['td_error']:.3f}")
print(f"Crisis: {result['crisis']}")
print(f"Conscious contents: {result['conscious_contents']}")
```

---

## Training with Adaptive Curriculum

```python
from fdqc_v4_train_CLEANED import FDQCv4Training

# Initialize trainer
trainer = FDQCv4Training()

# Run adaptive training (difficulty adjusts automatically)
trainer.train(n_episodes=200)

# Check results
stats = trainer.system.get_statistics()
print(f"Mean success rate: {stats.get('mean_reward', 0):.2f}")

# Save results
trainer.save_results('my_training.npz')
```

---

## Introspection & Metacognition

```python
ai = FDQC_AI(name="Conscious", verbose=True)

# Do some processing
for i in range(50):
    ai.think(f"thought {i}")
    ai.learn(reward=np.random.uniform(0, 1))

# Introspect
state = ai.introspect()

# What is the AI aware of?
print("\nConscious of:")
for key in state.get('conscious_contents', {}).keys():
    print(f"  - {key}")
```

---

## Parameter Customization

```python
from core.fdqc_core import FDQCCore

# Custom configuration
core = FDQCCore(
    n_global=100,     # Larger workspace (more complex representations)
    n_wm_min=6,       # Higher baseline capacity
    n_wm_max=20,      # Higher crisis capacity
    f_c=15            # Faster processing (15 Hz instead of 10 Hz)
)

# Use as normal
result = core.process_cycle(stimulus, goals={'success': 0.8})
```

---

## Common Patterns

### Pattern 1: Perception → Learning Loop

```python
for stimulus in stimulus_stream:
    result = ai.perceive_image(stimulus)
    reward = evaluate_performance(result)
    ai.learn(reward=reward)
```

### Pattern 2: Decision → Feedback Loop

```python
for situation in situations:
    action = ai.decide(situation)
    outcome = execute_action(action)
    ai.learn(reward=outcome)
```

### Pattern 3: Exploration → Exploitation

```python
# Early: High exploration
for i in range(50):
    ai.think(f"explore {i}")
    ai.learn(reward=np.random.uniform(0, 1))

# Later: Exploit learned knowledge
for i in range(50):
    action = ai.decide()  # Uses learned values
    ai.learn(reward=high_reward)
```

---

## Troubleshooting

### Problem: No novelty detected

**Solution**: System has seen similar inputs. Try more diverse stimuli.

```python
# Bad: All same
for i in range(100):
    ai.think("same thing")

# Good: Diverse
for i in range(100):
    ai.think(f"experience {i} with variation {np.random.randint(10)}")
```

### Problem: Crisis mode never triggers

**Solution**: Need consistent baseline then large deviation.

```python
# Build baseline (30+ samples needed)
for i in range(50):
    ai.think("normal")
    ai.learn(reward=0.5, success=0.7)

# Then introduce crisis
ai.think("catastrophic failure!")
ai.learn(reward=-1.0, success=0.0)
```

### Problem: Memory retrieval returns nothing

**Solution**: Need to encode memories with consolidation.

```python
# Ensure high importance for consolidation
ai.think("important experience")
ai.learn(reward=0.9, success=0.95)  # High values → consolidation

# Then retrieve
memories = ai.remember("important", k=5)
```

---

## Next Steps

1. **Read ARCHITECTURE.md** for detailed system design
2. **Run demos/full_demo.py** for comprehensive examples
3. **Explore core/fdqc_core.py** to understand internals
4. **Customize** for your specific use case

---

## Quick Reference

### Main API Methods

| Method | Purpose |
|--------|---------|
| `ai.think(text)` | Process text input |
| `ai.perceive_image(image)` | Process visual input |
| `ai.decide(situation)` | Make decision |
| `ai.learn(reward, success)` | Update from feedback |
| `ai.remember(query, k)` | Retrieve memories |
| `ai.introspect()` | Report internal state |
| `ai.get_statistics()` | Get system statistics |
| `ai.reset()` | Reset to initial state |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| `ai.core` | Core cognitive system |
| `ai.core.perception` | Perception system |
| `ai.core.memory` | Memory system |
| `ai.core.attention` | Attention system |
| `ai.core.affect` | Affective system |
| `ai.core.motor` | Motor system |
| `ai.core.learning` | Learning system |
| `ai.core.global_workspace` | Consciousness |

---

## That's It!

You now have a **complete, functional AI system** with:

✅ Perception  
✅ Memory  
✅ Attention  
✅ Emotion  
✅ Decision-making  
✅ Learning  
✅ Consciousness  

**All biologically grounded. Ready to use.**

Happy coding! 🧠🚀

