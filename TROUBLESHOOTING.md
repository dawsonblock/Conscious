# FDQC v4.0 - Troubleshooting Guide

Quick solutions to common issues.

---

## Installation Issues

### Problem: ImportError: No module named 'numpy'

**Solution:**
```bash
pip install numpy scipy matplotlib
```

Or with user flag:
```bash
pip install --user numpy scipy matplotlib
```

### Problem: pip not found

**Solution:**
```bash
# Install pip
sudo apt-get install python3-pip

# Or use python -m pip
python3 -m pip install numpy scipy matplotlib
```

---

## Import Issues

### Problem: ModuleNotFoundError: No module named 'fdqc_ai'

**Solution:** Add workspace to Python path:
```bash
export PYTHONPATH="/path/to/workspace:$PYTHONPATH"
python3 your_script.py
```

Or in Python:
```python
import sys
sys.path.insert(0, '/path/to/workspace')
from fdqc_ai import FDQC_AI
```

### Problem: No module named 'core.fdqc_core'

**Solution:** Make sure `core/__init__.py` exists:
```bash
touch core/__init__.py
```

---

## Runtime Issues

### Problem: No novelty detected

**Cause:** System has seen similar inputs repeatedly.

**Solution:** Provide more diverse stimuli:
```python
# Bad: All the same
for i in range(100):
    ai.think("same thing")

# Good: Diverse inputs
for i in range(100):
    ai.think(f"experience {i} - {np.random.randint(10)}")
```

### Problem: Crisis mode never triggers

**Cause:** Need baseline + outlier (5-sigma).

**Solution:** Build baseline first:
```python
# Build baseline (30+ samples)
for i in range(50):
    ai.think("normal")
    ai.learn(reward=0.5, success=0.7)

# Then introduce crisis
ai.think("catastrophic failure!")
ai.learn(reward=-1.0, success=0.0)
```

### Problem: Memory retrieval returns nothing

**Cause:** No memories consolidated yet.

**Solution:** Ensure high importance scores:
```python
ai.think("important experience")
ai.learn(reward=0.9, success=0.95)  # High = consolidated

memories = ai.remember("important", k=5)
```

---

## Visualization Issues

### Problem: matplotlib backend error

**Solution 1:** Use non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
import matplotlib.pyplot as plt
```

**Solution 2:** Install GUI backend:
```bash
sudo apt-get install python3-tk
```

**Solution 3:** Save instead of show:
```python
visualize_all(ai, output_dir="results")  # Saves files
```

### Problem: No display available

**Solution:** Save plots to files:
```python
from utils.visualization import visualize_all
visualize_all(ai, output_dir="my_plots")
# Files saved in my_plots/ directory
```

---

## Performance Issues

### Problem: Processing very slow

**Possible causes and solutions:**

1. **Too many memories:**
   ```python
   # Limit memory size
   ai.core.memory.episodic_memory = deque(maxlen=1000)
   ```

2. **Large input dimensions:**
   ```python
   # Resize images before processing
   image = cv2.resize(image, (28, 28))
   ai.perceive_image(image)
   ```

3. **Too frequent consolidation:**
   ```python
   # Increase consolidation threshold
   ai.core.memory.consolidation.percentile = 90  # Top 10%
   ```

---

## Testing Issues

### Problem: Timeout in tests

**Solution:** Increase timeout:
```bash
timeout 60 python3 demos/full_demo.py basic
```

Or disable timeout:
```bash
python3 demos/full_demo.py basic
```

### Problem: Tests fail randomly

**Cause:** Randomness in neural encoding.

**Solution:** Set random seed:
```python
import numpy as np
np.random.seed(42)

ai = FDQC_AI()
# Now behavior is deterministic
```

---

## Environment Issues

### Problem: Different behavior on different machines

**Check:**
```bash
python3 --version          # Should be 3.8+
pip list | grep numpy      # Check versions
python3 .python-version-check.py
```

**Solution:** Match package versions:
```bash
pip install numpy==1.20.0 scipy==1.7.0 matplotlib==3.3.0
```

---

## Build Issues

### Problem: build_test.py fails

**Run individual checks:**
```bash
# Check dependencies
python3 -c "import numpy, scipy, matplotlib; print('OK')"

# Check imports
python3 -c "from fdqc_ai import FDQC_AI; print('OK')"

# Check syntax
python3 -m py_compile fdqc_ai.py
```

---

## Common Error Messages

### "ZeroDivisionError in compute_salience"

**Solution:** Check that inputs are not all zeros:
```python
# Add small epsilon
stimulus = stimulus + 1e-8
```

### "IndexError: list index out of range"

**Cause:** Empty working memory.

**Solution:** Process some inputs first:
```python
ai.think("initialize")  # Populate working memory
action = ai.decide()    # Now OK
```

### "KeyError in value_function"

**Cause:** State hasn't been seen before.

**Solution:** This is normal - Q-learning initializes to 0.

---

## Getting Help

### Debug Mode

Enable verbose output:
```python
ai = FDQC_AI(verbose=True)  # Shows detailed info
```

### Check System State

```python
ai.introspect()  # Print internal state
stats = ai.get_statistics()  # Get statistics
print(stats)
```

### Run Diagnostics

```bash
# Full diagnostic
python3 build_test.py --full

# Quick check
./run_tests.sh
```

---

## Still Having Issues?

1. **Check documentation:**
   - `README.md` - Main guide
   - `QUICK_START.md` - Getting started
   - `ARCHITECTURE.md` - Technical details

2. **Run build test:**
   ```bash
   python3 build_test.py --full
   ```

3. **Check versions:**
   ```bash
   python3 .python-version-check.py
   ```

4. **Test minimal example:**
   ```bash
   python3 examples/simple_example.py
   ```

---

**Last Updated:** January 2025  
**Version:** 4.0.0
