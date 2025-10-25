#!/usr/bin/env python3
"""
FDQC v4.0 - Build and Test Script

Comprehensive build, test, and validation script for the FDQC AI system.

Usage:
    python build_test.py [--quick] [--full] [--visual] [--no-deps]

Options:
    --quick     Run quick tests only (skip visualizations)
    --full      Run full test suite including all demos
    --visual    Generate visualizations (requires matplotlib)
    --no-deps   Skip dependency check

Author: FDQC Research Team
Date: January 2025
"""

import sys
import os
import subprocess
import argparse
from typing import List, Tuple, Dict

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Tuple of (success, missing_packages)
    """
    print_header("Checking Dependencies")
    
    required_packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print_success(f"{package_name} found")
        except ImportError:
            print_error(f"{package_name} not found")
            missing.append(package_name)
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        print_info("Install with: pip install " + " ".join(missing))
        return False, missing
    
    print_success("All dependencies installed")
    return True, []

def check_file_structure() -> bool:
    """
    Check if all required files are present.
    
    Returns:
        True if all files present, False otherwise
    """
    print_header("Checking File Structure")
    
    required_files = [
        'fdqc_ai.py',
        'fdqc_v4_demo_compact_CLEANED.py',
        'core/fdqc_core.py',
        'core/__init__.py',
        'utils/visualization.py',
        'utils/__init__.py',
        'demos/full_demo.py',
        'examples/simple_example.py',
        'requirements.txt',
        'README.md',
        'ARCHITECTURE.md',
        'QUICK_START.md'
    ]
    
    all_present = True
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print_success(f"{filepath}")
        else:
            print_error(f"{filepath} not found")
            all_present = False
    
    if all_present:
        print_success("All required files present")
    else:
        print_error("Some files are missing")
    
    return all_present

def compile_check() -> bool:
    """
    Check if all Python files compile without errors.
    
    Returns:
        True if all files compile, False otherwise
    """
    print_header("Syntax Check")
    
    python_files = [
        'fdqc_ai.py',
        'fdqc_v4_demo_compact_CLEANED.py',
        'fdqc_v4_train_CLEANED.py',
        'core/fdqc_core.py',
        'utils/visualization.py',
        'demos/full_demo.py',
        'examples/simple_example.py'
    ]
    
    all_compile = True
    
    for filepath in python_files:
        if not os.path.exists(filepath):
            continue
            
        try:
            with open(filepath, 'r') as f:
                compile(f.read(), filepath, 'exec')
            print_success(f"{filepath} compiles successfully")
        except SyntaxError as e:
            print_error(f"{filepath} has syntax error: {e}")
            all_compile = False
    
    if all_compile:
        print_success("All files compile successfully")
    else:
        print_error("Some files have syntax errors")
    
    return all_compile

def test_imports() -> bool:
    """
    Test if all modules can be imported.
    
    Returns:
        True if all imports successful, False otherwise
    """
    print_header("Testing Imports")
    
    test_cases = [
        ('fdqc_ai', 'FDQC_AI'),
        ('core.fdqc_core', 'FDQCCore'),
        ('utils.visualization', 'visualize_all'),
        ('fdqc_v4_demo_compact_CLEANED', 'FDQCv4System')
    ]
    
    all_import = True
    
    for module_name, class_name in test_cases:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print_success(f"from {module_name} import {class_name}")
        except Exception as e:
            print_error(f"Failed to import {class_name} from {module_name}: {e}")
            all_import = False
    
    if all_import:
        print_success("All imports successful")
    else:
        print_error("Some imports failed")
    
    return all_import

def run_basic_tests() -> bool:
    """
    Run basic functionality tests.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print_header("Running Basic Tests")
    
    try:
        # Test 1: Create AI instance
        print_info("Test 1: Creating AI instance...")
        from fdqc_ai import FDQC_AI
        ai = FDQC_AI(name="TestAI", verbose=False)
        print_success("AI instance created")
        
        # Test 2: Process input
        print_info("Test 2: Processing input...")
        result = ai.think("test input")
        assert 'action' in result, "Missing 'action' in result"
        assert 'valence' in result, "Missing 'valence' in result"
        print_success("Input processing works")
        
        # Test 3: Learning
        print_info("Test 3: Testing learning...")
        ai.learn(reward=0.5, success=0.7)
        print_success("Learning works")
        
        # Test 4: Decision making
        print_info("Test 4: Testing decision making...")
        action = ai.decide()
        assert action in ['wait', 'explore', 'approach', 'avoid'], f"Invalid action: {action}"
        print_success("Decision making works")
        
        # Test 5: Memory
        print_info("Test 5: Testing memory...")
        memories = ai.remember("test", k=3)
        assert isinstance(memories, list), "Memory retrieval failed"
        print_success("Memory works")
        
        # Test 6: Introspection
        print_info("Test 6: Testing introspection...")
        state = ai.introspect()
        assert 'valence' in state, "Missing valence in state"
        print_success("Introspection works")
        
        # Test 7: Statistics
        print_info("Test 7: Testing statistics...")
        stats = ai.get_statistics()
        assert 'total_timesteps' in stats, "Missing timesteps in stats"
        print_success("Statistics work")
        
        print_success("All basic tests passed")
        return True
        
    except Exception as e:
        print_error(f"Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_demo(demo_name: str, timeout: int = 30) -> bool:
    """
    Run a specific demo.
    
    Args:
        demo_name: Name of the demo to run
        timeout: Timeout in seconds
    
    Returns:
        True if demo runs successfully, False otherwise
    """
    print_info(f"Running {demo_name} demo...")
    
    try:
        result = subprocess.run(
            ['python3', 'demos/full_demo.py', demo_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print_success(f"{demo_name} demo completed")
            return True
        else:
            print_error(f"{demo_name} demo failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print_error(f"{demo_name} demo timed out")
        return False
    except Exception as e:
        print_error(f"{demo_name} demo failed: {e}")
        return False

def run_all_demos() -> bool:
    """
    Run all demo scenarios.
    
    Returns:
        True if all demos pass, False otherwise
    """
    print_header("Running All Demos")
    
    demos = [
        'basic',
        'vision',
        'memory',
        'decision',
        'emotion'
    ]
    
    results = {}
    for demo in demos:
        results[demo] = run_demo(demo, timeout=30)
    
    # Summary
    print("\nDemo Results:")
    for demo, success in results.items():
        if success:
            print_success(f"{demo}: PASS")
        else:
            print_error(f"{demo}: FAIL")
    
    all_pass = all(results.values())
    if all_pass:
        print_success("All demos passed")
    else:
        print_error("Some demos failed")
    
    return all_pass

def generate_visualizations() -> bool:
    """
    Generate test visualizations.
    
    Returns:
        True if visualizations generated successfully
    """
    print_header("Generating Visualizations")
    
    try:
        from fdqc_ai import FDQC_AI
        from utils.visualization import visualize_all
        import numpy as np
        
        print_info("Creating test AI...")
        ai = FDQC_AI(name="VisualizationTest", verbose=False)
        
        # Generate some activity
        print_info("Generating activity...")
        for i in range(50):
            ai.think(f"test {i}")
            ai.learn(reward=np.random.uniform(-0.5, 1.0), success=np.random.uniform(0.5, 1.0))
        
        # Create visualizations
        print_info("Generating plots...")
        visualize_all(ai, output_dir="test_visualizations")
        
        print_success("Visualizations generated in test_visualizations/")
        return True
        
    except Exception as e:
        print_error(f"Visualization generation failed: {e}")
        return False

def print_summary(results: Dict[str, bool]):
    """Print test summary."""
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total tests: {total}")
    print_success(f"Passed: {passed}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    
    print("\nDetailed Results:")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        color = Colors.OKGREEN if success else Colors.FAIL
        print(f"{color}{status}{Colors.ENDC} - {test_name}")
    
    if all(results.values()):
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'ALL TESTS PASSED!':^70}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        return True
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.FAIL}{Colors.BOLD}{'SOME TESTS FAILED':^70}{Colors.ENDC}")
        print(f"{Colors.FAIL}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        return False

def main():
    """Main build and test entry point."""
    parser = argparse.ArgumentParser(description='FDQC v4.0 Build and Test Script')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--visual', action='store_true', help='Generate visualizations')
    parser.add_argument('--no-deps', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                  FDQC v4.0 BUILD & TEST SUITE                     ║")
    print("║               Complete AI System Validation                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    results = {}
    
    # Dependency check
    if not args.no_deps:
        deps_ok, missing = check_dependencies()
        results['Dependencies'] = deps_ok
        if not deps_ok:
            print_error("Cannot proceed without dependencies")
            return 1
    
    # File structure check
    results['File Structure'] = check_file_structure()
    
    # Syntax check
    results['Syntax Check'] = compile_check()
    
    # Import test
    results['Import Test'] = test_imports()
    
    # Basic functionality tests
    results['Basic Tests'] = run_basic_tests()
    
    # Full test suite
    if args.full or not args.quick:
        results['All Demos'] = run_all_demos()
    
    # Visualizations
    if args.visual:
        results['Visualizations'] = generate_visualizations()
    
    # Summary
    success = print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
