#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDQC v4.0 - Visualization Tools

Tools for visualizing the internal states and dynamics of the FDQC system.

Author: FDQC Research Team
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import json


def plot_cognitive_timeline(history: List[Dict], save_path: Optional[str] = None):
    """
    Plot timeline of cognitive states.
    
    Args:
        history: List of cycle results from FDQC system
        save_path: Optional path to save figure
    """
    if not history:
        print("No history to plot")
        return
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('FDQC Cognitive Timeline', fontsize=16, fontweight='bold')
    
    timesteps = [h['timestep'] for h in history]
    
    # 1. Working Memory Capacity
    capacity = [h['capacity'] for h in history]
    crisis = [h['crisis'] for h in history]
    
    axes[0].plot(timesteps, capacity, 'b-', linewidth=2, label='Capacity')
    axes[0].scatter([t for t, c in zip(timesteps, crisis) if c],
                   [cap for cap, c in zip(capacity, crisis) if c],
                   color='red', s=100, marker='*', label='Crisis', zorder=5)
    axes[0].set_ylabel('WM Capacity', fontsize=12)
    axes[0].set_title('Working Memory Dynamics')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Affect (Valence & Arousal)
    valence = [h['valence'] for h in history]
    arousal = [h['arousal'] for h in history]
    
    axes[1].plot(timesteps, valence, 'g-', linewidth=2, label='Valence')
    axes[1].plot(timesteps, arousal, 'orange', linewidth=2, label='Arousal')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Affect', fontsize=12)
    axes[1].set_title('Emotional Dynamics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Reward & Dopamine
    reward = [h['reward'] for h in history]
    dopamine = [h['dopamine'] for h in history]
    
    axes[2].plot(timesteps, reward, 'purple', linewidth=2, label='Reward')
    axes[2].plot(timesteps, dopamine, 'cyan', linewidth=2, label='Dopamine (RPE)')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Reward Signal', fontsize=12)
    axes[2].set_title('Reward & Dopamine Dynamics')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Learning
    learning_rate = [h['learning_rate'] for h in history]
    td_error = [h['td_error'] for h in history]
    
    ax4a = axes[3]
    ax4b = ax4a.twinx()
    
    ax4a.plot(timesteps, learning_rate, 'brown', linewidth=2, label='Learning Rate')
    ax4b.plot(timesteps, td_error, 'pink', linewidth=2, label='TD Error')
    
    ax4a.set_ylabel('Learning Rate', fontsize=12, color='brown')
    ax4b.set_ylabel('TD Error', fontsize=12, color='pink')
    ax4a.set_title('Learning Dynamics')
    ax4a.grid(True, alpha=0.3)
    
    # 5. Attention & Novelty
    salience = [h['salience'] for h in history]
    novelty = [h['novelty'] for h in history]
    
    axes[4].plot(timesteps, salience, 'm-', linewidth=2, label='Salience')
    axes[4].plot(timesteps, novelty, 'y-', linewidth=2, label='Novelty')
    axes[4].set_xlabel('Timestep', fontsize=12)
    axes[4].set_ylabel('Attention', fontsize=12)
    axes[4].set_title('Attention & Novelty')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_memory_landscape(memory_system, save_path: Optional[str] = None):
    """
    Visualize the structure of episodic memory.
    
    Args:
        memory_system: MemorySystem instance
        save_path: Optional path to save figure
    """
    if not memory_system.episodic_memory:
        print("No episodic memories to visualize")
        return
    
    # Extract memory features
    memories = list(memory_system.episodic_memory)
    n_memories = len(memories)
    
    valences = [m.valence for m in memories]
    arousals = [m.arousal for m in memories]
    timestamps = [m.timestamp for m in memories]
    retrieval_counts = [m.retrieval_count for m in memories]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Episodic Memory Landscape', fontsize=16, fontweight='bold')
    
    # 1. Valence-Arousal space
    scatter = axes[0, 0].scatter(valences, arousals, 
                                c=timestamps, cmap='viridis',
                                s=np.array(retrieval_counts)*20 + 20,
                                alpha=0.6, edgecolors='black')
    axes[0, 0].set_xlabel('Valence', fontsize=12)
    axes[0, 0].set_ylabel('Arousal', fontsize=12)
    axes[0, 0].set_title('Affective Memory Space')
    axes[0, 0].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Timestamp')
    
    # 2. Memory timeline
    axes[0, 1].scatter(timestamps, valences, c='green', alpha=0.6, label='Valence')
    axes[0, 1].scatter(timestamps, arousals, c='orange', alpha=0.6, label='Arousal')
    axes[0, 1].set_xlabel('Timestamp', fontsize=12)
    axes[0, 1].set_ylabel('Affective Value', fontsize=12)
    axes[0, 1].set_title('Memory Timeline')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Retrieval statistics
    axes[1, 0].hist(retrieval_counts, bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Retrieval Count', fontsize=12)
    axes[1, 0].set_ylabel('Number of Memories', fontsize=12)
    axes[1, 0].set_title('Memory Retrieval Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Affective distribution
    axes[1, 1].hist(valences, bins=20, alpha=0.5, label='Valence', color='green', edgecolor='black')
    axes[1, 1].hist(arousals, bins=20, alpha=0.5, label='Arousal', color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Value', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Affective Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_action_distribution(history: List[Dict], save_path: Optional[str] = None):
    """
    Visualize action selection patterns.
    
    Args:
        history: List of cycle results
        save_path: Optional path to save figure
    """
    if not history:
        print("No history to plot")
        return
    
    actions = [h['action'] for h in history]
    action_types = list(set(actions))
    action_counts = {a: actions.count(a) for a in action_types}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Action Selection Patterns', fontsize=16, fontweight='bold')
    
    # 1. Action distribution (pie chart)
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    axes[0].pie(action_counts.values(), labels=action_counts.keys(),
               autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Action Distribution')
    
    # 2. Action timeline
    action_to_num = {a: i for i, a in enumerate(action_types)}
    action_nums = [action_to_num[a] for a in actions]
    timesteps = [h['timestep'] for h in history]
    
    axes[1].scatter(timesteps, action_nums, c=action_nums, cmap='tab10', alpha=0.6)
    axes[1].set_xlabel('Timestep', fontsize=12)
    axes[1].set_ylabel('Action', fontsize=12)
    axes[1].set_yticks(range(len(action_types)))
    axes[1].set_yticklabels(action_types)
    axes[1].set_title('Action Timeline')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def generate_report(ai_system, save_path: str = "fdqc_report.txt"):
    """
    Generate comprehensive text report of system state.
    
    Args:
        ai_system: FDQC_AI instance
        save_path: Path to save report
    """
    stats = ai_system.core.get_statistics()
    
    report = []
    report.append("="*70)
    report.append("FDQC v4.0 AI SYSTEM REPORT")
    report.append("="*70)
    report.append("")
    
    report.append(f"System Name: {ai_system.name}")
    report.append(f"Timestamp: {ai_system.core.timestep}")
    report.append(f"Episodes: {ai_system.core.episode_count}")
    report.append("")
    
    report.append("-"*70)
    report.append("MEMORY SYSTEMS")
    report.append("-"*70)
    report.append(f"Working Memory Capacity: {ai_system.core.current_capacity} items")
    report.append(f"Episodic Memories: {len(ai_system.core.memory.episodic_memory)}")
    report.append(f"Semantic Memories: {len(ai_system.core.memory.semantic_memory)}")
    report.append(f"Mean Capacity: {stats.get('mean_capacity', 0):.2f} ± {stats.get('capacity_std', 0):.2f}")
    report.append("")
    
    report.append("-"*70)
    report.append("AFFECTIVE STATE")
    report.append("-"*70)
    report.append(f"Current Valence: {ai_system.core.affect.current_valence:+.2f}")
    report.append(f"Current Arousal: {ai_system.core.affect.current_arousal:.2f}")
    report.append(f"Energy Level: {ai_system.core.affect.energy_level:.2%}")
    report.append(f"Exploration Drive: {ai_system.core.affect.exploration_drive:.2%}")
    report.append(f"Mean Reward: {stats.get('mean_reward', 0):+.2f}")
    report.append(f"Mean Valence: {stats.get('mean_valence', 0):+.2f}")
    report.append(f"Mean Arousal: {stats.get('mean_arousal', 0):.2f}")
    report.append("")
    
    report.append("-"*70)
    report.append("LEARNING DYNAMICS")
    report.append("-"*70)
    report.append(f"Mean TD Error: {stats.get('mean_td_error', 0):+.2f}")
    report.append(f"Mean Dopamine: {stats.get('mean_dopamine', 0):+.2f}")
    report.append(f"Mean Learning Rate: {stats.get('mean_learning_rate', 0):.4f}")
    report.append(f"Value Function Size: {len(ai_system.core.learning.value_function)}")
    report.append("")
    
    report.append("-"*70)
    report.append("CRISIS DETECTION")
    report.append("-"*70)
    report.append(f"Crisis Count: {stats.get('crisis_count', 0)}")
    report.append(f"Crisis Rate: {stats.get('crisis_rate', 0):.1%}")
    report.append(f"Currently in Crisis: {'YES' if ai_system.core.current_capacity > ai_system.core.n_wm_min else 'NO'}")
    report.append("")
    
    report.append("-"*70)
    report.append("ENERGY CONSUMPTION")
    report.append("-"*70)
    report.append(f"Mean Energy Cost: {stats.get('mean_energy', 0):.2e} J/s")
    report.append(f"Total Energy Used: {stats.get('total_energy', 0):.2e} J")
    report.append("")
    
    report.append("-"*70)
    report.append("CONSCIOUSNESS")
    report.append("-"*70)
    if ai_system.core.global_workspace:
        report.append(f"Conscious Contents: {len(ai_system.core.global_workspace)} items")
        report.append(f"Contents: {list(ai_system.core.global_workspace.keys())}")
    else:
        report.append("No conscious contents yet")
    report.append("")
    
    report.append("="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    # Write to file
    report_text = "\n".join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to {save_path}")
    print()
    print(report_text)
    
    return report_text


def visualize_all(ai_system, output_dir: str = "visualizations"):
    """
    Generate all visualizations for an AI system.
    
    Args:
        ai_system: FDQC_AI instance
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}/...")
    
    # Cognitive timeline
    if ai_system.core.history:
        plot_cognitive_timeline(
            ai_system.core.history,
            save_path=f"{output_dir}/cognitive_timeline.png"
        )
    
    # Memory landscape
    if ai_system.core.memory.episodic_memory:
        plot_memory_landscape(
            ai_system.core.memory,
            save_path=f"{output_dir}/memory_landscape.png"
        )
    
    # Action distribution
    if ai_system.core.history:
        plot_action_distribution(
            ai_system.core.history,
            save_path=f"{output_dir}/action_distribution.png"
        )
    
    # Text report
    generate_report(
        ai_system,
        save_path=f"{output_dir}/system_report.txt"
    )
    
    print(f"✓ All visualizations generated in {output_dir}/")


if __name__ == "__main__":
    print("Visualization tools loaded")
    print("Import and use with: from utils.visualization import *")

