import operator
import random
import numpy as np
import sys
import textwrap
import pickle
import copy
from pathlib import Path

# USER IMPORTS (Assuming evaluate is provided in your evaluation.py)
from evaluation import evaluate

# -----------------------------------------------------------------------------
# USER IMPORTS / MOCKS
# -----------------------------------------------------------------------------
try:
    import marioai
    from agents import CodeAgent, Mario, Sprite
except ImportError:
    # Mocks for standalone testing if libraries are missing
    class Mario:
        KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_JUMP, KEY_SPEED = 0, 1, 2, 3, 4
    class Sprite:
        KIND_GOOMBA = 80
        KIND_GOOMBA_WINGED = 81
        KIND_RED_KOOPA = 82
        KIND_RED_KOOPA_WINGED = 83
        KIND_GREEN_KOOPA = 84
        KIND_GREEN_KOOPA_WINGED = 85
        KIND_BULLET_BILL = 86
        KIND_SPIKY = 87
        KIND_SPIKY_WINGED = 88
    class CodeAgent: pass
    print("Warning: marioai/agents modules not found. Using mocks.")

from deap import base, creator, tools, gp

# -----------------------------------------------------------------------------
# 0. HELPER: Safe Generator
# -----------------------------------------------------------------------------
def safe_gen_grow(pset, min_, max_, type_=None):
    if type_ is None: type_ = pset.ret
    expr = []
    stack = [(0, type_)]
    while stack:
        depth, type_ = stack.pop()
        try: has_primitives = len(pset.primitives[type_]) > 0
        except KeyError: has_primitives = False
        try: has_terminals = len(pset.terminals[type_]) > 0
        except KeyError: has_terminals = False
        
        if not has_terminals and not has_primitives:
            raise IndexError(f"Type '{type_.__name__}' has no primitives/terminals!")

        should_grow = False
        if not has_terminals: should_grow = True
        elif not has_primitives: should_grow = False
        else:
            if depth < min_: should_grow = True
            elif depth >= max_: should_grow = False
            else: should_grow = (random.random() < 0.5)

        if should_grow:
            prim = random.choice(pset.primitives[type_])
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
        else:
            term = random.choice(pset.terminals[type_])
            if isinstance(term, type): term = term()
            expr.append(term)
    return expr

def indent(text):
    return "\n".join("    " + line for line in text.split("\n"))

# -----------------------------------------------------------------------------
# 1. TYPE DEFINITIONS (Stripped Down)
# -----------------------------------------------------------------------------
class Expr: pass
class Condition: pass
class Key: pass
class Bool: pass

# -----------------------------------------------------------------------------
# 2. PRIMITIVES: STRING BUILDERS (Stripped Down)
# -----------------------------------------------------------------------------
def str_if_then(cond, expr):
    return f"if {cond}:\n{indent(expr)}"

def str_sequence(expr1, expr2):
    return f"{expr1}\n{expr2}"

def str_set_action(key, val):
    return f"action[{key}] = int({val})"

# -----------------------------------------------------------------------------
# 3. GRAMMAR CONFIGURATION (Bare Minimum)
# -----------------------------------------------------------------------------
pset = gp.PrimitiveSetTyped("MAIN", [], Expr)

# Core Logic
pset.addPrimitive(str_if_then, [Condition, Expr], Expr)
pset.addPrimitive(str_sequence, [Expr, Expr], Expr)
pset.addPrimitive(str_set_action, [Key, Bool], Expr)
pset.addTerminal("pass", Expr, name="NoOp")

# Basic Senses (Provided directly by the environment variables)
pset.addTerminal("on_ground", Condition, name="IsMarioOnGround")
pset.addTerminal("can_jump", Condition, name="MayMarioJump")

# Constants
pset.addTerminal("True", Bool)

# Limited Actions (Only Right and Jump)
pset.addTerminal("Mario.KEY_RIGHT", Key, name="RIGHT")
pset.addTerminal("Mario.KEY_JUMP", Key, name="JUMP")

# -----------------------------------------------------------------------------
# 4. RANDOM GENERATION SETUP
# -----------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", safe_gen_grow, pset=pset, min_=3, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate_gp_individual(individual):
    """Converts a tree individual into Python code and evaluates it."""
    code_body = toolbox.compile(individual)
    agent_prototype = CodeAgent
    full_code_str = f"""
def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
{indent(code_body)}
""" 
    try:
        reward = evaluate(agent_prototype, full_code_str)
    except NameError:
        # If your evaluation isn't loaded properly, mock a random score for testing
        print(" [Sim] Evaluation not linked properly. Returning 0 score.")
        reward = 0
        
    return reward

# -----------------------------------------------------------------------------
# 5. PERSISTENCE HELPERS
# -----------------------------------------------------------------------------
def save_best_individual(best_ind, toolbox, filename_py="mario_best.py"):
    
    """Saves the best individual as a readable Python script."""
    if best_ind is None:
        print("No individual to save.")
        return

    code_body = toolbox.compile(best_ind)
    fitness_val = best_ind.fitness.values[0] if best_ind.fitness.valid else "Unknown"
    
    full_code = f"""
# Evolved Mario Controller (Random Search)
# Fitness: {fitness_val}

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
{indent(code_body)}
"""
    Path("data/gp_best_agents").mkdir(parents=True, exist_ok=True)
    output_path = Path("data/gp_best_agents") / filename_py
    with output_path.open("w") as f:
        f.write(full_code)
    print(f"Saved executable code to '{filename_py}'")

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION: RANDOM SEARCH
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(int(sys.argv[1]))
    
    NUM_ITERATIONS = 50  # Number of random agents to generate and test
    
    best_individual = None
    best_fitness = -float('inf')
    
    print(f"Starting Random Search for {NUM_ITERATIONS} iterations...")
    
    for i in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{NUM_ITERATIONS} ---")
        
        # 1: Generate a random individual
        current_individual = toolbox.individual()
        
        # 2: Evaluate the generated individual
        fitness_score = evaluate_gp_individual(current_individual)
        
        # 3: Assign the fitness score to the individual's DEAP fitness attribute
        # DEAP requires fitness to be a tuple, so we add a comma
        current_individual.fitness.values = (fitness_score,)
        
        print(f"Fitness Score: {fitness_score}")
        
        # 4: Compare this fitness to your `best_fitness`
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            # We copy the individual so later changes (if we had them) wouldn't affect our best
            best_individual = copy.deepcopy(current_individual)
            print(f">>> New Best Found! Score: {best_fitness}")
            
    if best_individual:
        print(f"Final Best Fitness Found: {best_fitness}")
        # 5: Save the best individual using the helper function
        save_best_individual(best_individual, toolbox)
    else:
        print("No valid programs were found or evaluated.")