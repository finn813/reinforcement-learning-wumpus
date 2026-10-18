from collections import deque
import gymnasium as gym

# Same action encoding as the environment: 0 up, 1 left, 2 down, 3 right
MOVES = [(0, 1), (-1, 0), (0, -1), (1, 0)]
START = (0, 0)


def dpll(clauses):
    """True if the clauses (frozensets of int literals, -x = not x) are satisfiable."""
    while True:  # unit propagation
        unit = next((next(iter(c)) for c in clauses if len(c) == 1), None)
        if unit is None:
            break
        clauses = assign(clauses, unit)
        if clauses is None:
            return False
    if not clauses:
        return True
    literal = next(iter(clauses[0]))
    for choice in (literal, -literal):
        reduced = assign(clauses, choice)
        if reduced is not None and dpll(reduced):
            return True
    return False


def assign(clauses, literal):
    """Set literal to true. Returns the simplified clauses, or None on an empty clause."""
    result = []
    for c in clauses:
        if literal in c:
            continue
        if -literal in c:
            c = c - {-literal}
            if not c:
                return None
        result.append(c)
    return result


class KBWumpusAgent:
    def __init__(self, env: gym.Env):
        """Initialize a knowledge-based agent that only enters cells it can prove safe.

        The knowledge base holds propositional clauses (CNF) over the symbols
        P (pit), W (wumpus), B (breeze) and S (stench) for every cell.
        """
        self.width, self.height = env.unwrapped.width, env.unwrapped.height
        self.cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.axioms = self.build_axioms()
        self.reset()

    # Propositional symbols as integers, one block of numbers per symbol type
    def P(self, cell): return 1 + self.cells.index(cell)
    def W(self, cell): return 1 + len(self.cells) + self.cells.index(cell)
    def B(self, cell): return 1 + 2 * len(self.cells) + self.cells.index(cell)
    def S(self, cell): return 1 + 3 * len(self.cells) + self.cells.index(cell)

    def neighbors(self, cell):
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in MOVES
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height]

    def build_axioms(self):
        """Rules of the world in CNF."""
        clauses = [{-self.P(START)}, {-self.W(START)}]  # the start cell is safe
        for cell in self.cells:
            for percept, hazard in ((self.B, self.P), (self.S, self.W)):
                # B(cell) <=> P(n1) or P(n2) or ...   (same for S and W)
                clauses.append({-percept(cell)} | {hazard(n) for n in self.neighbors(cell)})
                clauses += [{-hazard(n), percept(cell)} for n in self.neighbors(cell)]
        clauses.append({self.W(cell) for cell in self.cells})  # at least one wumpus
        clauses += [{-self.W(a), -self.W(b)}                   # at most one wumpus
                    for i, a in enumerate(self.cells) for b in self.cells[i + 1:]]
        return [frozenset(c) for c in clauses]

    def reset(self):
        """Forget everything about the previous map."""
        self.kb = list(self.axioms)
        self.visited = set()
        self.safe = {START}

    def tell(self, clause):
        self.kb.append(frozenset(clause))

    def ask(self, clause):
        """KB |= clause? Proof by refutation: KB and not(clause) is unsatisfiable."""
        return not dpll(self.kb + [frozenset({-literal}) for literal in clause])

    def get_action(self, obs) -> int:
        """Returns: action: 0 (up), 1 (left), 2 (down), 3 (right)"""
        x, y, breeze, stench = (int(v) for v in obs)
        cell = (x, y)
        if cell not in self.visited:
            self.visited.add(cell)
            # The agent is alive here, so the cell is safe, plus what it perceives
            self.tell({-self.P(cell)})
            self.tell({-self.W(cell)})
            self.tell({self.B(cell) if breeze else -self.B(cell)})
            self.tell({self.S(cell) if stench else -self.S(cell)})
            for other in self.cells:
                if other not in self.safe and self.ask({-self.P(other)}) and self.ask({-self.W(other)}):
                    self.safe.add(other)

        action = self.route(cell, self.safe - self.visited)
        if action is None:  # nothing left that is provably safe: go back to the start and wait
            action = self.route(cell, {START}) if cell != START else 1  # left = bump into the wall
        return action

    def route(self, start, goals):
        """First action of a shortest path through safe cells to the nearest goal (BFS)."""
        first_action = {start: None}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell in goals:
                return first_action[cell]
            for action, (dx, dy) in enumerate(MOVES):
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in self.safe and nxt not in first_action:
                    first_action[nxt] = action if cell == start else first_action[cell]
                    queue.append(nxt)
        return None