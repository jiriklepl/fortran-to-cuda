"""Source-budgeted descriptor control traversal, independent of call depth.

Resume the existing classifier in original depth-first order. Suspended loop
frames supply only lexical EXIT/CYCLE context; they never execute, skip an arm,
or authorize continuation after an original native control operation.
"""


def visit_descriptor_control(check, routine, graph, identities):
    frames = [iter(check.control_children(routine, graph, identities))]
    try:
        while frames:
            try:
                children = next(frames[-1])
            except StopIteration:
                frames.pop()
                continue
            frames.append(iter(check.control_children(routine, graph, children)))
    finally:
        # A hard failure or typed frontend demand abandons this full attempt.
        # Closing inner frames first runs the original lexical-loop finally
        # clauses, including when a nested procedure fails before returning.
        for frame in reversed(frames):
            frame.close()
