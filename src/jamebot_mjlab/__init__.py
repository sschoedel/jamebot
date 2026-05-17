"""jamebot_mjlab: jamebot robot + RL tasks for mjlab.

Importing this package registers all jamebot tasks in mjlab's task registry
so that ``mjlab.scripts.train`` / ``play`` can resolve them by name.
"""

from jamebot_mjlab import tasks as tasks
