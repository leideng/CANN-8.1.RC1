# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

from collections import defaultdict, deque
from typing import List
from ms_service_profiler.plugins.base import PluginBase


class DependencyNotFoundError(Exception):
    def __init__(self, plugin_name, missing_dependency):
        super().__init__(f"Dependency '{missing_dependency}' not found for plugin '{plugin_name}'")


class DependencyCycleError(Exception):
    def __init__(self, message=f"A cycle was detected in the plugin dependencies."):
        super().__init__(message)


def sort_plugins(plugins: List[PluginBase]) -> List[PluginBase]:
    # Build the dependency graph
    graph = defaultdict(list)
    indegree = {plugin.name: 0 for plugin in plugins}

    for plugin in plugins:
        for dependency in plugin.depends:
            if dependency not in indegree:
                raise DependencyNotFoundError(plugin.name, dependency)
            graph[dependency].append(plugin.name)
            indegree[plugin.name] += 1

    # Perform topological sorting
    queue = deque([plugin.name for plugin in plugins if indegree[plugin.name] == 0])
    sorted_plugins = []

    while queue:
        current = queue.popleft()
        sorted_plugins.append(current)

        for neighbor in graph[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # Check if topological sorting was successful (i.e., no cycles)
    if len(sorted_plugins) != len(indegree):
        raise DependencyCycleError("A cycle was detected in the plugin dependencies.")

    # Create a mapping to return the sorted plugins
    sorted_plugin_objects = {plugin.name: plugin for plugin in plugins}
    return [sorted_plugin_objects[name] for name in sorted_plugins]
