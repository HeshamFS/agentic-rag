"""
Component registry for RAG pipeline.

Provides centralized registration and discovery of
pipeline components (embedders, retrievers, generators, etc.).
"""

from typing import Any, TypeVar

T = TypeVar("T")


class ComponentRegistry[T]:
    """
    Generic component registry.

    Allows registration and lookup of components by name.
    """

    def __init__(self, component_type: str = "component"):
        """
        Initialize registry.

        Args:
            component_type: Human-readable component type name.
        """
        self._type = component_type
        self._components: dict[str, type[T]] = {}
        self._instances: dict[str, T] = {}
        self._default: str | None = None

    def register(
        self,
        name: str,
        component_class: type[T],
        default: bool = False,
    ) -> None:
        """
        Register a component class in the registry.

        Args:
            name: Unique name identifier for the component (e.g., "qwen3", "hybrid").
            component_class: The class to be registered.
            default: If True, this component becomes the default for its type.
        """
        self._components[name] = component_class
        if default or self._default is None:
            self._default = name

    def register_instance(self, name: str, instance: T) -> None:
        """
        Register a pre-created instance.

        Args:
            name: Instance name.
            instance: Component instance.
        """
        self._instances[name] = instance

    def get_class(self, name: str) -> type[T]:
        """
        Get a component class by name.

        Args:
            name: Component name.

        Returns:
            Component class.

        Raises:
            KeyError: If component not found.
        """
        if name not in self._components:
            raise KeyError(
                f"Unknown {self._type}: '{name}'. Available: {list(self._components.keys())}"
            )
        return self._components[name]

    def get(self, name: str | None = None, **kwargs: Any) -> T:
        """
        Retrieve a component instance by name.

        If the name is not provided, it returns the default component.
        It first checks for pre-registered singleton instances, otherwise
        it creates a new instance of the registered class using provided kwargs.

        Args:
            name: Component identifier (None for default).
            **kwargs: Configuration parameters passed to the component constructor.

        Returns:
            An instance of the requested component.

        Raises:
            ValueError: If no component name is provided and no default is set.
            KeyError: If the specified name is not registered.
        """
        name = name or self._default
        if name is None:
            raise ValueError(f"No {self._type} registered")

        # Check for pre-created instance
        if name in self._instances:
            return self._instances[name]

        # Create new instance
        component_class = self.get_class(name)
        return component_class(**kwargs)

    def list_components(self) -> list[str]:
        """List registered component names."""
        return list(self._components.keys())

    @property
    def default(self) -> str | None:
        """Get default component name."""
        return self._default

    def set_default(self, name: str) -> None:
        """Set default component."""
        if name not in self._components:
            raise KeyError(f"Unknown {self._type}: '{name}'")
        self._default = name


# Global registries for each component type
_registries: dict[str, ComponentRegistry] = {}


def get_registry(component_type: str) -> ComponentRegistry:
    """
    Get or create a registry for a component type.

    Args:
        component_type: Type of component.

    Returns:
        Registry for that component type.
    """
    if component_type not in _registries:
        _registries[component_type] = ComponentRegistry(component_type)
    return _registries[component_type]


# Convenience registries
embedder_registry: ComponentRegistry = get_registry("embedder")
retriever_registry: ComponentRegistry = get_registry("retriever")
generator_registry: ComponentRegistry = get_registry("generator")
chunker_registry: ComponentRegistry = get_registry("chunker")
reranker_registry: ComponentRegistry = get_registry("reranker")


def register_embedder(name: str, default: bool = False):
    """Decorator to register an embedder class."""

    def decorator(cls):
        embedder_registry.register(name, cls, default=default)
        return cls

    return decorator


def register_retriever(name: str, default: bool = False):
    """Decorator to register a retriever class."""

    def decorator(cls):
        retriever_registry.register(name, cls, default=default)
        return cls

    return decorator


def register_generator(name: str, default: bool = False):
    """Decorator to register a generator class."""

    def decorator(cls):
        generator_registry.register(name, cls, default=default)
        return cls

    return decorator


def register_chunker(name: str, default: bool = False):
    """Decorator to register a chunker class."""

    def decorator(cls):
        chunker_registry.register(name, cls, default=default)
        return cls

    return decorator


def register_reranker(name: str, default: bool = False):
    """Decorator to register a reranker class."""

    def decorator(cls):
        reranker_registry.register(name, cls, default=default)
        return cls

    return decorator
