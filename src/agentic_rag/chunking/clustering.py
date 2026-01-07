"""
Clustering algorithms for RAPTOR tree construction.

RAPTOR uses clustering to group similar chunks before summarization.
Supports KMeans and GMM (Gaussian Mixture Models) clustering.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ClusterResult(BaseModel):
    """Result of clustering operation."""

    labels: list[int] = Field(
        default_factory=list,
        description="Cluster assignment for each item",
    )
    n_clusters: int = Field(
        default=0,
        description="Number of clusters found",
    )
    centroids: list[list[float]] = Field(
        default_factory=list,
        description="Cluster centroids (if applicable)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional clustering metadata",
    )


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def cluster(
        self,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
    ) -> ClusterResult:
        """
        Cluster embeddings.

        Args:
            embeddings: Array of shape (n_samples, n_features).
            n_clusters: Target number of clusters (if applicable).

        Returns:
            ClusterResult with assignments and metadata.
        """
        pass


class KMeansClusterer(BaseClusterer):
    """
    KMeans clustering for RAPTOR.

    Fast and deterministic clustering. Best when:
    - Number of clusters is known
    - Clusters are roughly spherical
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
    ):
        """
        Initialize KMeans clusterer.

        Args:
            random_state: Random seed for reproducibility.
            max_iter: Maximum iterations.
            n_init: Number of initializations.
        """
        self._random_state = random_state
        self._max_iter = max_iter
        self._n_init = n_init

    def cluster(
        self,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
    ) -> ClusterResult:
        """Cluster embeddings using KMeans."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn required for RAPTOR clustering. "
                "Install with: pip install scikit-learn>=1.4.0"
            )

        n_samples = len(embeddings)

        # Auto-determine clusters if not specified
        if n_clusters is None:
            # Heuristic: sqrt(n/2) clusters, min 2, max 10
            n_clusters = max(2, min(10, int(np.sqrt(n_samples / 2))))

        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, n_samples)

        if n_samples < 2:
            return ClusterResult(
                labels=[0] * n_samples,
                n_clusters=1,
                centroids=[embeddings[0].tolist()] if n_samples > 0 else [],
            )

        logger.debug(f"KMeans clustering {n_samples} items into {n_clusters} clusters")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self._random_state,
            max_iter=self._max_iter,
            n_init=self._n_init,
        )

        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        return ClusterResult(
            labels=labels.tolist(),
            n_clusters=n_clusters,
            centroids=centroids.tolist(),
            metadata={
                "algorithm": "kmeans",
                "inertia": float(kmeans.inertia_),
                "n_iter": kmeans.n_iter_,
            },
        )


class GMMClusterer(BaseClusterer):
    """
    Gaussian Mixture Model clustering for RAPTOR.

    Soft clustering that handles overlapping clusters better.
    Recommended for RAPTOR as it produces better cluster boundaries.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 100,
        n_init: int = 5,
        covariance_type: str = "full",
    ):
        """
        Initialize GMM clusterer.

        Args:
            random_state: Random seed.
            max_iter: Maximum EM iterations.
            n_init: Number of initializations.
            covariance_type: Covariance type (full, tied, diag, spherical).
        """
        self._random_state = random_state
        self._max_iter = max_iter
        self._n_init = n_init
        self._covariance_type = covariance_type

    def _determine_n_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 10,
    ) -> int:
        """
        Determine optimal cluster count using BIC.

        Args:
            embeddings: Input embeddings.
            max_clusters: Maximum clusters to try.

        Returns:
            Optimal number of clusters.
        """
        from sklearn.mixture import GaussianMixture

        n_samples = len(embeddings)
        max_k = min(max_clusters, n_samples - 1)

        if max_k < 2:
            return 1

        best_bic = float("inf")
        best_k = 2

        for k in range(2, max_k + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=self._random_state,
                    max_iter=50,  # Quick fit for selection
                    n_init=1,
                )
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)

                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception:
                continue

        return best_k

    def cluster(
        self,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
    ) -> ClusterResult:
        """Cluster embeddings using Gaussian Mixture Model."""
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError(
                "scikit-learn required for RAPTOR clustering. "
                "Install with: pip install scikit-learn>=1.4.0"
            )

        n_samples = len(embeddings)

        if n_samples < 2:
            return ClusterResult(
                labels=[0] * n_samples,
                n_clusters=1,
                centroids=[embeddings[0].tolist()] if n_samples > 0 else [],
            )

        # Auto-determine if not specified
        if n_clusters is None:
            n_clusters = self._determine_n_clusters(embeddings)

        n_clusters = min(n_clusters, n_samples)

        logger.debug(f"GMM clustering {n_samples} items into {n_clusters} clusters")

        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=self._random_state,
            max_iter=self._max_iter,
            n_init=self._n_init,
            covariance_type=self._covariance_type,
        )

        labels = gmm.fit_predict(embeddings)

        # Get cluster means as centroids
        centroids = gmm.means_

        return ClusterResult(
            labels=labels.tolist(),
            n_clusters=n_clusters,
            centroids=centroids.tolist(),
            metadata={
                "algorithm": "gmm",
                "bic": float(gmm.bic(embeddings)),
                "converged": gmm.converged_,
                "n_iter": gmm.n_iter_,
            },
        )


def create_clusterer(
    algorithm: str = "gmm",
    **kwargs: Any,
) -> BaseClusterer:
    """
    Factory function to create a clusterer.

    Args:
        algorithm: Clustering algorithm (kmeans, gmm).
        **kwargs: Additional arguments for clusterer.

    Returns:
        Configured clusterer instance.
    """
    if algorithm == "kmeans":
        return KMeansClusterer(**kwargs)
    elif algorithm == "gmm":
        return GMMClusterer(**kwargs)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
