import numpy as np
from scipy.linalg import lstsq, norm
from scipy.linalg import solve_triangular, qr
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd
import time

class DEIM:
    """Discrete Empirical Interpolation Method for index selection."""

    def __init__(self, method='standard'):
        """
        Initialize DEIM selector.

        Parameters:
        -----------
        method : str
            'standard', 'q-deim', 'l-deim', or 's-deim'
        """
        self.method = method

    def select_indices(self, U, k=None):
        """
        Select k indices using DEIM algorithm.

        Parameters:
        -----------
        U : array-like, shape (n, k)
            Basis matrix (e.g., left singular vectors)
        k : int, optional
            Number of indices to select (default: all columns of U)

        Returns:
        --------
        indices : array of integers
            Selected indices
        """
        n, m = U.shape
        if k is None:
            k = m
        if k > m:
            raise ValueError(f"k ({k}) cannot be larger than number of columns ({m})")

        if self.method == 'standard':
            return self._standard_deim(U, k)
        elif self.method == 'q-deim':
            return self._q_deim(U, k)
        elif self.method == 'l-deim':
            return self._l_deim(U, k)
        elif self.method == 's-deim':
            return self._s_deim(U, k)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _standard_deim(self, U, k):
        """Standard DEIM algorithm."""
        n, m = U.shape
        indices = np.zeros(k, dtype=int)

        # First index
        indices[0] = np.argmax(np.abs(U[:, 0]))

        # Iterative selection
        for j in range(1, k):
            U_selected = U[indices[:j], :j]
            c = solve_triangular(U_selected, U[indices[:j], j], lower=True)
            r = U[:, j] - U[:, :j] @ c
            indices[j] = np.argmax(np.abs(r))

        return indices

    def _q_deim(self, U, k):
        """
        Q-DEIM: QR-based DEIM

        Advantages:
        - Numerically stable
        - Guaranteed condition number bounds
        - Efficient for full rank selection (k = m)

        Disadvantages:
        - Less adaptive than iterative methods
        - May not capture local features well
        - Higher computational cost for QR
        """
        n, m = U.shape

        # QR decomposition with column pivoting on U^T
        # This finds most linearly independent rows of U
        Q, R, P = qr(U.T, mode='full', pivoting=True)

        # First k pivot indices
        indices = P[:k]

        # Alternative implementation using row QR
        # _, _, P_row = qr(U, mode='full', pivoting=True)
        # indices = sorted(P_row[:k])

        return np.sort(indices)

    def _l_deim(self, U, k, spatial_coords=None):
        """
        L-DEIM: Localized DEIM

        Advantages:
        - Spatially coherent selections
        - Better for localized phenomena
        - Incorporates domain knowledge

        Disadvantages:
        - Requires spatial information
        - May sacrifice optimality for locality
        - Parameter-dependent (locality radius)
        """
        n, m = U.shape
        indices = np.zeros(k, dtype=int)

        # Generate spatial coordinates if not provided
        if spatial_coords is None:
            # Assume 1D uniform spacing
            spatial_coords = np.arange(n).reshape(-1, 1)

        # Locality parameters
        locality_radius = n / (2 * k)  # Adaptive radius
        decay_rate = 1.0 / locality_radius

        # First index: standard DEIM
        indices[0] = np.argmax(np.abs(U[:, 0]))

        # Subsequent indices with locality weighting
        for j in range(1, k):
            # Standard DEIM components
            U_selected = U[indices[:j], :j]
            c = solve_triangular(U_selected, U[indices[:j], j], lower=True)
            r = U[:, j] - U[:, :j] @ c

            # Compute locality weights
            selected_coords = spatial_coords[indices[:j]]
            distances = cdist(spatial_coords, selected_coords, metric='euclidean')
            min_distances = np.min(distances, axis=1)

            # Exponential decay weight
            locality_weights = np.exp(-decay_rate * min_distances)

            # Alternative: Gaussian weight
            # locality_weights = np.exp(-0.5 * (min_distances / locality_radius)**2)

            # Weighted selection
            weighted_residual = np.abs(r) * locality_weights
            indices[j] = np.argmax(weighted_residual)

        return indices

    def _s_deim(self, U, k):
        """
        Strong DEIM (S-DEIM) - Proper Implementation

        Algorithm:
        1. Select index with maximum magnitude across ALL columns
        2. Orthogonalize remaining basis against interpolation at selected points
        3. Repeat until k indices selected

        This ensures maximum linear independence of selected rows.
        """
        n, m = U.shape
        indices = []

        # Copy to avoid modifying original
        V = U.copy()

        for i in range(k):
            # Find global maximum across all columns and rows
            abs_V = np.abs(V)
            idx, col = np.unravel_index(np.argmax(abs_V), abs_V.shape)

            # Store selected index
            indices.append(idx)

            # Orthogonalize: remove the contribution of row idx
            # from all columns to ensure linear independence

            if i < k - 1:  # No need to orthogonalize after last selection
                # Method 1: Direct orthogonalization against selected row
                row_vec = V[idx, :].copy()
                norm_sq = np.dot(row_vec, row_vec)

                if norm_sq > 1e-10:
                    # Project out component in direction of selected row
                    for row in range(n):
                        if row != idx:
                            projection = np.dot(V[row, :], row_vec) / norm_sq
                            V[row, :] -= projection * row_vec

                # Zero out the selected row to avoid reselection
                V[idx, :] = 0

        return np.array(indices)


class DEIM_CX:
    """CX decomposition using DEIM for column selection."""

    def __init__(self, n_components=10, oversampling=5, method='standard', solver='ridge', regularization=0.1):
        self.n_components = n_components
        self.oversampling = oversampling
        self.method = method
        self.reg_type = solver
        self.regularization = regularization

    def fit_transform(self, A):
        """
        Perform CX decomposition with DEIM column selection.

        Returns:
        --------
        C : Selected columns
        X : Coefficient matrix
        indices : Selected column indices
        """
        # m, n = A.shape
        # k = self.n_components
        #
        # # Compute right singular vectors
        # _, _, Vt = np.linalg.svd(A, full_matrices=False)
        # V = Vt.conj().T[:, :k + self.oversampling]
        #
        # # Apply DEIM to select columns
        # deim = DEIM(method=self.method)
        # indices = deim.select_indices(V, k)

        indices = self.deim_selection(A)

        # Extract columns
        C = A[:, indices]

        # Compute X with regularization
        if self.reg_type == 'l2':
            X = self._solve_l2_regularized(C, A)
        elif self.reg_type == 'lstsq':
            X = self._solve_lstsq(C, A)
        elif self.reg_type == 'ridge':
            X = self._solve_ridge(C, A)
        elif self.reg_type == 'pinv':
            X = self._solve_pinv(C, A)
        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")

        return C, X, indices

    def deim_selection(self, A):
        """DEIM selection adapted for complex matrices."""
        print('deim')

        # Compute SVD

        if A.dtype == complex:
            n_components = min(self.n_components, min(A.shape) - 1)
            U, s, Vh = np.linalg.svd(A, full_matrices=False)
            Vh = Vh[:n_components, :]
        else:
            U, s, Vh = randomized_svd(A, n_components=self.n_components)

        # Complex SVD
        V = Vh.conj().T

        indices = np.zeros(self.n_components, dtype=int)

        # First index: maximum magnitude
        indices[0] = np.argmax(np.abs(V[:, 0]))

        # Subsequent indices
        for j in range(1, self.n_components):
            V_selected = V[indices[:j], :j]

            # Solve with conjugate transpose
            c = np.linalg.solve(V_selected, V[indices[:j], j])

            # Residual
            r = V[:, j] - V[:, :j] @ c

            # Maximum magnitude of residual
            indices[j] = np.argmax(np.abs(r))

        return indices

    def _solve_ridge(self, C, A):
        print('ridge', self.regularization)
        CH_C = C.conj().T @ C

        # Add regularization
        lambda_I = self.regularization * np.eye(self.n_components, dtype=np.complex128)

        # Compute regularized inverse
        inv1 = np.linalg.inv(CH_C + lambda_I)

        # Compute X
        X = inv1 @ C.conj().T @ A

        return X

    def _solve_pinv(self, C, A):
        print('pinv')
        # Compute X
        X = np.linalg.pinv(C) @ A
        return X

    def _solve_lstsq(self, C, A):
        print('lstsq')
        # Compute X
        X = np.linalg.lstsq(C, A, rcond=None)[0]
        return X

    def _solve_l2_regularized(self, C, A):
        """Solve with L2 (Ridge) regularization."""
        print('l2', self.regularization)
        # Solve: min ||A - CX||_F^2 + lambda ||X||_F^2
        # Solution: X = (C^T C + lambda I)^{-1} C^T A

        k = C.shape[1]
        CHC = C.conj().T @ C
        lambda_I = self.regularization * np.eye(k, dtype=np.complex128)

        # Cholesky for Hermitian positive definite
        try:
            L = np.linalg.cholesky(CHC + lambda_I)
            CHA = C.conj().T @ A
            Y = np.linalg.solve(L, CHA)
            X = np.linalg.solve(L.conj().T, Y)
        except np.linalg.LinAlgError:
            # Fallback to direct solve
            X = np.linalg.solve(CHC + lambda_I, C.conj().T @ A)

        return X

def analyze_conditioning(C, X, A):
    """Analyze numerical conditioning of complex CX decomposition."""
    # Condition numbers
    cond_C = np.linalg.cond(C)
    cond_X = np.linalg.cond(X)

    # Effective condition number
    CHC = C.conj().T @ C
    cond_CHC = np.linalg.cond(CHC)

    # Reconstruction stability
    reconstruction = C @ X
    forward_error = norm(A - reconstruction, 'fro') / norm(A, 'fro')

    # Backward error analysis
    residual = A - reconstruction
    backward_error = norm(residual, 'fro') / (norm(C, 'fro') * norm(X, 'fro'))

    # Phase stability
    phase_A = np.angle(A)
    phase_CX = np.angle(reconstruction)
    phase_error = np.mean(np.abs(phase_A - phase_CX))

    return {
        'cond_C': cond_C,
        'cond_X': cond_X,
        'cond_CHC': cond_CHC,
        'forward_error': forward_error,
        'backward_error': backward_error,
        'phase_error': phase_error
    }

if __name__ == '__main__':
    # Example: Basic DEIM usage
    np.random.seed(42)

    # Generate test basis (e.g., from SVD)
    n = 100
    k = 10
    A = np.random.randn(n, 50)
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    U_basis = U[:, :k]

    cur_rank = 15

    for method in ['standard', 'q-deim', 's-deim', 'l-deim']:
        # Apply DEIM

        print('method', method)

        time1 = time.time()
        # deim = DEIM(method=method)
        # indices = deim.select_indices(U_basis)

        deim = DEIM_CX(method=method, n_components=cur_rank)
        C, X, indices = deim.fit_transform(A)
        print('time', time.time() - time1)
        print('error', np.linalg.norm(A - C @ X) / np.linalg.norm(A))

        print(f"Selected indices: {indices}")
        print(f"Condition number of P^T U: {np.linalg.cond(U_basis[indices, :]):.2f}")
        print(analyze_conditioning(C, X, A))


