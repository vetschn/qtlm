from qtlm import NDArray, xp, linalg


def sancho_rubio(
    a_ii: NDArray,
    a_ij: NDArray,
    a_ji: NDArray,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
) -> NDArray:
    """Calculates the surface Green's function iteratively.[^1].

    [^1]: M P Lopez Sancho et al., "Highly convergent schemes for the
    calculation of bulk and surface Green functions", 1985 J. Phys. F:
    Met. Phys. 15 851

    Parameters
    ----------
    a_ii : NDArray
        Diagonal boundary block of a system matrix.
    a_ij : NDArray
        Superdiagonal boundary block of a system matrix.
    a_ji : NDArray
        Subdiagonal boundary block of a system matrix.
    contact : str
        The contact to which the boundary blocks belong.
    max_iterations : int, optional
        The maximum number of iterations to perform.
    convergence_tol : float, optional
        The convergence tolerance for the iterative scheme. The
        criterion for convergence is that the average Frobenius norm of
        the update matrices `alpha` and `beta` is less than this value.
    Returns
    -------
    x_ii : NDArray
        The system's surface Green's function.

    """

    epsilon = a_ii.copy()
    epsilon_s = a_ii.copy()
    alpha = a_ji.copy()
    beta = a_ij.copy()

    delta = float("inf")
    for __ in range(max_iterations):
        inverse = linalg.inv(epsilon)

        epsilon = epsilon - alpha @ inverse @ beta - beta @ inverse @ alpha
        epsilon_s = epsilon_s - alpha @ inverse @ beta

        alpha = alpha @ inverse @ alpha
        beta = beta @ inverse @ beta

        delta = xp.linalg.norm(xp.abs(alpha) + xp.abs(beta), axis=(-2, -1)).max() / 2

        if delta < convergence_tol:
            break

    else:  # Did not break, i.e. max_iterations reached.
        raise RuntimeError(
            f"Sancho-Rubio did not converge within {max_iterations} iterations."
        )

    return linalg.inv(epsilon_s)
