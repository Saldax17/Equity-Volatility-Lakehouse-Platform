class MyHelper:
    """
    Utility helper class containing static computations used across the project.
    """

    @classmethod
    def min_max_target(cls, open_price: float) -> tuple[float, float]:
        """
        Computes dynamic min/max threshold levels based on the open price.
        Used for volatility/trend detection in intraday bars.

        Parameters
        ----------
        open_price : float
            Opening price of the bar. Must be > 0.

        Returns
        -------
        (min_target, max_target) : tuple[float, float]
            Lower and upper price thresholds based on volatility factor rules.

        Rules (factor & precision)
        --------------------------
        open_price <= 0.10    → factor = 0.12, decimals = 4
        open_price <= 0.50    → factor = 0.08, decimals = 4
        open_price <= 2       → factor = 0.05, decimals = 3
        open_price <= 10      → factor = 0.03, decimals = 2
        open_price > 10       → factor = 0.02, decimals = 2
        """

        if open_price <= 0:
            raise ValueError("open_price must be greater than zero.")

        # Determine factor & precision
        factor, decimals = cls._determine_threshold_params(open_price)

        # Round input to required precision
        open_rounded = round(open_price, decimals)

        # Compute thresholds
        min_target = round(open_rounded * (1 - factor), decimals)
        max_target = round(open_rounded + (open_rounded - min_target), decimals)

        return min_target, max_target

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    @staticmethod
    def _determine_threshold_params(open_price: float) -> tuple[float, int]:
        """Return (factor, decimal_places) based on price rules."""
        if open_price <= 0.10:
            return 0.12, 4
        if open_price <= 0.50:
            return 0.08, 4
        if open_price <= 2:
            return 0.05, 3
        if open_price <= 10:
            return 0.03, 2
        return 0.02, 2
