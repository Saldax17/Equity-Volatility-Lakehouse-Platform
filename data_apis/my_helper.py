class MyHelper:
    @classmethod
    def min_max_target(cls, open_price: float) -> tuple[float, float]:
        if open_price <= 0:
            raise ValueError("Open price must be greater than zero.")
        elif open_price <= 0.1:
            factor = 0.12
            decimal_places = 4
        elif open_price <= 0.5:
            factor = 0.08
            decimal_places = 4
        elif open_price <= 2:
            factor = 0.05
            decimal_places = 3
        elif open_price <= 10:
            factor = 0.03
            decimal_places = 2
        else:
            factor = 0.02
            decimal_places = 2
        open_price = round(open_price, decimal_places)
        min_target =  round(open_price * (1 - factor), decimal_places)
        max_target = (open_price - min_target) + open_price
        return (min_target, max_target)