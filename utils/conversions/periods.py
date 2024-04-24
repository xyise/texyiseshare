def tenor_to_annum(tenor: str) -> float:

    tenor = tenor.strip()
    unit = tenor[-1]
    val = float(tenor[:-1])
    if unit == "m":
        return val / 12.0
    elif unit == "y":
        return val
    else:
        raise ValueError(f"Unknown tenor unit: {unit}")