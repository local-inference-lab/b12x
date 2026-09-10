"""One requested TP4 FP8 tile policy; no automatic sweep or tuning."""
def select_tile(expected_m):
    if type(expected_m) is not int or expected_m < 1:
        raise ValueError("expected_m must be a positive integer")
    return (16 if expected_m <= 1 else 32 if expected_m <= 128 else 64, 128)


def row_tiles(valid_rows, tile_m):
    if type(valid_rows) is not int or valid_rows < 0 or tile_m not in (16, 32, 64):
        raise ValueError("invalid row tiling")
    return tuple((start, min(tile_m, valid_rows - start))
                 for start in range(0, valid_rows, tile_m))
