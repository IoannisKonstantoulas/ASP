import time


def time_print_loop(start_time, i, interval=1000):
    if (i + 1) % interval == 0:
        now = time.time()
        elapsed = now - start_time
        print(f"Iteration {i + 1}: {elapsed:.2f} seconds")
