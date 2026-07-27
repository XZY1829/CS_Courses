from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
rng = np.random.default_rng(0)

mu = 2
sigma = 0.5


def sample_lognormal(n):
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def true_pdf(grid):
    y = np.zeros_like(grid, dtype=float)
    positive = grid > 0
    x = grid[positive]
    y[positive] = (
        1
        / (np.sqrt(2 * np.pi) * sigma * x)
        * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
    )
    return y


def silverman_bandwidth(samples):
    n = len(samples)
    std = np.std(samples, ddof=1)
    iqr = np.subtract(*np.percentile(samples, [75, 25]))
    scale = min(std, iqr / 1.34)
    return 0.9 * scale * n ** (-1 / 5)


def kde_with_bandwidth(samples, bandwidth, grid):
    # Pure NumPy Gaussian KDE to avoid SciPy DLL issues on Windows.
    samples = np.asarray(samples, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = samples.size
    coef = 1.0 / (n * bandwidth * np.sqrt(2 * np.pi))

    density = np.empty_like(grid, dtype=float)
    chunk_size = 80
    for start in range(0, grid.size, chunk_size):
        stop = min(start + chunk_size, grid.size)
        z = (grid[start:stop, None] - samples[None, :]) / bandwidth
        density[start:stop] = coef * np.exp(-0.5 * z**2).sum(axis=1)

    return density


def part_a():
    x = sample_lognormal(1000)
    grid = np.linspace(0.001, x.max() * 1.2, 1000)

    theoretical_mean = np.exp(mu + sigma**2 / 2)
    theoretical_var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)

    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=40, density=True, alpha=0.5, label="sample histogram")
    plt.plot(grid, true_pdf(grid), "r", linewidth=2, label="true pdf")
    plt.title("Lognormal Samples and True PDF, n=1000")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kde_part_a_true_pdf.png", dpi=160)
    plt.close()

    print("Part (a)")
    print(f"Sample mean = {x.mean():.6f}")
    print(f"Sample variance = {x.var(ddof=1):.6f}")
    print(f"Theoretical mean = {theoretical_mean:.6f}")
    print(f"Theoretical variance = {theoretical_var:.6f}")
    print()

    return x, grid


def part_b(x, grid):
    bw = silverman_bandwidth(x)
    y = kde_with_bandwidth(x, bw, grid)

    plt.figure(figsize=(7, 4))
    plt.plot(grid, true_pdf(grid), "k", linewidth=2, label="true pdf")
    plt.plot(grid, y, "r--", linewidth=2, label="KDE")
    plt.title("KDE with Automatic Bandwidth, n=1000")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kde_part_b_auto_bandwidth.png", dpi=160)
    plt.close()

    print("Part (b)")
    print(f"Automatic bandwidth for n = 1000: {bw:.6f}")
    print()

    return bw


def part_c(x, grid):
    y_02 = kde_with_bandwidth(x, 0.2, grid)
    y_5 = kde_with_bandwidth(x, 5, grid)

    plt.figure(figsize=(7, 4))
    plt.plot(grid, true_pdf(grid), "k", linewidth=2, label="true pdf")
    plt.plot(grid, y_02, "b--", linewidth=1.5, label="bandwidth = 0.2")
    plt.plot(grid, y_5, "r-.", linewidth=1.5, label="bandwidth = 5")
    plt.title("KDE with Different Bandwidths")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kde_part_c_bandwidth_compare.png", dpi=160)
    plt.close()

    print("Part (c)")
    print("Bandwidth 0.2: rougher curve, larger variance.")
    print("Bandwidth 5: over-smoothed curve, larger bias.")
    print()


def part_d(grid):
    sample_sizes = [1000, 10000, 100000]
    bws = []

    plt.figure(figsize=(7, 4))
    plt.plot(grid, true_pdf(grid), "k", linewidth=2, label="true pdf")

    styles = ["r--", "b-.", "g:"]
    for n, style in zip(sample_sizes, styles):
        x = sample_lognormal(n)
        bw = silverman_bandwidth(x)
        bws.append(bw)
        y = kde_with_bandwidth(x, bw, grid)
        plt.plot(grid, y, style, linewidth=1.5, label=f"n={n}")

    plt.title("KDE with Automatic Bandwidth for Different Sample Sizes")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kde_part_d_sample_size_compare.png", dpi=160)
    plt.close()

    print("Part (d)")
    for n, bw in zip(sample_sizes, bws):
        print(f"n = {n}, automatic bandwidth = {bw:.6f}")


def main():
    x, grid = part_a()
    part_b(x, grid)
    part_c(x, grid)
    part_d(grid)


if __name__ == "__main__":
    main()