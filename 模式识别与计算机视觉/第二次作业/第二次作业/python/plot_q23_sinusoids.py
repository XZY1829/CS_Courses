from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    points = [(10, 10), (20, 20), (30, 30)]
    theta = np.linspace(0, 2 * np.pi, 1000)

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for x, y in points:
        rho = x * np.cos(theta) + y * np.sin(theta)
        plt.plot(theta, rho, label=f"({x}, {y})")

    # Mark the shared intersections for y=x line: theta=3π/4 gives rho=0 for all these points.
    theta_intersect = 3 * np.pi / 4
    plt.scatter(
        [theta_intersect],
        [0.0],
        color="red",
        zorder=5,
        label=r"intersection at $(\theta,\rho)=(3\pi/4, 0)$",
    )

    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\rho$ (pixel)")
    plt.title("Hough-space sinusoids for points (10,10), (20,20), (30,30)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_path = out_dir / "q23_hough_sinusoids.png"
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


if __name__ == "__main__":
    main()
