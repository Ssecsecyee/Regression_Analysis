from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MASK_PATH = ROOT / "data" / "masks" / "active_mask.nc"
LAND_MASK_PATH = ROOT / "data" / "masks" / "land_mask.nc"
OUTPUT_PATH = ROOT / "experiments" / "01_baseline_linear_ridge" / "figures" / "regression" / "nsr_bottlenecks_jul_oct.png"


POINTS = {
    "Vilkitsky Strait": {
        "yx": (259, 183),
        "id": "1",
    },
    "NW Laptev Sea": {
        "yx": (262, 171),
        "id": "2",
    },
    "Sannikov Strait": {
        "yx": (281, 150),
        "id": "3",
    },
    "Dmitry Laptev Strait": {
        "yx": (286, 146),
        "id": "4",
    },
    "Central East Siberian Sea": {
        "yx": (273, 126),
        "id": "5",
    },
    "Long Strait / Wrangel": {
        "yx": (277, 94),
        "id": "6",
    },
}


MONTHS = [7, 8, 9, 10]
OCEAN_COLOR = np.array([48, 28, 70]) / 255
LAND_COLOR = np.array([18, 18, 18]) / 255
ACTIVE_COLOR = np.array([42, 204, 210]) / 255


def main() -> None:
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Mask file not found: {MASK_PATH}")
    if not LAND_MASK_PATH.exists():
        raise FileNotFoundError(f"Land mask file not found: {LAND_MASK_PATH}")

    with h5py.File(MASK_PATH, "r") as mask_file, h5py.File(LAND_MASK_PATH, "r") as land_file:
        y = mask_file["y"][:]
        x = mask_file["x"][:]
        land = land_file["land"][:].astype(bool)

        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
        land_overlay = np.where(land, 1.0, np.nan)

        route_x = []
        route_y = []
        for info in POINTS.values():
            iy, ix = info["yx"]
            route_y.append(float(y[iy]))
            route_x.append(float(x[ix]))

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(17, 10),
            constrained_layout=False,
            sharex=True,
            sharey=True,
        )
        fig.subplots_adjust(left=0.06, right=0.78, top=0.88, bottom=0.08, wspace=0.16, hspace=0.22)

        for ax, month in zip(axes.ravel(), MONTHS):
            active = mask_file[f"active_{month:02d}"][:].astype(bool)
            background = np.zeros((*active.shape, 3), dtype=float)
            background[:, :] = OCEAN_COLOR
            background[active] = ACTIVE_COLOR
            background[land] = LAND_COLOR

            ax.imshow(
                background,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                zorder=1,
            )

            ax.plot(
                route_x,
                route_y,
                color="#ffffff",
                linestyle="--",
                linewidth=1.5,
                alpha=0.95,
                zorder=3,
                label="Representative NSR corridor",
            )

            for name, info in POINTS.items():
                iy, ix = info["yx"]
                point_y = float(y[iy])
                point_x = float(x[ix])

                ax.scatter(
                    point_x,
                    point_y,
                    s=150,
                    c="#ffb000",
                    edgecolor="black",
                    linewidth=1.1,
                    zorder=5,
                )
                ax.text(
                    point_x,
                    point_y,
                    info["id"],
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#111111",
                    zorder=6,
                )

            ax.set_title(f"Active mask and NSR bottlenecks - {month:02d}")
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.grid(True, alpha=0.25)

        handles = [
            Patch(facecolor=ACTIVE_COLOR, label="Active sea-ice mask"),
            Patch(facecolor=OCEAN_COLOR, label="Ocean / non-active"),
            Patch(facecolor=LAND_COLOR, label="Land"),
            Line2D([0], [0], color="#ffffff", linestyle="--", linewidth=1.5, label="Representative NSR corridor"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#ffb000",
                markeredgecolor="black",
                markersize=9,
                label="Bottleneck point",
            ),
        ]
        labels = [handle.get_label() for handle in handles]

        for name, info in POINTS.items():
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=f"${info['id']}$",
                    color="none",
                    markerfacecolor="#ffb000",
                    markeredgecolor="black",
                    markersize=12,
                    label=name,
                )
            )
            labels.append(f"{info['id']} {name}")

        fig.legend(handles, labels, loc="center right", frameon=False)
        fig.suptitle(
            "Northern Sea Route bottleneck candidates on July-October active masks",
            fontsize=14,
            y=0.97,
        )

        fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
