import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import math


def extract_data(zip_path: str, output_path: str) -> None:
    """
    Exctract data to a specific path
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load the csv file and create the dataframe
    """
    return pd.read_csv(path)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot and save the correlation heatmap of all numerical features.
    """

    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))

    plt.imshow(corr_matrix, cmap="coolwarm")

    plt.colorbar()

    plt.xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns
    )

    plt.tight_layout()

    plt.savefig(
        "../assets/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import os


def scatter_with_target(df: pd.DataFrame, features: List[str]) -> None:
    """
    Plot scatter plots of multiple features against the target variable.
    """
    target = "median_house_value"

    for col in features:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[col], df[target], alpha=0.3)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.savefig(os.path.join("../assets", f"{col}_vs_{target}.png"), dpi=300)
        plt.show()


def plot_housing_scatter(df: pd.DataFrame, color_column: str) -> None:
    """
    Plot geographic distribution of California housing dataset.
    Points are plotted using longitude and latitude,
    and colored by a selected feature (default: median_house_value).
    """

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df[color_column],
        cmap="jet",
        alpha=0.4
    )

    plt.colorbar(scatter, label=color_column)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"California Housing Map colored by {color_column}")

    plt.savefig(f'../assets/{color_column}_scatter.png', dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 50) -> None:
    """
    Plot histogram for a selected feature and optionally save it.
    """

    plt.figure(figsize=(8, 5))

    df[column].hist(bins=bins)

    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.savefig(f'../assets/{column}_histrogram.png', dpi=300, bbox_inches="tight")

    plt.show()


def plot_multi_histograms(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """
    Plot histograms for multiple features and save the figure.
    """

    n_cols = 2
    n_rows = math.ceil(len(columns) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4 * n_rows)
    )

    if len(columns) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        df[column].hist(
            bins=50,
            ax=axes[i]
        )

        axes[i].set_title(f"{column} Distribution")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")

    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    plt.savefig(f"../assets/{name}.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_boxplot(df: pd.DataFrame, column: str) -> None:
    """
    Plot a boxplot for a selected feature and optionally save it.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    df.boxplot(column=column)
    plt.title(f"{column} Boxplot")
    plt.ylabel(column)

    plt.savefig(f'../assets/{column}_boxplot.png', dpi=300, bbox_inches="tight")

    plt.show()


def plot_multi_boxplots(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """
    Plot boxplots for multiple features and save the figure.
    """

    n_cols = 2
    n_rows = math.ceil(len(columns) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4 * n_rows)
    )

    if len(columns) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        df.boxplot(
            column=column,
            ax=axes[i]
        )

        axes[i].set_title(f"{column} Boxplot")
        axes[i].set_ylabel(column)

    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    plt.savefig(f"../assets/{name}.png", dpi=300, bbox_inches="tight")

    plt.show()