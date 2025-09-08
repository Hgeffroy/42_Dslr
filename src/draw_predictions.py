import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_path

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
HOUSE_COLORS = {
	"Gryffindor": "red",
	"Hufflepuff": "yellow",
	"Ravenclaw": "blue",
	"Slytherin": "green",
}


def load_predictions(csv_path: str) -> pd.DataFrame:
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"Predictions file not found: {csv_path}")
		
	df = pd.read_csv(csv_path)
	expected_cols = {"Index", "Hogwarts House"}
	if not expected_cols.issubset(df.columns):
		raise ValueError(f"CSV must contain columns {expected_cols}, got {set(df.columns)}")
	return df


def ensure_figures_dir():
	outdir = get_path("figures")
	os.makedirs(outdir, exist_ok=True)
	return outdir


def build_parser():
	parser = argparse.ArgumentParser(description="Plot predictions made by LogisticModel.")
	parser.add_argument(
		"-p", "--predictions",
		type=str,
		default=get_path("predictions/predictions.csv"),
		help="Path to predictions CSV (default: predictions/predictions.csv)"
	)
	return parser


def plot_bar_counts(counts: pd.Series, outdir: str):
	colors = [HOUSE_COLORS[h] for h in counts.index]

	plt.figure()
	bars = plt.bar(counts.index, counts.values, color=colors)
	plt.title("Predictions per House")
	plt.xlabel("House")
	plt.ylabel("Count")

	for bar in bars:
		height = bar.get_height()
		if height > 0:
			plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}",
					 ha="center", va="bottom")

	out = os.path.join(outdir, "predictions_bar.png")
	plt.tight_layout()
	plt.savefig(out, dpi=150)
	plt.close()

	return out


def plot_pie_counts(counts: pd.Series, outdir: str):
	colors = [HOUSE_COLORS[h] for h in counts.index]

	plt.figure()
	plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
		 startangle=90, colors=colors)
	plt.title("Predictions Distribution")
	plt.axis("equal") # circle
	out = os.path.join(outdir, "predictions_pie.png")
	plt.tight_layout()
	plt.savefig(out, dpi=150)
	plt.close()
	return out


def main():
	args = build_parser().parse_args()
	df = load_predictions(args.predictions)
	outdir = ensure_figures_dir()

	counts = df["Hogwarts House"].value_counts()
	
	bar_path = plot_bar_counts(counts, outdir)
	pie_path = plot_pie_counts(counts, outdir)

	print("✅ Figures saved:")
	print(f" - {bar_path}")
	print(f" - {pie_path}")

if __name__ == "__main__":
	main()